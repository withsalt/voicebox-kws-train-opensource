"""第八步导出：把第七步训练出的「最优解」checkpoint 导出为流式 ONNX 模型 + 部署资源。

训练产物是 modelscope 格式 checkpoint（键名 `backbone.*` + `global_cmvn.*`）；参考实现
（voicebox/kws-xiaoyun-onnx-export/02_export_streaming_onnx.py）用的是 funasr 模型（键名
`encoder.*`）。二者是**同一个 4 层 cFSMN 网络**，权重一一对应。本模块：

  1) 用 funasr `AutoModel` 在基线模型目录构出 encoder（拿到正确的网络结构）；
  2) 把训练 checkpoint 的键**确定性映射**为 funasr encoder 键并 `load_state_dict(strict)`；
  3) 复用参考的 `StreamingFsmnKws` wrapper（精确 lookahead、每层显式 cache），导出 ONNX；
  4) 导出前后两道自检（逐帧流式 vs 整段前向；onnxruntime vs torch），拷贝部署资源、写 meta。

模块顶层只用标准库的**纯函数**（remap_state_dict_keys / find_best_checkpoint /
keyword_token_ids / read_backbone_dims / build_meta / summarize，可单测、不依赖 funasr/onnx）；
torch/funasr/onnx/onnxruntime 等重依赖在 export_onnx / 工厂里**延迟导入**。
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


# --------------------------- 纯逻辑（可单测，无重依赖） ---------------------------

# modelscope FSMN 的每个 fsmn 层是 Sequential[proj, fsmn_block, affine]，键用下标 .0/.1/.2；
# funasr 同一层是带名子模块 linear/fsmn_block/affine。下标→名字的确定性映射。
_FSMN_RE = re.compile(r"fsmn\.(\d+)\.(\d+)\.(.*)")
_FSMN_SUB = {"0": "linear", "1": "fsmn_block", "2": "affine"}


def remap_state_dict_keys(state_dict: dict) -> dict:
    """modelscope checkpoint 键 → funasr encoder 键（load 进 AutoModel(...).model.encoder）。

    - 丢弃 `global_cmvn.*`（CMVN 在前端/am.mvn，finetune 时冻结，不属于 encoder）；
    - `backbone.` 前缀剥掉（直接 load 进 encoder，不含 encoder. 前缀）；
    - `fsmn.{i}.0/1/2.<rest>` → `fsmn.{i}.linear/fsmn_block/affine.<rest>`。
    其余键（in_linear1/2、out_linear1/2）原样保留（仅剥 backbone. 前缀）。
    """
    out: dict = {}
    for k, v in state_dict.items():
        if k.startswith("global_cmvn."):
            continue
        if not k.startswith("backbone."):
            raise ValueError(f"非预期的 checkpoint 键（既非 backbone. 也非 global_cmvn.）：{k}")
        kk = k[len("backbone."):]
        m = _FSMN_RE.match(kk)
        if m:
            i, idx, rest = m.group(1), m.group(2), m.group(3)
            sub = _FSMN_SUB.get(idx)
            if sub is None:
                raise ValueError(f"非预期的 fsmn 子下标：{k}")
            kk = f"fsmn.{i}.{sub}.{rest}"
        out[kk] = v
    return out


def _read_yaml_scalars(path: Path) -> dict:
    """读训练器写的极简 yaml（每行 `key: value`，如 cv_loss/epoch/lr）。value 能转 float 就转。"""
    d: dict = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key, val = key.strip(), val.strip()
        if not key:
            continue
        try:
            d[key] = float(val)
        except ValueError:
            d[key] = val
    return d


def find_best_checkpoint(training_dir, num_avg: int = 10):
    """在训练产物目录里挑「最优解」checkpoint。

    优先级：
      1) `avg_{num_avg}.pt`（评测所用的 cv_loss 选优 best-N 平均模型，首选）；
      2) 任意 `avg_*.pt`（取 N 最大者）；
      3) 回退：扫 `<e>.yaml` 取 cv_loss 最小的 `<e>.pt`（早停过早、未评测时无 avg）。

    Returns: (path: Path, kind: str, info: dict)。kind ∈ {"avg", "best_epoch"}。
    找不到任何可用 checkpoint 抛 FileNotFoundError。
    """
    d = Path(training_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"训练产物目录不存在：{d}")

    preferred = d / f"avg_{num_avg}.pt"
    if preferred.exists():
        return preferred, "avg", {"num_avg": num_avg}

    avg_cands = []
    for p in d.glob("avg_*.pt"):
        m = re.match(r"avg_(\d+)\.pt$", p.name)
        if m:
            avg_cands.append((int(m.group(1)), p))
    if avg_cands:
        n, p = max(avg_cands, key=lambda x: x[0])
        return p, "avg", {"num_avg": n}

    # 回退：最优单 epoch（cv_loss 最小）
    best = None  # (cv_loss, epoch, path)
    for y in d.glob("*.yaml"):
        m = re.match(r"(\d+)\.yaml$", y.name)
        if not m:
            continue
        pt = d / f"{m.group(1)}.pt"
        if not pt.exists():
            continue
        info = _read_yaml_scalars(y)
        cv = info.get("cv_loss")
        if not isinstance(cv, (int, float)):
            continue
        if best is None or cv < best[0]:
            best = (float(cv), int(m.group(1)), pt)
    if best is not None:
        return best[2], "best_epoch", {"cv_loss": best[0], "epoch": best[1]}

    raise FileNotFoundError(
        f"{d} 下没有可导出的 checkpoint（既无 avg_*.pt，也无带 cv_loss 的 <e>.yaml/<e>.pt）。"
        "请先完成第七步训练（如需 avg 模型，确保评测开启）。")


def read_backbone_dims(model_dir) -> dict:
    """从基线模型 configuration.json 读 FSMN 结构维度（跟随基线模型，避免与配置漂移）。"""
    conf = json.loads((Path(model_dir) / "configuration.json").read_text(encoding="utf-8"))
    m = conf["model"]
    b = m["backbone"]
    dims = {
        "input_dim": int(m["input_dim"]),
        "output_dim": int(m["output_dim"]),
        "num_layers": int(b["num_layers"]),
        "proj_dim": int(b["proj_dim"]),
        "left_order": int(b["left_order"]),
        "right_order": int(b["right_order"]),
        "linear_dim": int(b.get("linear_dim", 0)),
    }
    dims["cache_len"] = (dims["left_order"] - 1) + dims["right_order"]
    dims["output_delay_frames"] = dims["num_layers"] * dims["right_order"]
    dims["warmup_frames"] = (dims["num_layers"] - 1) * (dims["left_order"] - 1) \
        + dims["output_delay_frames"]
    return dims


def _read_tokens(tokens_path) -> list[str]:
    return [line.rstrip("\n") for line in Path(tokens_path).read_text(encoding="utf-8").splitlines()]


def _read_lexicon(lexicon_path) -> dict:
    """读 funasr seg_dict（每行 `word\\ttok1 tok2 ...`）为 {word: [tokens]}。"""
    d: dict = {}
    for line in Path(lexicon_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("\t") if "\t" in line else line.split(None, 1)
        if len(parts) < 2:
            continue
        d[parts[0]] = parts[1].split()
    return d


def keyword_token_ids(wake: str, tokens_path, lexicon_path=None):
    """把唤醒词转成 token id 序列（token id = tokens.txt 的 0 基行号）。

    逐字查 tokens；缺字时（如英文/符号）尝试用 lexicon 展开成子 token 再查；仍缺则记入 OOV。
    Returns: (ids: list[int], oov: list[str])。
    """
    tokens = _read_tokens(tokens_path)
    tok2id = {t: i for i, t in enumerate(tokens)}
    lex = _read_lexicon(lexicon_path) if lexicon_path and Path(lexicon_path).exists() else {}
    ids: list[int] = []
    oov: list[str] = []
    for ch in wake:
        if ch in tok2id:
            ids.append(tok2id[ch])
        elif ch in lex and all(t in tok2id for t in lex[ch]):
            ids.extend(tok2id[t] for t in lex[ch])
        else:
            oov.append(ch)
    return ids, oov


def build_meta(*, wake, ids, oov, dims, cmvn, source_checkpoint, checkpoint_kind, opset):
    """拼 kws_meta.json（部署/文档用：特征管线 + 模型维度 + 延迟/预热 + 关键词 token）。"""
    return {
        "keyword": wake,
        "keyword_token_ids": list(ids),
        "keyword_oov_chars": list(oov),
        "sample_rate": 16000,
        "fbank": {"num_mel_bins": 80, "frame_length_ms": 25, "frame_shift_ms": 10,
                  "window": "hamming", "dither": 0.0, "snip_edges": True,
                  "waveform_scale": 32768.0},
        "lfr": {"m": 5, "n": 3},
        "cmvn": cmvn,  # {"add_shift": [...400], "rescale": [...400]} 或 None
        "model": {
            "input_dim": dims["input_dim"], "output_dim": dims["output_dim"],
            "num_layers": dims["num_layers"], "proj_dim": dims["proj_dim"],
            "left_order": dims["left_order"], "right_order": dims["right_order"],
            "cache_len": dims["cache_len"],
            "output_delay_frames": dims["output_delay_frames"],
            "warmup_frames": dims["warmup_frames"], "blank_id": 0,
        },
        "source_checkpoint": str(source_checkpoint),
        "checkpoint_kind": checkpoint_kind,
        "opset": opset,
    }


def summarize(m: dict) -> str:
    """导出结果的人读摘要。"""
    lines = [
        f"导出汇总（唤醒词={m['keyword']}）：",
        f"  源 checkpoint：{m.get('checkpoint_kind')}  {m.get('source_checkpoint')}",
        f"  ONNX：{m.get('onnx_path')}  ({m.get('onnx_size_mb', 0):.1f} MB, 单文件)",
    ]
    sc = m.get("self_check_diff")
    ort = m.get("ort_diff")
    if sc is not None:
        lines.append(f"  自检：流式 vs 整段 max diff {sc:.2e}"
                     + (f"  ｜ onnxruntime vs torch {ort:.2e}" if ort is not None else "  ｜ onnxruntime 校验：跳过"))
    ids = m.get("keyword_token_ids")
    if ids is not None:
        lines.append(f"  关键词 token ids：{ids}"
                     + (f"  ⚠ OOV：{m.get('keyword_oov_chars')}" if m.get("keyword_oov_chars") else ""))
    res = m.get("resources")
    if res:
        lines.append(f"  资源：{', '.join(res)}")
    return "\n".join(lines)


# --------------------------- 流式 wrapper（延迟导入 torch） ---------------------------

def _make_streaming_class(dims: dict):
    """构造流式 FSMN-KWS wrapper 类（torch 在此延迟导入）。

    复用 voicebox/kws-xiaoyun-onnx-export/02_export_streaming_onnx.py 的精确 lookahead 方案：
    每层维护 (lorder-1)+rorder 帧投影 cache，稳态输出与整段前向逐位一致。
    结构常量 lorder/rorder/proj 从 configuration.json 读出；本 4 层 cFSMN 用显式 cache0..3
    （与基线模型一致；若 num_layers 变化需同步扩展 forward 签名）。
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    if dims["num_layers"] != 4:
        raise ValueError(
            f"当前导出 wrapper 仅支持 4 层 cFSMN（基线 speech_charctc_kws_phone），"
            f"但 configuration.json num_layers={dims['num_layers']}。"
            "若基线网络层数变化，请同步扩展 StreamingFsmnKws.forward 的 cache 入参。")

    LORDER = dims["left_order"]
    RORDER = dims["right_order"]
    CACHE_LEN = dims["cache_len"]

    class StreamingFsmnKws(nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.in_linear1 = encoder.in_linear1
            self.in_linear2 = encoder.in_linear2
            self.blocks = nn.ModuleList(list(encoder.fsmn))
            self.out_linear1 = encoder.out_linear1
            self.out_linear2 = encoder.out_linear2

        @staticmethod
        def _block_forward(block, x, cache):
            p = block.linear(x)                       # [1, T, proj]
            b = p.transpose(1, 2).unsqueeze(-1)       # [1, proj, T, 1]
            buf = torch.cat([cache, b], dim=2)        # [1, proj, T+cache_len, 1]
            new_cache = buf[:, :, -CACHE_LEN:, :]
            center = buf[:, :, LORDER - 1:-RORDER, :]                    # T frames
            y_left = block.fsmn_block.conv_left(buf[:, :, :-RORDER, :])  # T frames
            y_right = block.fsmn_block.conv_right(buf[:, :, LORDER:, :])  # T frames
            out = center + y_left + y_right
            out = out.squeeze(-1).transpose(1, 2)     # [1, T, proj]
            out = F.relu(block.affine(out))           # [1, T, linear]
            return out, new_cache

        def forward(self, feats, cache0, cache1, cache2, cache3):
            x = F.relu(self.in_linear2(self.in_linear1(feats)))
            caches = [cache0, cache1, cache2, cache3]
            new_caches = []
            for block, cache in zip(self.blocks, caches):
                x, nc = self._block_forward(block, x, cache)
                new_caches.append(nc)
            logits = self.out_linear2(self.out_linear1(x))
            probs = torch.softmax(logits, dim=-1)
            return (probs, *new_caches)

    return StreamingFsmnKws


def _parse_cmvn(am_mvn_path):
    """解析 kaldi 文本 CMVN（funasr load_cmvn）→ (add_shift[400], rescale[400])。"""
    from funasr.frontends.wav_frontend import load_cmvn
    cmvn = load_cmvn(str(am_mvn_path))  # [2, dim]: row0 add_shift(=负均值), row1 rescale(=逆标准差)
    return cmvn[0].tolist(), cmvn[1].tolist()


# --------------------------- 导出主流程（延迟导入重依赖） ---------------------------

def export_onnx(wake, dataset_dir, ecfg, *, model_dir, config_path="config.toml"):
    """编排：构 encoder → load 最优 checkpoint → 流式 wrapper → 自检 → 导出 ONNX → 校验 → 拷资源。

    Args:
        wake: 唤醒词（如 小木小木）。
        dataset_dir: 数据集目录（含 <work_subdir>/ 训练产物）。
        ecfg: ExportConfig。
        model_dir: 基线模型目录（speech_charctc_kws_phone；提供 AutoModel 构图 + tokens/lexicon/am.mvn）。
        config_path: config.toml 路径（仅记录用）。
    Returns:
        manifest dict（onnx 路径/大小、源 checkpoint、token ids、自检/ort 误差、资源清单、outputs）。
    """
    import numpy as np
    import torch

    dataset_dir = Path(dataset_dir)
    model_dir = Path(model_dir)
    training_dir = dataset_dir / ecfg.work_subdir
    out_dir = dataset_dir / ecfg.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = read_backbone_dims(model_dir)
    NUM_LAYERS = dims["num_layers"]
    PROJ_DIM = dims["proj_dim"]
    CACHE_LEN = dims["cache_len"]
    delay = dims["output_delay_frames"]
    warmup = dims["warmup_frames"]

    ckpt_path, ckpt_kind, ckpt_info = find_best_checkpoint(training_dir, ecfg.num_avg)
    print(f"  最优 checkpoint：{ckpt_kind}  {ckpt_path}"
          + (f"  {ckpt_info}" if ckpt_info else ""))

    # 1) funasr AutoModel 构出 encoder（拿正确网络结构）
    from funasr import AutoModel
    am = AutoModel(model=str(model_dir), keywords=wake, device="cpu", disable_update=True)
    encoder = am.model.encoder.eval()

    # 2) 训练 checkpoint → 映射键 → strict load 进 encoder
    ck = torch.load(str(ckpt_path), map_location="cpu")
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    remapped = remap_state_dict_keys(sd)
    missing, unexpected = encoder.load_state_dict(remapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"encoder 权重加载不严格匹配：missing={list(missing)} unexpected={list(unexpected)}。"
            "训练模型与基线网络结构不一致？")

    # 3) CMVN 一致性软校验：checkpoint 的 global_cmvn 应等于基线 am.mvn（finetune 冻结，不应漂移）
    am_mvn = model_dir / "funasr" / "am.mvn.dim80_l2r2"
    add_shift, rescale = _parse_cmvn(am_mvn)
    gc_mean = sd.get("global_cmvn.mean")
    gc_istd = sd.get("global_cmvn.istd")
    if gc_mean is not None and gc_istd is not None:
        d_shift = (gc_mean.cpu() + torch.tensor(add_shift)).abs().max().item()  # add_shift = -mean
        d_scale = (gc_istd.cpu() - torch.tensor(rescale)).abs().max().item()
        if max(d_shift, d_scale) > 1e-2:
            print(f"  ⚠ CMVN 与基线 am.mvn 存在差异（add_shift {d_shift:.2e} / rescale {d_scale:.2e}）；"
                  "导出仍用基线 am.mvn（请确认基线模型目录与训练时一致）。")

    # 4) 流式 wrapper + 自检（逐帧流式 vs 整段前向，稳态逐位一致）
    StreamingFsmnKws = _make_streaming_class(dims)
    model = StreamingFsmnKws(encoder).eval()
    torch.manual_seed(0)
    T = 50
    feats = torch.randn(1, T, dims["input_dim"])
    with torch.no_grad():
        full_probs = torch.softmax(encoder(feats), dim=-1)
        caches = [torch.zeros(1, PROJ_DIM, CACHE_LEN, 1) for _ in range(NUM_LAYERS)]
        outs = []
        for t in range(T):
            res = model(feats[:, t:t + 1, :], *caches)
            outs.append(res[0])
            caches = list(res[1:])
        stream_probs = torch.cat(outs, dim=1)
    self_diff = (stream_probs[0, warmup:T, :]
                 - full_probs[0, warmup - delay:T - delay, :]).abs().max().item()
    print(f"  自检：流式 vs 整段 max diff {self_diff:.3e}（阈 {ecfg.self_check_tol:g}）")
    if self_diff >= ecfg.self_check_tol:
        raise RuntimeError(f"流式 wrapper 与整段前向不一致（{self_diff:.3e} ≥ {ecfg.self_check_tol:g}）。")

    # 5) 导出 ONNX（T 轴动态）；新版 torch dynamo 会外置 .onnx.data，合并为单文件
    onnx_path = out_dir / ecfg.onnx_name
    cache_inputs = [f"cache{i}" for i in range(NUM_LAYERS)]
    cache_outputs = [f"new_cache{i}" for i in range(NUM_LAYERS)]
    torch.onnx.export(
        model,
        (feats[:, :3, :], *[torch.zeros(1, PROJ_DIM, CACHE_LEN, 1) for _ in range(NUM_LAYERS)]),
        str(onnx_path),
        input_names=["feats", *cache_inputs],
        output_names=["probs", *cache_outputs],
        dynamic_axes={"feats": {1: "T"}, "probs": {1: "T"}},
        opset_version=ecfg.opset,
    )
    import onnx
    onnx_model = onnx.load(str(onnx_path))  # 自动加载外部数据
    data_file = str(onnx_path) + ".data"
    if Path(data_file).exists():
        Path(data_file).unlink()
    onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=False)
    onnx_size_mb = onnx_path.stat().st_size / 1e6
    print(f"  导出：{onnx_path}（{onnx_size_mb:.1f} MB, 单文件）")

    # 6) onnxruntime 不规则分块 vs torch 流式 校验
    ort_diff = None
    if ecfg.verify:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        caches = [np.zeros((1, PROJ_DIM, CACHE_LEN, 1), dtype=np.float32) for _ in range(NUM_LAYERS)]
        outs = []
        feats_np = feats.numpy()
        for s in range(0, T, 7):  # 不规则分块（与分块大小无关性）
            chunk = feats_np[:, s:s + 7, :]
            res = sess.run(None, {"feats": chunk,
                                  **{f"cache{i}": caches[i] for i in range(NUM_LAYERS)}})
            outs.append(res[0])
            caches = res[1:]
        ort_probs = np.concatenate(outs, axis=1)
        ort_diff = float(np.abs(ort_probs - stream_probs.numpy()).max())
        print(f"  校验：onnxruntime vs torch max diff {ort_diff:.3e}（阈 {ecfg.ort_tol:g}）")
        if ort_diff >= ecfg.ort_tol:
            raise RuntimeError(f"onnxruntime 与 torch 流式不一致（{ort_diff:.3e} ≥ {ecfg.ort_tol:g}）。")

    # 7) 拷部署资源 + 写 meta
    resources = []
    for src_name, dst_name in (("tokens_2599.txt", "tokens.txt"),
                               ("lexicon.txt", "lexicon.txt"),
                               ("am.mvn.dim80_l2r2", "am.mvn")):
        src = model_dir / "funasr" / src_name
        if src.exists():
            shutil.copy(str(src), str(out_dir / dst_name))
            resources.append(dst_name)

    # 训练第七步自动选阈值产出的部署 keywords.json（含 threshold1）：随导出资源一起打包，
    # 使 onnx/ 成为可直接部署的完整资源包。
    kw_json = training_dir / "test_dir" / "keywords.json"
    if kw_json.exists():
        shutil.copy(str(kw_json), str(out_dir / "keywords.json"))
        resources.append("keywords.json")

    ids, oov = keyword_token_ids(wake, out_dir / "tokens.txt",
                                 (out_dir / "lexicon.txt") if (out_dir / "lexicon.txt").exists() else None)
    if oov:
        print(f"  ⚠ 关键词含 OOV 字（不在 token 表）：{oov}；这些字无法被该模型唤醒。")
    meta = build_meta(wake=wake, ids=ids, oov=oov, dims=dims,
                      cmvn={"add_shift": add_shift, "rescale": rescale},
                      source_checkpoint=ckpt_path, checkpoint_kind=ckpt_kind, opset=ecfg.opset)
    meta_path = out_dir / "kws_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    resources.append("kws_meta.json")

    return {
        "keyword": wake,
        "onnx_path": str(onnx_path),
        "onnx_size_mb": onnx_size_mb,
        "source_checkpoint": str(ckpt_path),
        "checkpoint_kind": ckpt_kind,
        "self_check_diff": self_diff,
        "ort_diff": ort_diff,
        "keyword_token_ids": ids,
        "keyword_oov_chars": oov,
        "out_dir": str(out_dir),
        "resources": resources,
        "outputs": {"onnx": str(onnx_path), "meta": str(meta_path), "out_dir": str(out_dir)},
    }
