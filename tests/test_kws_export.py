"""kws_export 自测（纯逻辑，不跑 onnx/funasr/torch、不导出）。

运行：python tests/test_kws_export.py
不依赖 pytest——纯 assert + 计数，结尾打印 PASS/FAIL 汇总。
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kws_export import (ExportConfig, build_meta, find_best_checkpoint,  # noqa: E402
                        keyword_token_ids, load_export_config,
                        read_backbone_dims, remap_state_dict_keys)

_passed = 0
_failed = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global _passed, _failed
    if cond:
        _passed += 1
        print(f"  ✓ {name}")
    else:
        _failed += 1
        print(f"  ✗ {name}  {detail}")


def test_remap():
    print("[REMAP]")
    sd = {
        "global_cmvn.mean": "drop",
        "global_cmvn.istd": "drop",
        "backbone.in_linear1.linear.weight": 1,
        "backbone.in_linear2.linear.bias": 2,
        "backbone.fsmn.0.0.linear.weight": 3,       # proj
        "backbone.fsmn.1.1.conv_left.weight": 4,    # fsmn_block
        "backbone.fsmn.2.1.conv_right.weight": 5,
        "backbone.fsmn.3.2.linear.bias": 6,         # affine
        "backbone.out_linear2.linear.weight": 7,
    }
    out = remap_state_dict_keys(sd)
    check("丢弃 global_cmvn", not any(k.startswith("global_cmvn") for k in out))
    check("剥 backbone 前缀", "in_linear1.linear.weight" in out)
    check("proj .0->linear", out.get("fsmn.0.linear.linear.weight") == 3)
    check("conv .1->fsmn_block(left)", out.get("fsmn.1.fsmn_block.conv_left.weight") == 4)
    check("conv .1->fsmn_block(right)", out.get("fsmn.2.fsmn_block.conv_right.weight") == 5)
    check("affine .2->affine", out.get("fsmn.3.affine.linear.bias") == 6)
    check("out_linear2 原样", out.get("out_linear2.linear.weight") == 7)
    check("键数=输入-2(cmvn)", len(out) == len(sd) - 2, str(len(out)))
    # 非预期键应报错
    try:
        remap_state_dict_keys({"weird.key": 1})
        check("非预期键报错", False)
    except ValueError:
        check("非预期键报错", True)


def test_find_best_checkpoint():
    print("[FIND-BEST]")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # 1) 优先 avg_<num_avg>
        (td / "avg_10.pt").write_bytes(b"x")
        (td / "avg_5.pt").write_bytes(b"x")
        p, kind, info = find_best_checkpoint(td, num_avg=10)
        check("优先 avg_10", p.name == "avg_10.pt" and kind == "avg", f"{p.name}/{kind}")
        # 2) 无 avg_10 时取最大 N 的 avg_*
        (td / "avg_10.pt").unlink()
        p, kind, info = find_best_checkpoint(td, num_avg=10)
        check("回退最大 avg_5", p.name == "avg_5.pt" and info.get("num_avg") == 5, p.name)
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        # 3) 无 avg：按 cv_loss 最小的 epoch
        for e, cv in [(0, 30.0), (1, 22.5), (2, 25.1)]:
            (td / f"{e}.pt").write_bytes(b"x")
            (td / f"{e}.yaml").write_text(f"cv_loss: {cv}\nepoch: {e}\nlr: 0.0001\n", encoding="utf-8")
        (td / "init.pt").write_bytes(b"x")  # 干扰项，不应被选
        p, kind, info = find_best_checkpoint(td, num_avg=10)
        check("回退最优 epoch=1", p.name == "1.pt" and kind == "best_epoch", f"{p.name}/{kind}")
        check("best_epoch cv_loss", abs(info.get("cv_loss", 0) - 22.5) < 1e-9, str(info))
    with tempfile.TemporaryDirectory() as d:
        # 4) 空目录报错
        try:
            find_best_checkpoint(Path(d), num_avg=10)
            check("空目录报错", False)
        except FileNotFoundError:
            check("空目录报错", True)


def test_keyword_token_ids():
    print("[TOKEN-IDS]")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        tokens = td / "tokens.txt"
        tokens.write_text("\n".join(["<blank>", "<unk>", "小", "木", "a", "b"]) + "\n",
                          encoding="utf-8")
        ids, oov = keyword_token_ids("小木", tokens)
        check("命中 token id", ids == [2, 3] and oov == [], str((ids, oov)))
        ids, oov = keyword_token_ids("小Z", tokens)
        check("OOV 记录", ids == [2] and oov == ["Z"], str((ids, oov)))
        # lexicon 展开：词 'Z' -> 'a b'
        lex = td / "lexicon.txt"
        lex.write_text("Z\ta b\n", encoding="utf-8")
        ids, oov = keyword_token_ids("小Z", tokens, lex)
        check("lexicon 展开", ids == [2, 4, 5] and oov == [], str((ids, oov)))


def test_read_backbone_dims():
    print("[BACKBONE-DIMS]")
    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        (td / "configuration.json").write_text(json.dumps({
            "model": {"input_dim": 400, "output_dim": 2599, "backbone": {
                "num_layers": 4, "proj_dim": 128, "left_order": 10, "right_order": 2,
                "linear_dim": 250}}}), encoding="utf-8")
        dims = read_backbone_dims(td)
        check("input/output", dims["input_dim"] == 400 and dims["output_dim"] == 2599)
        check("cache_len=(10-1)+2=11", dims["cache_len"] == 11, str(dims["cache_len"]))
        check("delay=4*2=8", dims["output_delay_frames"] == 8, str(dims["output_delay_frames"]))
        check("warmup=3*9+8=35", dims["warmup_frames"] == 35, str(dims["warmup_frames"]))


def test_build_meta():
    print("[BUILD-META]")
    dims = {"input_dim": 400, "output_dim": 2599, "num_layers": 4, "proj_dim": 128,
            "left_order": 10, "right_order": 2, "cache_len": 11,
            "output_delay_frames": 8, "warmup_frames": 35}
    m = build_meta(wake="小木小木", ids=[1462, 1774, 1462, 1774], oov=[], dims=dims,
                   cmvn={"add_shift": [0.0] * 400, "rescale": [1.0] * 400},
                   source_checkpoint="/x/avg_10.pt", checkpoint_kind="avg", opset=17)
    check("keyword", m["keyword"] == "小木小木")
    check("token ids", m["keyword_token_ids"] == [1462, 1774, 1462, 1774])
    check("model 维度齐全", m["model"]["cache_len"] == 11 and m["model"]["blank_id"] == 0)
    check("delay/warmup", m["model"]["output_delay_frames"] == 8 and m["model"]["warmup_frames"] == 35)
    check("源 checkpoint", m["source_checkpoint"].endswith("avg_10.pt") and m["checkpoint_kind"] == "avg")
    check("cmvn 维度", len(m["cmvn"]["rescale"]) == 400)
    # 可 JSON 序列化
    json.dumps(m, ensure_ascii=False)
    check("可 JSON 序列化", True)


def test_config():
    print("[CONFIG]")
    c = ExportConfig()
    check("默认 enabled", c.enabled is True)
    check("默认 opset 17", c.opset == 17)
    check("默认 verify", c.verify is True)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "config.toml"
        p.write_text("[export]\nopset = 18\nverify = false\nonnx_name = \"m.onnx\"\n",
                     encoding="utf-8")
        c2 = load_export_config(str(p))
        check("解析 opset", c2.opset == 18)
        check("解析 verify=false", c2.verify is False)
        check("解析 onnx_name", c2.onnx_name == "m.onnx")
        # work_subdir/num_avg 跟随 [train]（未在 [export] 显式给出时）
        p2 = Path(d) / "config2.toml"
        p2.write_text("[train]\nwork_subdir = \"t2\"\nnum_avg = 7\n[export]\nopset = 17\n",
                      encoding="utf-8")
        c3 = load_export_config(str(p2))
        check("work_subdir 跟随 train", c3.work_subdir == "t2", c3.work_subdir)
        check("num_avg 跟随 train", c3.num_avg == 7, str(c3.num_avg))
        # [export] 显式覆盖优先于 [train]
        p3 = Path(d) / "config3.toml"
        p3.write_text("[train]\nnum_avg = 7\n[export]\nnum_avg = 3\n", encoding="utf-8")
        c4 = load_export_config(str(p3))
        check("export 覆盖优先", c4.num_avg == 3, str(c4.num_avg))
    c5 = load_export_config(str(Path(tempfile.gettempdir()) / "no_such_export.toml"))
    check("缺配置回退默认", c5.opset == ExportConfig().opset)


def main() -> int:
    print("==== kws_export 自测 ====")
    test_remap()
    test_find_best_checkpoint()
    test_keyword_token_ids()
    test_read_backbone_dims()
    test_build_meta()
    test_config()
    print(f"\n==== 结果：PASS={_passed}  FAIL={_failed} ====")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
