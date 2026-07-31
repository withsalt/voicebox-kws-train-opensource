"""自动选阈值 + 回写部署 keywords.json（第七步评测后）。

modelscope/funasr 的 evaluate 末尾会调用 compute_det，把**全量 DET 扫描**写到
`test_dir/stats.<关键词>.txt`，每行四列：

    threshold  recall(true_detect_rate)  fa_rate  fa_per_hour

阈值从 0→1（step 0.001）；随阈值升高，召回↓、误唤醒↓。本模块：
  1) 解析该 stats 文件；
  2) 在「误唤醒/小时 ≤ 目标」的前提下取**召回最高**的工作点（达不到目标则回退到误唤醒最低点）；
  3) 把选中的 threshold1 写进部署用 `keywords.json`（沿用基线 kwsbp_resource 模板的其余字段）。

纯逻辑、仅依赖标准库，便于单测（不跑训练/评测）。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

# 基线 runtime/kwsbp_resource/keywords.json 缺失时的兜底模板（字段对齐 speech_charctc_kws_phone）。
_DEFAULT_KEYWORD_ENTRY = {
    "name": "",
    "type": "wakeup",
    "activation": True,
    "is_main": True,
    "lm_boost": 0.0,
    "am_boost": 0.0,
    "threshold1": 0.0,
    "threshold2": -1.0,
    "subseg_threshold": -0.6,
    "high_threshold": 90.0,
    "min_dur": 0.25,
    "max_dur": 2.5,
    "cc_name": "commoncc",
}


def parse_stats_file(path) -> list[dict]:
    """解析 compute_det 产出的 stats 文件，返回按 threshold 升序的工作点列表。

    每个元素：{threshold, recall, fa_rate, fa_per_hour}。忽略空行/列数不足的行。
    """
    rows: list[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            threshold, recall, fa_rate, fa_per_hour = (float(x) for x in parts[:4])
        except ValueError:
            continue
        rows.append({
            "threshold": threshold, "recall": recall,
            "fa_rate": fa_rate, "fa_per_hour": fa_per_hour,
        })
    rows.sort(key=lambda r: r["threshold"])
    return rows


def select_operating_point(
    rows: list[dict], *, target_fa_per_hour: float, min_recall: float = 0.0
) -> dict:
    """在「误唤醒/小时 ≤ target」前提下选召回最高的工作点。

    返回 {threshold, recall, fa_rate, fa_per_hour, met_target, low_recall, note}。
      - 有满足误唤醒预算的点：取其中**召回最高**（并列时取误唤醒更低、再取阈值更低）；met_target=True。
      - 一个都不满足（即便阈值=1 仍超预算）：回退到**误唤醒/小时最低**的点；met_target=False。
      - low_recall：所选点召回 < min_recall（min_recall>0 时才有意义），用于提示“目标过严”。
    rows 为空时抛 ValueError。
    """
    if not rows:
        raise ValueError("stats 工作点为空，无法选阈值。")

    candidates = [r for r in rows if r["fa_per_hour"] <= target_fa_per_hour]
    if candidates:
        best = max(candidates, key=lambda r: (r["recall"], -r["fa_per_hour"], -r["threshold"]))
        met = True
        note = f"满足误唤醒预算 ≤{target_fa_per_hour:g}/h 的最高召回点"
    else:
        # 没有点满足预算：退而求其次，选误唤醒最低（并列取召回更高）的点。
        best = min(rows, key=lambda r: (r["fa_per_hour"], -r["recall"]))
        met = False
        note = (f"无工作点满足误唤醒 ≤{target_fa_per_hour:g}/h，回退到误唤醒最低点"
                f"（{best['fa_per_hour']:.3f}/h）")

    low_recall = min_recall > 0.0 and best["recall"] < min_recall
    if low_recall:
        note += f"；注意召回 {best['recall']*100:.1f}% < 目标下限 {min_recall*100:.1f}%"

    return {
        "threshold": best["threshold"], "recall": best["recall"],
        "fa_rate": best["fa_rate"], "fa_per_hour": best["fa_per_hour"],
        "met_target": met, "low_recall": low_recall, "note": note,
    }


def build_keywords_json(keyword: str, threshold1: float, *, template_path=None) -> dict:
    """构造部署用 keywords.json（dict）。

    优先沿用基线 `runtime/kwsbp_resource/keywords.json` 模板的其余字段（threshold2/subseg_threshold/
    cc_name 等），仅覆盖 name 与 threshold1；模板缺失时用内置兜底字段。name 用原唤醒词（不空格分隔，
    与基线 kwsbp_resource 一致）。
    """
    entry = None
    if template_path is not None:
        tp = Path(template_path)
        if tp.exists():
            try:
                tmpl = json.loads(tp.read_text(encoding="utf-8"))
                wl = tmpl.get("word_list") or []
                if wl and isinstance(wl[0], dict):
                    entry = copy.deepcopy(wl[0])
            except (json.JSONDecodeError, OSError):
                entry = None
    if entry is None:
        entry = copy.deepcopy(_DEFAULT_KEYWORD_ENTRY)

    entry["name"] = keyword
    entry["threshold1"] = round(float(threshold1), 6)
    return {"word_list": [entry]}


def write_keywords_json(out_path, keyword: str, threshold1: float, *, template_path=None) -> Path:
    """生成 keywords.json 并落盘，返回路径。"""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = build_keywords_json(keyword, threshold1, template_path=template_path)
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=4), encoding="utf-8")
    return out_path


def select_and_write(
    stats_path, *, keyword: str, out_path, target_fa_per_hour: float,
    min_recall: float = 0.0, template_path=None,
) -> dict:
    """读 stats → 选工作点 → 写 keywords.json。返回工作点 dict（含 keywords_json 路径）。"""
    rows = parse_stats_file(stats_path)
    op = select_operating_point(
        rows, target_fa_per_hour=target_fa_per_hour, min_recall=min_recall)
    kj = write_keywords_json(out_path, keyword, op["threshold"], template_path=template_path)
    op["keywords_json"] = str(kj)
    op["target_fa_per_hour"] = target_fa_per_hour
    return op
