"""编排：客观指标 → 大模型判断 → 组装确定性最终 JSON。

总分与等级在此用规则确定性计算（不信任模型算术）：
  final = max(0, A+B+C+D+E − 40×含不可建模字符)
A/E/建模硬扣分恒取脚本值；B/D 优先取脚本(pypinyin)值、缺失时取模型补全；C 取模型世界知识判断。
"""

from __future__ import annotations

from .candidates import generate_positive_candidates
from .client import AnthropicClient, ScoreError
from .config import ScoreConfig, load_score_config
from .metrics import compute_metrics, grade
from .prompt import SYSTEM_PROMPT, build_user_content
from .schema import tool_spec

RECOMMEND_THRESHOLD = 55  # 合格线：≥55 视为可用（推荐）


def _clamp(v, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(v)))
    except (TypeError, ValueError):
        return default


def _assemble(metric: dict, judged: dict | None, samples: dict) -> dict:
    obj = metric["objective_scores"]
    A = obj["A_syllable"]
    E = obj["E_modelability"]
    penalty = metric["modelability_penalty"]

    # B/D：脚本(pypinyin)算出的优先；为 None 时用模型补全值
    B = obj["B_phonetic_richness"]
    if B is None:
        B = _clamp(judged and judged.get("b_score"), 0, 20, 0)
    D = obj["D_pronunciation"]
    if D is None:
        D = _clamp(judged and judged.get("d_score"), 0, 15, 0)
    # C：始终采用模型世界知识判断；模型缺失时退回脚本启发式
    if judged and judged.get("c_score") is not None:
        C = _clamp(judged.get("c_score"), 0, 30, obj["C_false_alarm_heuristic"])
    else:
        C = obj["C_false_alarm_heuristic"]

    raw_total = A + B + C + D + E
    final_total = max(0, raw_total - penalty)
    j = judged or {}

    return {
        "word": metric["word"],
        "syllables": metric["syllables"],
        "pinyin": " ".join(metric["pinyin"]),
        "reduplication": metric["reduplication"],
        "base_unit": metric["base_unit"],
        "scores": {
            "A_syllable": A,
            "B_phonetic_richness": B,
            "C_false_alarm": C,
            "D_pronunciation": D,
            "E_modelability": E,
        },
        "score_reasons": {
            "A": j.get("a_reason", ""),
            "B": j.get("b_reason", ""),
            "C": j.get("c_reason", ""),
            "D": j.get("d_reason", ""),
            "E": j.get("e_reason", ""),
        },
        "unmodelable_chars": metric["unmodelable_chars"],
        "modelability_penalty": penalty,
        "raw_total": raw_total,
        "final_total": final_total,
        "grade": grade(final_total),
        "recommended": final_total >= RECOMMEND_THRESHOLD,
        "false_alarm_analysis": j.get("false_alarm_analysis", ""),
        "key_risks": list(j.get("key_risks", []) or []),
        "training_advice": j.get("training_advice", ""),
        "alternatives": list(j.get("alternatives", []) or []),
        # —— 近音正样本（已过确定性安全闸门，作为正样本变体推荐；负样本另行处理）——
        "positive_sample_words": [s["word"] for s in samples.get("positives", [])],
        "positive_candidates": samples.get("positives", []),
        "rejected_samples": samples.get("rejected", []),
        "samples_skipped_reason": samples.get("skipped_reason", ""),
        "llm_finalized": judged is not None,
    }


def _build_samples(word: str, cfg: ScoreConfig) -> dict:
    """**确定性**从唤醒词计算近音正样本词（不依赖大模型提议），作为正样本变体推荐。
    合成/采集为正样本语料时统一标注为目标唤醒词。负样本另行处理。"""
    if not cfg.recommend_samples:
        return {"positives": [], "rejected": []}
    return generate_positive_candidates(
        word, max_tone=cfg.max_positive_samples, max_total=cfg.max_fuzzy_samples
    )


def _normalize_words(words) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for w in words:
        w = (w or "").strip()
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return out


def score_words(
    words, config: ScoreConfig | None = None, *, config_path: str = "config.toml"
) -> dict:
    """对一个或多个候选唤醒词评分，返回组装好的结构化结果。"""
    words = _normalize_words(words)
    if not words:
        raise ScoreError("没有候选唤醒词")

    cfg = config or load_score_config(config_path)
    metrics_list = [compute_metrics(w) for w in words]

    client = AnthropicClient(
        base_url=cfg.base_url,
        api_key=cfg.require_api_key(),
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        temperature=cfg.temperature,
        timeout=cfg.timeout,
        max_retries=cfg.max_retries,
    )
    raw = client.call_tool(
        SYSTEM_PROMPT, build_user_content(metrics_list), tool_spec()
    )

    judged_by_word: dict[str, dict] = {}
    for item in raw.get("results", []) or []:
        if isinstance(item, dict) and item.get("word"):
            judged_by_word[item["word"]] = item

    results = []
    for m in metrics_list:
        judged = judged_by_word.get(m["word"])
        samples = _build_samples(m["word"], cfg)
        results.append(_assemble(m, judged, samples))

    # ranking 按确定性最终分排序（与展示分数一致），同分用 C 误唤醒、原始顺序兜底；
    # 不采用模型自报的排序，避免出现「#1 分数反而比 #2 低」的矛盾。
    order = {r["word"]: i for i, r in enumerate(results)}
    ranking = [
        r["word"]
        for r in sorted(
            results,
            key=lambda r: (-r["final_total"], -r["scores"]["C_false_alarm"], order[r["word"]]),
        )
    ]

    return {
        "model": cfg.model,
        "weights": {"A": 20, "B": 20, "C": 30, "D": 15, "E": 15},
        "summary": raw.get("overall_summary", ""),
        "ranking": ranking,
        "results": results,
    }
