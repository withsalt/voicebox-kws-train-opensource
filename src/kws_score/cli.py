"""命令行入口：唤醒词评分（评估唤醒词质量）。

用法：
  python -m kws_score.cli 小白小白 你好小白
  python -m kws_score.cli 小白小白 --json scores.json
  python -m kws_score.cli                 # 不带词时用 config.toml [score].wake_words
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .client import ScoreError
from .config import load_score_config, load_wake_words
from .metrics import HAS_PINYIN as _HAS_PINYIN
from .scorer import score_words


def _force_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _render(data: dict) -> str:
    out: list[str] = []
    results = {r["word"]: r for r in data["results"]}
    for rank, word in enumerate(data["ranking"], 1):
        r = results[word]
        s = r["scores"]
        head = f"#{rank} {r['word']}  →  {r['final_total']}/100  {r['grade']}"
        if not r["recommended"]:
            head += "  [不推荐]"
        out.append(head)
        if r["pinyin"]:
            redup = f"  叠词:{r['reduplication']}(单元 {r['base_unit']})" if r["reduplication"] else ""
            out.append(f"   音节 {r['syllables']}  拼音 {r['pinyin']}{redup}")
        out.append(
            f"   A 音节 {s['A_syllable']}/20  B 区分 {s['B_phonetic_richness']}/20  "
            f"C 误唤醒 {s['C_false_alarm']}/30  D 清晰 {s['D_pronunciation']}/15  "
            f"E 建模 {s['E_modelability']}/15"
        )
        if r["modelability_penalty"]:
            out.append(
                f"   ⛔ 建模硬扣分 -{r['modelability_penalty']}"
                f"（不可建模字 {r['unmodelable_chars']}；小计 {r['raw_total']} → {r['final_total']}）"
            )
        if r["score_reasons"]["C"]:
            out.append(f"   误唤醒：{r['score_reasons']['C']}")
        if r["key_risks"]:
            out.append("   关键风险：" + "；".join(r["key_risks"]))
        pos = r.get("positive_sample_words", [])
        rej = r.get("rejected_samples", [])
        if r.get("samples_skipped_reason"):
            out.append(f"   近音正样本：跳过（{r['samples_skipped_reason']}）")
        else:
            out.append(f"   近音正样本(应唤醒) {len(pos)} 个：" + ("、".join(pos) if pos else "（无）"))
            if rej:
                out.append(f"   （安全闸门拒绝 {len(rej)} 个候选；明细见 --json 的 rejected_samples）")
        if r["alternatives"]:
            alts = "；".join(f"{a.get('word')}({a.get('reason')})" for a in r["alternatives"])
            out.append("   替代候选：" + alts)
        out.append("")
    if data.get("summary"):
        out.append("结论：" + data["summary"])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    parser = argparse.ArgumentParser(
        prog="kws_score", description="唤醒词评分（直接调用大模型，强制结构化 JSON 输出）"
    )
    parser.add_argument("words", nargs="*", help="候选唤醒词（空格分隔）；省略则用 config 的 wake_words")
    parser.add_argument("--config", default="config.toml", help="配置文件路径")
    parser.add_argument("--json", dest="json_out", default=None, help="把结构化结果写到该 JSON 文件")
    parser.add_argument("--print-json", action="store_true", help="把完整 JSON 打到 stdout")
    args = parser.parse_args(argv)

    try:
        words = args.words or load_wake_words(args.config)
        if not words:
            print("没有候选唤醒词：请在命令行给出，或在 config.toml 的 [score].wake_words 填写。",
                  file=sys.stderr)
            return 2
        cfg = load_score_config(args.config)
        if cfg.recommend_samples and not _HAS_PINYIN:
            print(
                "⚠ 未检测到 pypinyin：B/D 维度将由大模型补全，且【近音正样本推荐会被跳过】"
                "（候选安全闸门依赖拼音做声学验证）。请安装后重试：pip install pypinyin",
                file=sys.stderr,
            )
        data = score_words(words, cfg)
    except ScoreError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 2

    if args.print_json:
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(_render(data))

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"已写入 {args.json_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
