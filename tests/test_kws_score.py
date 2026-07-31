"""kws_score 自测：确定性近音正样本生成 + 闸门（拒绝完全同音，仅放行近音）。

运行：conda run -n kwstrain python tests/test_kws_score.py
不依赖 pytest——纯 assert + 计数，结尾打印 PASS/FAIL 汇总。需 pypinyin（缺失则整体跳过）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from kws_score.candidates import (                                       # noqa: E402
    generate_positive_candidates, relation, verify_positive)
from kws_score.metrics import HAS_PINYIN, chars_for_syllable, modelability  # noqa: E402

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


def test_gate_rejects_identical():
    print("[verify_positive]")
    w = "小木小木"
    # 完全同音（异形同音）：小目/晓木/晓目 = xiao3 mu4 xiao3 mu4，必须被拒
    for homo in ("小目小目", "晓木晓木", "晓目晓目"):
        check(f"reject identical {homo}", relation(homo, w)["kind"] == "identical"
              and not verify_positive(homo, w)[0])
    # 近音（声调差异）：小母小母 / 笑木笑木，应放行
    for near in ("小母小母", "笑木笑木"):
        ok, info = verify_positive(near, w)
        check(f"accept near {near}", ok and info == "near", info)
    # 唤醒词本身 / 明显不同 / 子串：拒
    check("reject self", not verify_positive(w, w)[0])
    check("reject distinct 小米小米", not verify_positive("小米小米", w)[0])
    check("reject partial 小木", not verify_positive("小木", w)[0])


def test_reverse_index():
    print("[chars_for_syllable]")
    # mu3（母/亩/拇…）应能反查到可建模汉字；返回项必须都可建模、读音匹配
    chars = chars_for_syllable("m", "u", 3)
    check("mu3 has modelable chars", len(chars) > 0, str(chars[:5]))
    check("all returned chars modelable", all(modelability(c)[0] for c in chars))
    check("sorted/deterministic", chars == sorted(chars))
    # 不存在的读音返回空
    check("unknown syllable -> empty", chars_for_syllable("zzz", "qqq", 9) == [])


def test_generator():
    print("[generate_positive_candidates]")
    w = "小木小木"
    out = generate_positive_candidates(w, max_tone=5, max_total=10)
    pos = out["positives"]
    words = [p["word"] for p in pos]
    check("produced some near positives", len(pos) > 0, str(words))
    check("respects max_total", len(pos) <= 10)
    # 声调近音类不超过 max_tone（模糊音类不受此限）
    n_tone = sum(1 for p in pos if p["reason"] == "声调近音")
    check("tone near <= max_tone", n_tone <= 5, f"tone={n_tone}")
    # u↔o 模糊音：小木(mu)→小默(mo) 必须生成，且排在最前（模糊音优先）
    check("u<->o fuzzy 小默小默 generated", "小默小默" in words, str(words))
    check("小默小默 is near", relation("小默小默", w)["kind"] == "near"
          and verify_positive("小默小默", w)[0])
    if "小默小默" in words:
        check("fuzzy comes first", pos[0]["reason"].startswith("模糊"),
              f'first={pos[0]["reason"]}')
    # 每个产出都是 near、可建模、非完全同音、非唤醒词
    bad = [x for x in words if verify_positive(x, w)[1] != "near"]
    check("every candidate is near", not bad, str(bad))
    check("none identical", all(relation(x, w)["kind"] != "identical" for x in words))
    check("no duplicates / no self", len(set(words)) == len(words) and w not in words)
    # 叠词结构保持（ABAB：前半==后半）
    redup_ok = all(len(x) == 4 and x[:2] == x[2:] for x in words)
    check("keeps reduplication (ABAB)", redup_ok, str(words))
    # 确定性：再跑一次完全一致
    out2 = generate_positive_candidates(w, max_tone=5, max_total=10)
    check("deterministic (same output)", [p["word"] for p in out2["positives"]] == words)
    print("    near positives:", "、".join(words))


def main() -> int:
    print("==== kws_score 自测 ====")
    if not HAS_PINYIN:
        print("  ⚠ 未安装 pypinyin，跳过近音正样本生成测试")
        return 0
    test_gate_rejects_identical()
    test_reverse_index()
    test_generator()
    print(f"\n==== 结果：PASS={_passed}  FAIL={_failed} ====")
    return 1 if _failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
