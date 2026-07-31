"""近音正样本：确定性生成 + 安全闸门（不依赖大模型）。

为什么这样设计：后续流水线无人工介入。这里用 pypinyin 做**确定性**计算——直接从唤醒词
逐音节做【单特征】扰动（仅声调差 或 单个模糊声母/韵母替换），用 token 表里可建模的汉字
实现该读音，**枚举出真正的近音正样本**，再经安全闸门复核：

  - 只放行 `near`（同音节数、仅声调/模糊音差异）——模拟口音/模糊发音，偏召回、应唤醒；
  - **拒绝 `identical`（完全同音，异形同音）**——合成出来与唤醒词几乎是重复音频，无增益；
  - 拒绝可建模性不过关或日常常见词（避免教模型对常见词也唤醒）。

这些近音正样本作为唤醒词的正样本变体推荐（合成/采集为正样本语料时统一标注为目标唤醒词）。
负样本不在本工具范围内。
"""

from __future__ import annotations

from .metrics import (
    COMMON,
    HAS_PINYIN,
    chars_for_syllable,
    display_pinyin,
    modelability,
    pinyin_syllables,
)

# 模糊音等价类（声学相近、易混的声母/韵母）。**故意从宽**：多收一对，只会让更多词被判为
# "近音"从而被【禁入负样本】（安全方向）；漏收一对，则可能让近音词混进负样本（最危险）。
# 对正样本而言从宽=多几个近音正样本，符合"重召回、模糊音也唤醒"的取向。
FUZZY_INITIALS = [
    {"b", "p"}, {"d", "t"}, {"g", "k"},          # 送气/不送气塞音对（快速口语常混）
    {"zh", "z"}, {"ch", "c"}, {"sh", "s"},        # 平翘舌
    {"zh", "j"}, {"ch", "q"}, {"sh", "x"},        # 翘舌/舌面（部分口音）
    {"n", "l"}, {"l", "r"}, {"f", "h"},            # n/l、l/r、f/h 常见方言合并
]
FUZZY_FINALS = [
    {"an", "ang"}, {"en", "eng"}, {"in", "ing"},  # 前后鼻音
    {"ian", "iang"}, {"uan", "uang"}, {"uen", "ueng"},
    {"ai", "ei"}, {"ao", "ou"}, {"u", "ou"},       # 相近元音
    {"u", "o"},                                    # mu↔mo（如 木↔默/摸）——口语易混，重召回方向从宽
]


def _fuzzy_eq(a: str, b: str, groups: list[set[str]]) -> bool:
    if a == b:
        return True
    return any(a in g and b in g for g in groups)


def _seg_relation(s1: tuple[str, str, int], s2: tuple[str, str, int]) -> str:
    """逐字关系：same(全同) / tone(仅声调差) / fuzzy(声韵均同或模糊近) / diff(明显不同)。"""
    i1, f1, t1 = s1
    i2, f2, t2 = s2
    if i1 == i2 and f1 == f2:
        return "same" if t1 == t2 else "tone"
    if _fuzzy_eq(i1, i2, FUZZY_INITIALS) and _fuzzy_eq(f1, f2, FUZZY_FINALS):
        return "fuzzy"
    return "diff"


def relation(candidate: str, wake: str) -> dict:
    """候选词与唤醒词的整体声学关系。

    kind:
      identical  — 同音节数且每字声韵调全同（异形同音，音频几乎一致）
      near       — 同音节数，仅声调/模糊音差异（无明显不同音节）→ 近音
      partial    — 音节数不同，且短词是长词的近音连续子串 或 二者互为子串（如叠词基本单元）
      distinct   — 其余（明显不同）
      unverifiable — 缺 pypinyin/无中文字，无法判定（fail-safe 当作不可用）
    near_equal_ratio: 同音节数时 (same|tone|fuzzy) 音节占比；其余给保守值。
    """
    cs = pinyin_syllables(candidate)
    ws = pinyin_syllables(wake)
    if cs is None or ws is None:
        return {"kind": "unverifiable", "near_equal_ratio": 1.0}

    if len(cs) != len(ws):
        if candidate in wake or wake in candidate:
            return {"kind": "partial", "near_equal_ratio": 1.0}
        short, long_ = (cs, ws) if len(cs) < len(ws) else (ws, cs)
        n = len(short)
        for start in range(0, len(long_) - n + 1):
            if all(
                _seg_relation(short[k], long_[start + k]) in ("same", "tone", "fuzzy")
                for k in range(n)
            ):
                return {"kind": "partial", "near_equal_ratio": 1.0}
        return {"kind": "distinct", "near_equal_ratio": 0.0}

    segs = [_seg_relation(cs[k], ws[k]) for k in range(len(cs))]
    hard = sum(1 for s in segs if s == "diff")
    ratio = (len(segs) - hard) / len(segs)
    if all(s == "same" for s in segs):
        kind = "identical"
    elif hard == 0:
        kind = "near"
    else:
        kind = "distinct"
    return {"kind": kind, "near_equal_ratio": ratio, "segs": segs}


def verify_positive(candidate: str, wake: str) -> tuple[bool, str]:
    """近音正样本闸门：必须是与唤醒词同音节数、**仅声调/模糊音差异**的近音词（near），
    且完全可建模、非日常常见词。**完全同音（identical）会被拒**——异形同音合出来与
    唤醒词几乎一致，不带来新声学信息。通过返回 (True, "near")；否则 (False, 原因)。"""
    if candidate == wake:
        return False, "与唤醒词完全相同"
    ok, unmod = modelability(candidate)
    if not ok:
        return False, (f"不可建模(含 {unmod})" if unmod else "无可建模中文音节")
    rel = relation(candidate, wake)
    if rel["kind"] != "near":
        return False, (
            f"非近音({rel['kind']})——近音正样本要求同音节数、仅声调/模糊音差异，"
            "且非完全同音(identical)"
        )
    if candidate in COMMON:
        return False, "是常见日常词，作正样本会引发过度误唤醒"
    return True, rel["kind"]


def _fuzzy_set(x: str, groups: list[set[str]]) -> set[str]:
    """与 x 互为模糊音的等价项（不含 x 自身）。"""
    out: set[str] = set()
    for g in groups:
        if x in g:
            out |= g - {x}
    return out


def _redup_unit_len(chars: list[str]) -> int:
    """叠词基本单元长度：ABAB（前半==后半，如 小木|小木）返回 n//2；否则整体作单元返回 n。
    用于让生成的近音词保持叠词结构（基本单元内同步替换），连读更自然。"""
    n = len(chars)
    if n >= 2 and n % 2 == 0 and chars[: n // 2] == chars[n // 2 :]:
        return n // 2
    return n


def _near_targets(syl: tuple[str, str, int]) -> list[tuple[str, str, int, str]]:
    """一个音节的近音目标读音（单特征差异），返回 (声母, 韵母, 声调, 差异类型)。
    顺序：声调变体 → 模糊声母 → 模糊韵母（确定性）。"""
    i, f, t = syl
    out: list[tuple[str, str, int, str]] = []
    if t != 0:
        for nt in (1, 2, 3, 4):
            if nt != t:
                out.append((i, f, nt, "声调"))
    for fi in sorted(_fuzzy_set(i, FUZZY_INITIALS)):
        out.append((fi, f, t, "模糊声母"))
    for ff in sorted(_fuzzy_set(f, FUZZY_FINALS)):
        out.append((i, ff, t, "模糊韵母"))
    return out


def generate_positive_candidates(wake: str, *, max_tone: int = 5, max_total: int = 10) -> dict:
    """**确定性**生成近音正样本词（不依赖大模型）。

    做法：对唤醒词（叠词则取其基本单元）逐音节做【单特征】扰动——换一个声调，或把声母/韵母
    替换成其模糊音；用 token 表里读音匹配的可建模汉字实现该读音，叠词同步替换各单元保持结构。
    每个候选再经 verify_positive 复核（仅放行 near、拒绝完全同音/常见词/不可建模）。轮转各
    音节位置以均衡覆盖。

    两级上限（模糊音近音是口音/方言混淆的关键来源，优先且不受声调上限约束）：
      · 模糊声母/韵母近音（如 木mu→默mo、in→ing、z→zh）：**优先生成**，仅受总数 max_total 封顶；
      · 声调近音（仅改声调）：补足剩余名额，自身 ≤ max_tone，且总数仍 ≤ max_total。
    即总数封顶 max_total，超出时先砍声调、保模糊音。

    返回:
      positives  [{word, pinyin, relation, reason}]   近音正样本（应唤醒，模糊音排在前）
      rejected   [{word, role, reason}]               生成但被闸门拒绝的项（审计用）
      skipped_reason  唤醒词不适合生成时给出（否则不含该键）
    """
    empty = {"positives": [], "rejected": []}
    if not HAS_PINYIN:
        return {**empty, "skipped_reason": "未安装 pypinyin，无法确定性生成近音正样本；请先 pip install pypinyin"}
    ok, unmod = modelability(wake)
    if not ok:
        return {**empty, "skipped_reason": f"唤醒词含不可建模字符 {unmod}，跳过近音正样本生成"}
    ws = pinyin_syllables(wake)
    if ws is None or len(ws) < 2:
        return {**empty, "skipped_reason": "唤醒词音节过短（<2），跳过近音正样本生成"}

    chars = [c for c in wake if not c.isspace()]
    unit = _redup_unit_len(chars)
    reps = len(chars) // unit
    per_pos = [_near_targets(ws[p]) for p in range(unit)]
    # 按差异类型拆两组，各自保留按位置的目标顺序：模糊音（声母/韵母）优先，声调补足。
    fuzzy_pos = [[t for t in per_pos[p] if t[3] != "声调"] for p in range(unit)]
    tone_pos = [[t for t in per_pos[p] if t[3] == "声调"] for p in range(unit)]

    accepted: list[dict] = []
    rejected: list[dict] = []
    seen: set[str] = {wake}

    def emit(targets_per_pos: list[list], limit: int) -> None:
        """轮转生成：按目标序号外层、位置内层均衡覆盖各音节；accepted 达 limit 即停。"""
        maxlen = max((len(p) for p in targets_per_pos), default=0)
        for r in range(maxlen):
            for pos in range(unit):
                if len(accepted) >= limit:
                    return
                if r >= len(targets_per_pos[pos]):
                    continue
                ti, tf, tt, kind = targets_per_pos[pos][r]
                for ch in chars_for_syllable(ti, tf, tt):
                    if ch == chars[pos]:
                        continue
                    unit_chars = chars[:unit].copy()
                    unit_chars[pos] = ch
                    cand = "".join(unit_chars) * reps
                    if cand in seen:
                        continue
                    seen.add(cand)
                    passed, info = verify_positive(cand, wake)
                    if passed:
                        accepted.append({
                            "word": cand, "pinyin": display_pinyin(cand),
                            "relation": info, "reason": f"{kind}近音",
                        })
                        break                    # 该目标取到一个可用汉字即可
                    rejected.append({"word": cand, "role": "positive", "reason": info})

    emit(fuzzy_pos, max_total)                                  # 模糊音优先，仅受总数封顶
    emit(tone_pos, min(max_total, len(accepted) + max_tone))    # 声调补足，自身≤max_tone 且总数≤max_total
    return {"positives": accepted, "rejected": rejected}
