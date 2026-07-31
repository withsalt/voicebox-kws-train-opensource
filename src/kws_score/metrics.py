"""唤醒词客观指标（确定性，不调模型）。

移植自原 `wakeword-score` 技能的 scripts/score.py，自包含：reference 数据随包携带。
基于阿里 speech_charctc_kws_phone（4 层 cFSMN / char-CTC，近场移动端）建模特性：

    A 音节数        20   客观
    B 音素区分度    20   需 pypinyin；缺失时为 None，交大模型按拼音知识补全
    C 误唤醒风险    30   仅给启发式值与线索；最终由大模型用世界知识判定
    D 发音清晰度    15   需 pypinyin；同 B
    E 建模可行性    15   客观（依据捆绑 token 表）

含「不可建模字符」（非中文或不在 token 表的 OOV 字）→ 总分额外硬扣 40，且总分封底 0：
    final = max(0, A + B + C + D + E - 40)
"""

from __future__ import annotations

import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REFERENCE = _HERE / "reference"
TOKENS_PATH = _REFERENCE / "tokens_2599.txt"
COMMON_PATH = _REFERENCE / "common_words.txt"

MODELABILITY_PENALTY = 40  # 含不可建模字符时对总分的硬扣分

try:  # pypinyin 可选：装了则 B/D 客观可算，否则留空交给大模型补全
    from pypinyin import Style, pinyin

    HAS_PINYIN = True
except Exception:  # pragma: no cover - 取决于环境
    HAS_PINYIN = False

_CJK = re.compile(r"[一-鿿]")


def _load_set(path: Path) -> set[str]:
    s: set[str] = set()
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            w = line.strip()
            if w and not w.startswith("#"):
                s.add(w)
    except FileNotFoundError:
        pass
    return s


TOKENS = _load_set(TOKENS_PATH)
COMMON = _load_set(COMMON_PATH)


def _is_cjk(ch: str) -> bool:
    return bool(_CJK.match(ch))


def _get_pinyin(cjk_chars: list[str]) -> tuple[list[str], list[str], list[int]]:
    """返回 (声母, 韵母, 声调)，均与 cjk_chars 等长。需 pypinyin。"""
    s = "".join(cjk_chars)
    inits = [x[0] for x in pinyin(s, style=Style.INITIALS, strict=False, heteronym=False)]
    finals = [x[0] for x in pinyin(s, style=Style.FINALS, strict=False, heteronym=False)]
    t3 = [
        x[0]
        for x in pinyin(
            s, style=Style.TONE3, strict=False, heteronym=False, neutral_tone_with_five=True
        )
    ]
    inits = ["0" if not i else i for i in inits]  # 零声母记为 0
    tones: list[int] = []
    for t in t3:
        m = re.search(r"[1-5]$", t)
        tones.append(int(m.group()) if m else 0)  # 0 = 未知/无调
    return inits, finals, tones


def _score_A(n: int) -> int:
    table = {1: 0, 2: 5, 3: 12, 4: 20, 5: 16, 6: 12}
    return table.get(n, 8 if n >= 7 else 0)


def _score_B(inits: list[str], finals: list[str], n: int):
    if n == 0:
        return None, None, None
    di = len(set(inits))
    df = len(set(finals))
    ratio = (di + df) / (2 * n)
    return round(20 * min(1.0, ratio)), di, df


def _score_D(inits: list[str], finals: list[str], tones: list[int], n: int):
    if n == 0:
        return None, {}
    distinct_tones = len({t for t in tones if t})
    run = maxrun = 1
    for i in range(1, n):
        if inits[i] == inits[i - 1]:
            run += 1
            maxrun = max(maxrun, run)
        else:
            run = 1
    neutral = sum(1 for t in tones if t in (5, 0))
    d = 15
    if distinct_tones <= 1:
        d -= 5
    elif distinct_tones == 2:
        d -= 1
    if maxrun >= n and n >= 3:
        d -= 5
    elif maxrun >= 3:
        d -= 3
    d -= min(4, neutral * 2)
    d = max(0, d)
    detail = {
        "distinct_tones": distinct_tones,
        "max_same_initial_run": maxrun,
        "neutral_syllables": neutral,
        "tone_pattern": tones,
    }
    return d, detail


def _score_E(cjk_chars: list[str], non_cjk: list[str]) -> tuple[int, list[str]]:
    e = 15
    oov = [c for c in cjk_chars if TOKENS and c not in TOKENS]
    if non_cjk:
        e -= min(15, len(non_cjk) * 8)  # 模型只建模中文字符
    if oov:
        e -= min(15, len(oov) * 8)
    return max(0, e), oov


def _detect_redup(cjk_chars: list[str], n: int):
    """检测叠词结构，返回 (类型, 基本单元 list)。"""
    if n >= 2 and n % 2 == 0 and cjk_chars[: n // 2] == cjk_chars[n // 2 :]:
        return "ABAB", cjk_chars[: n // 2]  # 单元整体重复，如 小白|小白
    if (
        n == 4
        and cjk_chars[0] == cjk_chars[1]
        and cjk_chars[2] == cjk_chars[3]
        and cjk_chars[0] != cjk_chars[2]
    ):
        return "AABB", [cjk_chars[0], cjk_chars[2]]  # 如 平平|安安
    return None, list(cjk_chars)


def _false_alarm(cjk_chars: list[str], base: list[str], redup):
    """启发式误唤醒线索。真正判定由大模型用世界知识完成。"""
    joined = "".join(cjk_chars)
    base_s = "".join(base)
    flags = sorted({w for w in COMMON if w and w in joined}, key=len, reverse=True)
    c = 30
    base_is_common = base_s in COMMON and len(base_s) >= 2
    if base_is_common:
        c -= 12  # 基本单元本身是常用词/名 → 连呼风险
    for w in flags:
        if w == base_s:
            continue
        c -= 6 if len(w) >= 2 else 2
    if redup and len(cjk_chars) >= 4:
        c += 6  # 4+ 音节叠词整体在自然语流中罕见
    c = max(0, min(30, c))
    return c, flags, base_is_common


def grade(total: int) -> str:
    if total >= 85:
        return "优秀（强烈推荐）"
    if total >= 70:
        return "良好（推荐）"
    if total >= 55:
        return "合格（可用，建议针对性优化）"
    return "不推荐（建议更换）"


def compute_metrics(word: str) -> dict:
    """计算单个候选词的客观指标（C 仅启发式，B/D 缺 pypinyin 时为 None）。"""
    chars = [c for c in word if not c.isspace()]
    cjk_chars = [c for c in chars if _is_cjk(c)]
    non_cjk = [c for c in chars if not _is_cjk(c)]
    n = len(cjk_chars)

    redup, base = _detect_redup(cjk_chars, n)
    A = _score_A(n)
    E, oov = _score_E(cjk_chars, non_cjk)
    C_heur, fa_flags, base_is_common = _false_alarm(cjk_chars, base, redup)

    if HAS_PINYIN and n > 0:
        inits, finals, tones = _get_pinyin(cjk_chars)
        B, di, df = _score_B(inits, finals, n)
        D, d_detail = _score_D(inits, finals, tones, n)
        pinyin_full = [
            x[0]
            for x in pinyin(
                "".join(cjk_chars), style=Style.TONE3, strict=False, neutral_tone_with_five=True
            )
        ]
    else:
        inits = finals = tones = []
        B = D = di = df = None
        d_detail = {}
        pinyin_full = []

    unmodelable_chars = list(non_cjk) + list(oov)
    modelability_penalty = MODELABILITY_PENALTY if unmodelable_chars else 0

    return {
        "word": word,
        "syllables": n,
        "non_chinese_chars": non_cjk,
        "oov_chars": oov,
        "unmodelable_chars": unmodelable_chars,
        "modelability_penalty": modelability_penalty,
        "reduplication": redup,
        "base_unit": "".join(base),
        "pinyin": pinyin_full,
        "objective_scores": {
            "A_syllable": A,
            "B_phonetic_richness": B,
            "C_false_alarm_heuristic": C_heur,
            "D_pronunciation": D,
            "E_modelability": E,
        },
        "weights": {"A": 20, "B": 20, "C": 30, "D": 15, "E": 15},
        "phonetic_detail": {
            "distinct_initials": di,
            "distinct_finals": df,
            "initials": inits,
            "finals": finals,
            **d_detail,
        },
        "false_alarm_clues": {
            "matched_common_words": fa_flags,
            "base_unit_is_common": base_is_common,
        },
        "has_pinyin": HAS_PINYIN and n > 0,
    }


# ---- 供候选词安全闸门(candidates.py)复用的轻量公共助手 ----

def pinyin_syllables(word: str) -> list[tuple[str, str, int]] | None:
    """返回词中每个中文字的 (声母, 韵母, 声调)。无 pypinyin 或无中文字时返回 None
    （表示无法做声学判定 → 调用方应按 fail-safe 处理）。"""
    cjk = [c for c in word if _is_cjk(c)]
    if not HAS_PINYIN or not cjk:
        return None
    inits, finals, tones = _get_pinyin(cjk)
    return list(zip(inits, finals, tones))


def modelability(word: str) -> tuple[bool, list[str]]:
    """该词能否被本模型建模：返回 (是否可建模, 不可建模字符列表)。
    含非中文字符或不在 token 表的 OOV 字即不可建模；无中文字也视为不可建模。"""
    chars = [c for c in word if not c.isspace()]
    cjk = [c for c in chars if _is_cjk(c)]
    non_cjk = [c for c in chars if not _is_cjk(c)]
    oov = [c for c in cjk if TOKENS and c not in TOKENS]
    unmodelable = list(non_cjk) + list(oov)
    return (not unmodelable and len(cjk) > 0), unmodelable


def display_pinyin(word: str) -> str:
    """带声调数字的展示用拼音（如 'xiao3 bai2'）；无 pypinyin/中文时返回空串。"""
    cjk = [c for c in word if _is_cjk(c)]
    if not HAS_PINYIN or not cjk:
        return ""
    return " ".join(
        x[0]
        for x in pinyin(
            "".join(cjk), style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    )


# ---- 反查：拼音(声母,韵母,声调) -> 可建模汉字（供 candidates.py 确定性生成近音正样本）----

_REVERSE_INDEX: dict[tuple[str, str, int], list[str]] | None = None


def _syllable_key(ch: str) -> tuple[str, str, int]:
    inits, finals, tones = _get_pinyin([ch])
    return (inits[0], finals[0], tones[0])


def chars_for_syllable(initial: str, final: str, tone: int) -> list[str]:
    """返回 token 表中读音恰为 (声母, 韵母, 声调) 的全部可建模汉字（按码位升序，确定性）。

    索引仅在首次调用时从 token 表惰性构建（约 1600+ 个可建模汉字过一遍 pypinyin）。
    无 pypinyin 时返回空列表。零声母记为 '0'，与 pinyin_syllables/relation 保持同一约定。
    """
    global _REVERSE_INDEX
    if not HAS_PINYIN:
        return []
    if _REVERSE_INDEX is None:
        idx: dict[tuple[str, str, int], list[str]] = {}
        for ch in sorted(c for c in TOKENS if len(c) == 1 and _is_cjk(c)):
            idx.setdefault(_syllable_key(ch), []).append(ch)
        _REVERSE_INDEX = idx
    return _REVERSE_INDEX.get((initial, final, tone), [])
