"""强制结构化输出的工具 schema。

大模型被强制调用 `emit_wakeword_scores` 工具，只输出它真正负责的判断：
  - B/C/D 三个需主观/世界知识的维度分值（A、E、建模硬扣分由 Python 确定性给出）
  - 各维度一句话理由、误唤醒分析、关键风险、训练建议、替代候选

近音正样本词**不再由大模型提议**——改为 candidates.generate_positive_candidates 从唤醒词
拼音确定性计算（仅声调/模糊音差异、排除完全同音），故此 schema 不含正样本候选字段。
总分、等级、是否推荐等派生值在 Python 侧按规则计算，不交给模型算术，保证一致。
"""

from __future__ import annotations

TOOL_NAME = "emit_wakeword_scores"

_ALTERNATIVE = {
    "type": "object",
    "properties": {
        "word": {"type": "string", "description": "替代候选唤醒词"},
        "reason": {"type": "string", "description": "为何更优（更易检测/误唤醒更低/可建模等）"},
    },
    "required": ["word", "reason"],
    "additionalProperties": False,
}

_RESULT = {
    "type": "object",
    "properties": {
        "word": {"type": "string", "description": "对应的候选唤醒词，必须与输入完全一致"},
        "b_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 20,
            "description": "B 音素区分度(满分20)。若 metrics 已给出非 null 值必须原样照抄；仅当为 null 时按拼音知识自算",
        },
        "c_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 30,
            "description": "C 误唤醒风险(满分30)。脚本启发式仅供参考，你必须用世界知识最终定分",
        },
        "d_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 15,
            "description": "D 发音清晰度(满分15)。若 metrics 已给出非 null 值必须原样照抄；仅当为 null 时自算",
        },
        "a_reason": {"type": "string", "description": "A 音节数得分的一句话理由"},
        "b_reason": {"type": "string", "description": "B 音素区分度的一句话理由"},
        "c_reason": {"type": "string", "description": "C 误唤醒风险定分的一句话理由（写明世界知识判断）"},
        "d_reason": {"type": "string", "description": "D 发音清晰度的一句话理由"},
        "e_reason": {"type": "string", "description": "E 建模可行性的一句话理由（是否含 OOV/非中文字）"},
        "false_alarm_analysis": {
            "type": "string",
            "description": "误唤醒详细分析：是否命中常用词/人名/产品词、近音、并专门评估『连续呼叫』场景与叠词整体罕见度",
        },
        "key_risks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "最主要的 1-3 条风险；含不可建模字符时务必列为第一条致命风险",
        },
        "training_advice": {
            "type": "string",
            "description": "训练与落地建议：数据量≥25h、正负比 1:2~1:10、全文标注、threshold1 调参、Linux+GPU",
        },
        "alternatives": {
            "type": "array",
            "items": _ALTERNATIVE,
            "description": "仅当最终分 <55 或风险高时给 2-3 个更优替代；否则空数组",
        },
    },
    "required": [
        "word",
        "b_score",
        "c_score",
        "d_score",
        "a_reason",
        "b_reason",
        "c_reason",
        "d_reason",
        "e_reason",
        "false_alarm_analysis",
        "key_risks",
        "training_advice",
        "alternatives",
    ],
    "additionalProperties": False,
}

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": _RESULT,
            "description": "每个候选唤醒词一条，顺序、数量与输入一致",
        },
        "overall_summary": {
            "type": "string",
            "description": "一句话总体结论（多候选时点明最佳）。注意最终排序由调用方按总分计算，你只需给结论。",
        },
    },
    "required": ["results", "overall_summary"],
    "additionalProperties": False,
}


def tool_spec() -> dict:
    """Anthropic Messages API 的 tools[] 条目。"""
    return {
        "name": TOOL_NAME,
        "description": "提交所有候选唤醒词的评分判断（B/C/D 维度分值与文字分析）。",
        "input_schema": INPUT_SCHEMA,
    }
