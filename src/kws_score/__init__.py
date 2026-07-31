"""唤醒词评分（KWS wake-word scoring）——评估唤醒词质量。

可独立运行的模块：确定性客观指标由 Python 计算（metrics.py），误唤醒等需世界知识的
维度直接调用大模型服务（Anthropic Messages 兼容，强制工具调用输出结构化 JSON）。

公开入口：
    from kws_score import score_words, load_score_config
"""

from __future__ import annotations

from .config import ScoreConfig, load_score_config
from .scorer import score_words

__all__ = ["score_words", "load_score_config", "ScoreConfig"]
