"""评分配置：读取 config.toml 的 [score] 段，允许环境变量覆盖。

环境变量：
  KWS_SCORE_BASE_URL / KWS_SCORE_API_KEY / KWS_SCORE_MODEL
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .client import ScoreError

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MODEL = "claude-opus-4-8"


@dataclass
class ScoreConfig:
    base_url: str = DEFAULT_BASE_URL
    api_key: str = ""
    model: str = DEFAULT_MODEL
    max_tokens: int = 8192
    temperature: float | None = None  # None=不下发（claude-opus-4-8 已弃用该参数）
    timeout: float = 120.0
    max_retries: int = 3
    # 近音正样本推荐（负样本不在本工具范围内，由外部数据集合成）
    recommend_samples: bool = True
    max_positive_samples: int = 5   # 声调近音上限（基础；模糊音近音不受此限）
    max_fuzzy_samples: int = 10     # 模糊音近音/近音正样本总数上限（模糊音可突破基础上限直至此值）

    def require_api_key(self) -> str:
        if not self.api_key:
            raise ScoreError(
                "缺少 [score].api_key（或环境变量 KWS_SCORE_API_KEY）。"
                "请在 config.toml 的 [score] 段填写大模型服务 key。"
            )
        return self.api_key


def _read_raw(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("rb") as f:
        return tomllib.load(f)


def load_score_config(path: str | Path = "config.toml") -> ScoreConfig:
    raw = _read_raw(path)
    s = raw.get("score", {})
    return ScoreConfig(
        base_url=os.environ.get("KWS_SCORE_BASE_URL", s.get("base_url", DEFAULT_BASE_URL)),
        api_key=os.environ.get("KWS_SCORE_API_KEY", s.get("api_key", "")),
        model=os.environ.get("KWS_SCORE_MODEL", s.get("model", DEFAULT_MODEL)),
        max_tokens=int(s.get("max_tokens", 8192)),
        temperature=(None if s.get("temperature") is None else float(s["temperature"])),
        timeout=float(s.get("timeout", 120.0)),
        max_retries=int(s.get("max_retries", 3)),
        recommend_samples=bool(s.get("recommend_samples", True)),
        max_positive_samples=int(s.get("max_positive_samples", 5)),
        max_fuzzy_samples=int(s.get("max_fuzzy_samples", 10)),
    )


def load_wake_words(path: str | Path = "config.toml") -> list[str]:
    """无命令行候选词时，回退到 config 的 [score].wake_words（可选列表）。"""
    raw = _read_raw(path)
    return list(raw.get("score", {}).get("wake_words", []))
