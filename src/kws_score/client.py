"""调用大模型服务（官方 anthropic Python SDK）。

不自己造 HTTP 轮子：base_url 拼接、重试退避、超时都交给 SDK。这里只做三件事：
强制 tool_choice 让模型必须以工具调用输出结构化 JSON、解析首个 tool_use 块、把
SDK 异常统一收敛成 ScoreError。
"""

from __future__ import annotations

import anthropic

# 锁定一个已知能过网关 WAF 的 User-Agent：部分网关会把某些客户端 UA 重定向到验证页(307)。
USER_AGENT = "kws-score/1.0"


class ScoreError(Exception):
    """评分调用相关错误（配置缺失、网络、鉴权、协议等）。"""


def _normalize_base_url(base_url: str) -> str:
    """SDK 会自行在 base_url 后拼 /v1/messages，因此去掉用户可能多写的 /v1 尾缀。"""
    b = (base_url or "").rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3].rstrip("/")
    return b or base_url


class AnthropicClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        max_tokens: int = 4096,
        temperature: float | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        if not api_key:
            raise ScoreError("缺少 api_key（请配置 [score].api_key 或环境变量 KWS_SCORE_API_KEY）")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=_normalize_base_url(base_url),
            timeout=timeout,
            max_retries=max_retries,
            default_headers={"User-Agent": USER_AGENT},
        )

    def call_tool(self, system: str, user_content: str, tool: dict) -> dict:
        """发起一次强制工具调用，返回首个 tool_use 块的 input(dict)。"""
        kwargs = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
        )
        # 部分模型（如 claude-opus-4-8）已弃用 temperature，传了会 400；仅在显式设置时附带。
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature

        try:
            msg = self._client.messages.create(**kwargs)
        except anthropic.APIStatusError as e:
            raise ScoreError(f"请求失败 HTTP {e.status_code}: {_brief(e)}") from e
        except anthropic.APIConnectionError as e:
            raise ScoreError(f"网络错误: {e}") from e
        except anthropic.AnthropicError as e:
            raise ScoreError(f"调用失败: {e}") from e

        return _extract_tool_input(msg)


def _brief(e: anthropic.APIStatusError) -> str:
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return str(e)[:500]


def _extract_tool_input(msg) -> dict:
    """从 Messages 响应里取首个 tool_use 块的 input（按 type 取，不按 name，代理可能改写工具名）。"""
    for block in msg.content:
        if getattr(block, "type", None) == "tool_use":
            inp = block.input
            if isinstance(inp, dict):
                return inp
            raise ScoreError(f"tool_use.input 不是对象：{inp!r}")
    texts = [getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text"]
    raise ScoreError(
        f"模型未按预期调用工具。stop_reason={getattr(msg, 'stop_reason', None)!r}；"
        f"文本：{' '.join(texts)[:500]!r}"
    )
