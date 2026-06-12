"""LLM API 调用封装。支持双模型、自动降级、指数退避重试。"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any

import structlog
from openai import AsyncOpenAI

from src.shared.config import get_settings
from src.shared.errors import LLMError, LLMRateLimitError, LLMTimeoutError

logger = structlog.get_logger()


class LLMClient:
    """OpenAI-compatible API 客户端。"""

    def __init__(self) -> None:
        settings = get_settings()
        self._primary_model = settings.llm_model
        self._fallback_model = settings.llm_fallback_model
        self._timeout = settings.tool_call_timeout
        self._client = AsyncOpenAI(
            api_key=settings.llm_api_key or "not-needed",
            base_url=settings.llm_base_url,
            timeout=float(self._timeout),
        )
        self._total_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        trace_id: str = "",
        allow_fallback: bool = True,
    ) -> str:
        """发送 chat completion 请求，失败时自动降级。"""
        primary_model = model or self._primary_model
        last_error: Exception | None = None

        for model_name in (primary_model, self._fallback_model):
            if model_name == self._fallback_model:
                if not allow_fallback:
                    break
                logger.warning(
                    "llm_fallback",
                    primary=primary_model,
                    fallback=self._fallback_model,
                    trace_id=trace_id,
                )
            try:
                result = await self._call_with_retry(
                    messages,
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    trace_id=trace_id,
                )
                return result
            except (LLMRateLimitError, LLMTimeoutError, LLMError) as e:
                last_error = e
                if model_name == self._fallback_model:
                    break
                logger.warning(
                    "llm_retry_next_model",
                    failed_model=model_name,
                    next_model=self._fallback_model,
                    error=str(e),
                    trace_id=trace_id,
                )

        raise LLMError(f"LLM call failed after all attempts: {last_error}")

    async def _call_with_retry(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
        trace_id: str,
    ) -> str:
        """指数退避重试，最多 3 次。"""
        delays = [2, 4, 8]
        last_error: Exception | None = None

        for attempt, delay in enumerate(delays):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                usage = response.usage
                if usage:
                    self._total_tokens += usage.total_tokens
                content = response.choices[0].message.content or ""
                logger.debug(
                    "llm_call_success",
                    model=model,
                    attempt=attempt + 1,
                    tokens=usage.total_tokens if usage else 0,
                    trace_id=trace_id,
                )
                return content

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                if "429" in error_str or "rate" in error_str:
                    exc: LLMError = LLMRateLimitError(str(e))
                elif "timeout" in error_str:
                    exc = LLMTimeoutError(str(e))
                else:
                    exc = LLMError(str(e))

                logger.warning(
                    "llm_call_retry",
                    model=model,
                    attempt=attempt + 1,
                    delay=delay,
                    error=str(exc),
                    trace_id=trace_id,
                )
                if attempt < len(delays) - 1:
                    await asyncio.sleep(delay)

        raise last_error if isinstance(last_error, LLMError) else LLMError(str(last_error))

    def estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数（~4 chars/token）。"""
        return max(1, len(text) // 4)

    @staticmethod
    def params_hash(params: dict[str, Any]) -> str:
        """计算参数 hash 用于审计日志。"""
        raw = str(sorted(params.items()))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


# 全局单例
_llm_client: LLMClient | None = None


def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
