"""Anthropic backend. Used when a `text_llm` resolves to the anthropic provider."""

from __future__ import annotations

import json
import logging
from typing import Any

from anthropic import AsyncAnthropic

from ...config import settings
from .base import LLMProvider, LLMResult, ToolCall, ToolOutcome, Usage

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self) -> None:
        # Bounded for the same reason as the Groq client - see config.py.
        self._client = AsyncAnthropic(
            api_key=settings.anthropic_api_key or None,
            timeout=settings.llm_timeout_s,
            max_retries=settings.llm_max_retries,
        )

    # -- request building ---------------------------------------------------

    @staticmethod
    def _translate_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
        if not tools:
            return None
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", {"type": "object", "properties": {}}),
            }
            for tool in tools
        ]

    @staticmethod
    def _thinking() -> dict[str, str]:
        # Opus 4.8 runs without thinking unless adaptive is set explicitly.
        return {"type": "adaptive"} if settings.thinking_mode == "adaptive" else {"type": "disabled"}

    @staticmethod
    def _usage(response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", 0) or 0,
        )

    @staticmethod
    def _text_of(response: Any) -> str:
        return "\n".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        ).strip()

    # -- LLMProvider --------------------------------------------------------

    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {
            # Per-call override wins - see LLMProvider.complete.
            "model": self.require_model(model),
            "max_tokens": max_tokens or settings.max_tokens,
            "system": system,
            "messages": messages,
            "thinking": self._thinking(),
            "output_config": {"effort": settings.effort},
        }

        translated = self._translate_tools(tools)
        if translated:
            kwargs["tools"] = translated

        response = await self._client.messages.create(**kwargs)

        tool_calls = [
            ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
            for block in response.content
            if getattr(block, "type", None) == "tool_use"
        ]

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            logger.warning(
                "model refused: %s", getattr(details, "category", None) if details else None
            )

        return LLMResult(
            text=self._text_of(response),
            tool_calls=tool_calls,
            stop_reason=response.stop_reason,
            model=response.model,
            usage=self._usage(response),
            refused=response.stop_reason == "refusal",
            # Echo content verbatim so thinking and tool_use blocks survive.
            assistant_message={"role": "assistant", "content": response.content},
        )

    def tool_result_messages(self, outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
        # Anthropic expects every tool result for a turn in ONE user message.
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": outcome.call.id,
                        "content": outcome.content,
                        "is_error": outcome.is_error,
                    }
                    for outcome in outcomes
                ],
            }
        ]

    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = "response",
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> tuple[dict[str, Any], LLMResult]:
        response = await self._client.messages.create(
            model=self.require_model(model),
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": "disabled"},
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
        )

        text = self._text_of(response)
        result = LLMResult(
            text=text,
            stop_reason=response.stop_reason,
            model=response.model,
            usage=self._usage(response),
            refused=response.stop_reason == "refusal",
        )
        return json.loads(text), result

    async def aclose(self) -> None:
        await self._client.close()
