"""Groq backend (default).

Groq serves an OpenAI-compatible chat-completions API, so tools use the
`{"type": "function", "function": {...}}` shape and tool results come back as
`{"role": "tool", "tool_call_id": ...}` messages.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from groq import AsyncGroq

from ...config import settings
from .base import LLMProvider, LLMResult, ToolCall, ToolOutcome, Usage

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    name = "groq"

    def __init__(self) -> None:
        # Bounded deliberately - the SDK default is 600s with 2 retries, which
        # outlives the gateway's own 60s cutoff many times over. See
        # settings.llm_timeout_s.
        self._client = AsyncGroq(
            api_key=settings.groq_api_key or None,
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
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            for tool in tools
        ]

    def _kwargs(self, max_tokens: int | None, model: str | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            # The model always comes from the resolver - see services/models.py.
            "model": self.require_model(model),
            "max_tokens": max_tokens or settings.max_tokens,
            "temperature": settings.groq_temperature,
        }
        # Only some Groq models accept reasoning controls; send it when configured.
        if settings.groq_reasoning_effort:
            kwargs["reasoning_effort"] = settings.groq_reasoning_effort
        return kwargs

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def _usage(response: Any) -> Usage:
        usage = getattr(response, "usage", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    @staticmethod
    def _parse_tool_calls(message: Any) -> tuple[list[ToolCall], list[dict[str, Any]]]:
        calls: list[ToolCall] = []
        raw: list[dict[str, Any]] = []

        for item in getattr(message, "tool_calls", None) or []:
            function = getattr(item, "function", None)
            if function is None:
                continue
            arguments_text = getattr(function, "arguments", "") or "{}"
            try:
                arguments = json.loads(arguments_text)
            except json.JSONDecodeError:
                logger.warning("tool arguments were not valid JSON: %s", arguments_text)
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}

            calls.append(ToolCall(id=item.id, name=function.name, arguments=arguments))
            raw.append(
                {
                    "id": item.id,
                    "type": "function",
                    "function": {"name": function.name, "arguments": arguments_text},
                }
            )

        return calls, raw

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
        payload: dict[str, Any] = {
            **self._kwargs(max_tokens, model),
            "messages": [{"role": "system", "content": system}, *messages],
        }

        translated = self._translate_tools(tools)
        if translated:
            payload["tools"] = translated
            payload["tool_choice"] = "auto"

        response = await self._client.chat.completions.create(**payload)
        choice = response.choices[0]
        message = choice.message

        tool_calls, raw_tool_calls = self._parse_tool_calls(message)
        content = (message.content or "").strip()

        assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if raw_tool_calls:
            assistant_message["tool_calls"] = raw_tool_calls

        return LLMResult(
            text=content,
            tool_calls=tool_calls,
            stop_reason=choice.finish_reason,
            model=response.model,
            usage=self._usage(response),
            refused=False,
            assistant_message=assistant_message,
        )

    def tool_result_messages(self, outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
        return [
            {
                "role": "tool",
                "tool_call_id": outcome.call.id,
                "name": outcome.call.name,
                "content": outcome.content,
            }
            for outcome in outcomes
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
        # JSON mode is supported across Groq's chat models; the schema is carried
        # in the prompt and the result is validated by the caller.
        instructions = (
            f"{system}\n\nReturn a single JSON object matching this schema exactly. "
            "No prose, no markdown fence:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )

        response = await self._client.chat.completions.create(
            model=self.require_model(model),
            max_tokens=max_tokens,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": user},
            ],
        )

        choice = response.choices[0]
        text = (choice.message.content or "").strip()
        result = LLMResult(
            text=text,
            stop_reason=choice.finish_reason,
            model=response.model,
            usage=self._usage(response),
        )
        return json.loads(text), result

    async def aclose(self) -> None:
        await self._client.close()
