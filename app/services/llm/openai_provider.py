"""OpenAI backend.

Added because the registry can name OpenAI models (`openai_gpt_5`) and there was
no backend able to serve them — the identifier resolved, then failed at provider
construction.

Deliberately a near-twin of the Groq provider: Groq implements OpenAI's
chat-completions API, so the request shape, the tool schema and the
`{"role": "tool", "tool_call_id": …}` result messages are identical. The two are
kept as separate classes rather than one shared base because they diverge on
exactly the things that matter — SDK, key, and which parameters each accepts —
and a shared base would hide that behind conditionals.

No model id appears in this file. The model always arrives from the resolver.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from openai import AsyncOpenAI

from ...config import settings
from .base import LLMProvider, LLMResult, ToolCall, ToolOutcome, Usage

logger = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        # Bounded for the same reason as the Groq client - see config.py.
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key or None,
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
            "model": self.require_model(model),
            "messages": [{"role": "system", "content": system}, *messages],
            # GPT-5 and the o-series reject `max_tokens` and want
            # `max_completion_tokens`; the older chat models accept both. Sending
            # the newer name only is the one form that works across all of them.
            "max_completion_tokens": max_tokens or settings.max_tokens,
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
        instructions = (
            f"{system}\n\nReturn a single JSON object matching this schema exactly. "
            "No prose, no markdown fence:\n"
            f"{json.dumps(schema, ensure_ascii=False)}"
        )

        response = await self._client.chat.completions.create(
            model=self.require_model(model),
            max_completion_tokens=max_tokens,
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
