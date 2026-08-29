"""Google Gemini backend.

Uses the `google-genai` SDK (`pip install google-genai`), which is the SDK Google
currently recommends - not the older `google-generativeai` package.

Two API surfaces exist in that SDK. This uses **`models.generate_content`**, the
stable one, rather than `interactions`, which the SDK itself flags at import time
as "experimental and may change in future versions". Everything the agent loop
needs is available on the stable surface: system instructions, multi-turn
history, function calling, JSON mode, temperature and token limits.

Shape differences from the OpenAI-style providers, all handled here so the agent
loop never sees them:

  - Roles are `user` and **`model`**, not `assistant`.
  - The system prompt is not a message; it is `config.system_instruction`.
  - Tool calls and tool results are *parts of content*, not separate message
    types. A tool result goes back as a `user` turn containing a
    `function_response` part.
  - Function calls have no id in the OpenAI sense, so calls are correlated by
    name. `ToolCall.id` is synthesised for the agent loop's bookkeeping.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from google import genai
from google.genai import types

from ...config import settings
from .base import LLMProvider, LLMResult, ToolCall, ToolOutcome, Usage

logger = logging.getLogger(__name__)

#: Finish reasons that mean the model declined, rather than ran out of room.
_REFUSAL_REASONS = {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"}


class GeminiProvider(LLMProvider):
    name = "gemini"

    def __init__(self) -> None:
        if not settings.gemini_key:
            # Raised at construction so the failure names the missing variable,
            # rather than surfacing later as an opaque 401 from Google.
            raise RuntimeError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set, so the Gemini provider "
                "cannot be used. Set it, or point this text_llm at another provider "
                "in the LLM registry (app/services/llm_registry.json)."
            )
        self._client = genai.Client(api_key=settings.gemini_key)

    # -- request building ---------------------------------------------------

    @staticmethod
    def _translate_tools(tools: list[dict[str, Any]] | None) -> list[types.Tool] | None:
        """Neutral {"name","description","parameters"} -> Gemini FunctionDeclaration.

        `parameters_json_schema` is used rather than `parameters` so the tool
        registry's plain JSON Schema (including `additionalProperties`) is passed
        through untouched instead of being coerced into Gemini's own Schema type.
        """
        if not tools:
            return None
        return [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        parameters_json_schema=tool.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    )
                    for tool in tools
                ]
            )
        ]

    @staticmethod
    def _to_contents(messages: list[dict[str, Any]]) -> list[types.Content]:
        """Agent-loop messages -> Gemini contents.

        Accepts both the neutral `{"role","content"}` form and the native
        `types.Content` objects this provider puts back into history itself, so
        the loop can append our own assistant turn without special-casing.
        """
        contents: list[types.Content] = []

        for message in messages:
            if isinstance(message, types.Content):
                contents.append(message)
                continue

            role = message.get("role", "user")
            content = message.get("content", "")

            # Tool results are appended by tool_result_messages() as ready-made
            # Content objects; anything reaching here is plain text.
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )

            contents.append(
                types.Content(
                    role="model" if role == "assistant" else "user",
                    parts=[types.Part(text=str(content))],
                )
            )

        return contents

    def _config(
        self,
        *,
        system: str,
        tools: list[dict[str, Any]] | None,
        max_tokens: int | None,
    ) -> types.GenerateContentConfig:
        config = types.GenerateContentConfig(
            system_instruction=system or None,
            temperature=settings.gemini_temperature,
            max_output_tokens=max_tokens or settings.max_tokens,
        )

        # Gemini 3.x reasons before answering and charges that reasoning to
        # max_output_tokens. Uncapped, a 400-token budget gets spent thinking and
        # the customer receives a sentence cut in half. A small explicit budget
        # makes the limit mean what the caller expects.
        budget = settings.gemini_thinking_budget
        if budget >= 0:
            config.thinking_config = types.ThinkingConfig(thinking_budget=budget)

            limit = config.max_output_tokens or 0
            if limit and limit <= budget:
                # Reasoning would consume the entire allowance, leaving nothing
                # for the answer - the reply comes back empty with MAX_TOKENS.
                logger.warning(
                    "max_tokens=%s is at or below GEMINI_THINKING_BUDGET=%s; the reply "
                    "will likely be empty. Raise max_tokens or lower the budget.",
                    limit,
                    budget,
                )

        translated = self._translate_tools(tools)
        if translated:
            config.tools = translated
            # The agent loop runs tools itself and feeds results back, so the
            # SDK's own execution loop must stay out of the way.
            config.automatic_function_calling = types.AutomaticFunctionCallingConfig(
                disable=True
            )
        return config

    # -- response parsing ---------------------------------------------------

    @staticmethod
    def _usage(response: Any) -> Usage:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return Usage()
        return Usage(
            input_tokens=getattr(usage, "prompt_token_count", 0) or 0,
            output_tokens=getattr(usage, "candidates_token_count", 0) or 0,
            cache_read_input_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
        )

    @staticmethod
    def _parse(response: Any) -> tuple[str, list[ToolCall], types.Content | None, str | None]:
        """Pull text, tool calls, the raw assistant turn, and the finish reason."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return "", [], None, None

        candidate = candidates[0]
        finish = getattr(candidate, "finish_reason", None)
        finish_reason = getattr(finish, "name", None) or (str(finish) if finish else None)

        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []

        texts: list[str] = []
        calls: list[ToolCall] = []

        for index, part in enumerate(parts):
            # Thought summaries are internal reasoning, never the customer reply.
            if getattr(part, "thought", None):
                continue
            if getattr(part, "text", None):
                texts.append(part.text)
            call = getattr(part, "function_call", None)
            if call is not None:
                args = call.args
                if not isinstance(args, dict):
                    args = {} if args is None else {"value": args}
                calls.append(
                    ToolCall(
                        # Gemini may omit an id; the loop needs a stable one.
                        id=getattr(call, "id", None) or f"{call.name}_{index}",
                        name=call.name,
                        arguments=dict(args),
                    )
                )

        return "".join(texts).strip(), calls, content, finish_reason

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
        target = self.require_model(model)

        response = await self._client.aio.models.generate_content(
            model=target,
            contents=self._to_contents(messages),
            config=self._config(system=system, tools=tools, max_tokens=max_tokens),
        )

        text, tool_calls, assistant_content, finish_reason = self._parse(response)

        return LLMResult(
            text=text,
            tool_calls=tool_calls,
            stop_reason=finish_reason,
            model=target,
            usage=self._usage(response),
            refused=finish_reason in _REFUSAL_REASONS,
            # Kept in Gemini's own form so the next turn replays it exactly,
            # including the function_call parts.
            assistant_message=assistant_content
            or types.Content(role="model", parts=[types.Part(text=text)]),
        )

    def tool_result_messages(self, outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
        """Gemini carries tool results as function_response parts on a user turn."""
        return [
            types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=outcome.call.name,
                            response=(
                                {"error": outcome.content}
                                if outcome.is_error
                                else {"result": outcome.content}
                            ),
                        )
                    )
                    for outcome in outcomes
                ],
            )
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
        response = await self._client.aio.models.generate_content(
            model=self.require_model(model),
            contents=[types.Content(role="user", parts=[types.Part(text=user)])],
            config=types.GenerateContentConfig(
                system_instruction=system or None,
                temperature=0,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
                # Native structured output - the model is constrained rather than
                # asked politely, so the result needs no repair.
                response_json_schema=schema,
            ),
        )

        text, _, _, finish_reason = self._parse(response)
        result = LLMResult(
            text=text,
            stop_reason=finish_reason,
            model=self.require_model(model),
            usage=self._usage(response),
        )
        return json.loads(text), result

    async def aclose(self) -> None:
        # The SDK's async client owns an httpx session; close it so shutdown is
        # clean and tests do not warn about unclosed transports.
        close = getattr(self._client.aio, "aclose", None)
        if close is not None:
            await close()
