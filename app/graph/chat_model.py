"""A LangChain chat model backed by THIS SERVICE'S existing providers.

Why an adapter instead of langchain-groq / langchain-openai / …
--------------------------------------------------------------
Installing those was measured, not assumed:

    pip install --dry-run langchain langchain-groq
    -> Would install groq-0.37.1 langchain-1.3.15 langchain-core-1.5.6
                     langchain-groq-1.1.3 langgraph-1.2.11

Two disqualifiers. `langchain` depends on `langgraph`, which this task forbids.
And `langchain-groq` upgrades groq 0.15.0 -> 0.37.1, swapping the SDK the
production /agent/reply path already runs on.

Beyond dependencies, the four provider classes in services/llm/ carry real
behaviour that a stock integration would discard: bounded 45s timeouts with one
retry (the SDK default is 600s x2), Groq's reasoning-effort control, Gemini's
thinking budget, refusal detection, and `require_model()` - which REFUSES to run
without a model id so no provider can silently pick one.

So this wraps them. `services/models.resolve(text_llm)` still chooses the model,
exactly as before; this class only presents that choice through LangChain's
interface so `bind_tools()` and LangChain messages work.

Message fidelity
----------------
LangChain messages are the canonical form here, but a provider's own assistant
turn (Groq's tool_calls array, Gemini's parts) cannot be reconstructed from an
AIMessage without loss. So the native turn is carried on
`AIMessage.additional_kwargs["__native__"]` and handed straight back on the next
call - the model sees byte-for-byte what it produced.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator, Sequence

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from ..config import settings
from ..services import llm
from ..services.llm.base import ToolCall, ToolOutcome

logger = logging.getLogger(__name__)

#: Where a provider's own assistant turn rides on an AIMessage.
NATIVE_KEY = "__native__"


class ServiceChatModel(BaseChatModel):
    """LangChain's chat interface over services/llm.

    Holds a provider NAME and a model id, never a model of its own: both come
    from services/models.resolve() and are passed in per request.
    """

    provider_name: str
    model_name: str
    max_tokens: int | None = None

    #: Neutral tool definitions - {"name", "description", "parameters"} - set by
    #: bind_tools(). Each provider translates them into its own schema.
    tool_definitions: list[dict[str, Any]] | None = None

    @property
    def _llm_type(self) -> str:
        return f"service-{self.provider_name}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # No key, no DSN. Names and numbers only - this ends up in LangChain's
        # own logging and callbacks.
        return {"provider": self.provider_name, "model": self.model_name}

    def bind_tools(self, tools: Sequence[BaseTool | dict[str, Any]], **kwargs: Any):
        """Give the model tools it may call.

        Returns a NEW instance rather than mutating: a bound model is a value in
        LangChain, and one request's tools must never leak into another's.
        """
        definitions: list[dict[str, Any]] = []
        for item in tools:
            if isinstance(item, dict):
                definitions.append(item)
                continue
            schema = item.args_schema
            parameters = (
                schema.model_json_schema()
                if schema is not None and hasattr(schema, "model_json_schema")
                else {"type": "object", "properties": {}}
            )
            # Providers reject unknown JSON-schema keys; strip what pydantic adds
            # for humans and keep the shape the function-calling APIs accept.
            parameters.pop("title", None)
            for field in (parameters.get("properties") or {}).values():
                if isinstance(field, dict):
                    field.pop("title", None)
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            parameters["additionalProperties"] = False
            definitions.append(
                {
                    "name": item.name,
                    "description": item.description or "",
                    "parameters": parameters,
                }
            )

        return self.__class__(
            provider_name=self.provider_name,
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            tool_definitions=definitions or None,
        )

    # -- message translation ------------------------------------------------

    def _to_native(self, messages: list[BaseMessage]) -> tuple[str, list[dict[str, Any]]]:
        """LangChain messages -> (system text, the provider's message array).

        Consecutive ToolMessages are grouped and handed to the provider's own
        `tool_result_messages`, because each backend carries results
        differently: OpenAI-shaped ones want one {"role":"tool"} per call,
        Anthropic wants a single user turn holding every tool_result block.
        """
        provider = llm.get_provider(self.provider_name)
        system_parts: list[str] = []
        native: list[dict[str, Any]] = []
        pending: list[ToolOutcome] = []

        def flush() -> None:
            if pending:
                native.extend(provider.tool_result_messages(pending))
                pending.clear()

        for message in messages:
            if isinstance(message, SystemMessage):
                flush()
                system_parts.append(str(message.content))
            elif isinstance(message, HumanMessage):
                flush()
                native.append({"role": "user", "content": str(message.content)})
            elif isinstance(message, ToolMessage):
                pending.append(
                    ToolOutcome(
                        call=ToolCall(
                            id=str(message.tool_call_id),
                            name=str(message.name or ""),
                            arguments={},
                        ),
                        content=str(message.content),
                        is_error=bool(message.additional_kwargs.get("is_error")),
                    )
                )
            elif isinstance(message, AIMessage):
                flush()
                remembered = message.additional_kwargs.get(NATIVE_KEY)
                if remembered is not None:
                    # The provider's own turn, replayed exactly.
                    native.append(remembered)
                elif message.content:
                    native.append({"role": "assistant", "content": str(message.content)})
            else:  # pragma: no cover - LangChain has no other concrete role here
                flush()
                native.append({"role": "user", "content": str(message.content)})

        flush()
        return "\n\n".join(system_parts), native

    @staticmethod
    def _to_ai_message(result: Any) -> AIMessage:
        """A provider LLMResult -> an AIMessage LangChain's loop understands."""
        return AIMessage(
            content=result.text or "",
            tool_calls=[
                {"name": call.name, "args": call.arguments, "id": call.id}
                for call in result.tool_calls
            ],
            additional_kwargs={NATIVE_KEY: result.assistant_message}
            if result.assistant_message is not None
            else {},
            response_metadata={
                "model": result.model,
                "stop_reason": result.stop_reason,
                "refused": result.refused,
                "provider": result.model and None,
            },
            usage_metadata={
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.input_tokens + result.usage.output_tokens,
            },
        )

    # -- generation ---------------------------------------------------------

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        provider = llm.get_provider(self.provider_name)
        system, native = self._to_native(messages)

        result = await provider.complete(
            system=system,
            messages=native,
            tools=self.tool_definitions or None,
            max_tokens=self.max_tokens or settings.max_tokens,
            # Always explicit. The provider refuses to guess, which is the point.
            model=self.model_name,
        )
        return ChatResult(generations=[ChatGeneration(message=self._to_ai_message(result))])

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Sync path. Not used: the whole service is async, and every provider
        client is an async client. Present because BaseChatModel requires it."""
        raise NotImplementedError(
            "ServiceChatModel is async-only. Use ainvoke/_agenerate - the "
            "provider SDKs underneath are async clients."
        )


def native_payload(system: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The array as sent, system turn included. For the dm_ai_logs trace."""
    return [{"role": "system", "content": system}, *messages]


def describe(messages: list[BaseMessage]) -> str:
    """Compact one-line summary of a message list, for debug logging only."""
    return json.dumps([type(m).__name__ for m in messages])
