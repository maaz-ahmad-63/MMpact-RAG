"""Agent orchestration.

Responsibilities: resolve the thread and prompt, build the system prompt, run
the tool loop, persist the exchange, return a reply.

Not its responsibility: WhatsApp, sessions, QR codes, or any business rule. And
no branch below asks who the caller is - Laravel, another backend and a curl
one-liner all take the same path. `client_id` only ever selects *which* records
to read; it never selects behaviour.

The loop is provider-agnostic - see `services/llm/` for the Groq and Anthropic
backends.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from ..config import settings
from ..schemas import (
    Classification,
    ClassifyRequest,
    ClassifyResponse,
    HistoryTurn,
    LLMTrace,
    PromptRecord,
    ReplyRequest,
    ReplyResponse,
    Thread,
    ThreadRef,
    ToolCallRecord,
    Usage,
)
from ..store import prompts as prompt_store
from ..store import threads as thread_store
from . import llm
from . import models
from . import token_diagnostics
from . import tools as tool_registry
from .llm.base import ToolOutcome, Usage as LLMUsage
from .prompts import CLASSIFY_SYSTEM, PromptUnavailable, build_system_prompt, resolve_inline

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK = (
    "Sorry, I'm having trouble answering right now. A colleague will get back to you shortly."
)


async def shutdown() -> None:
    await llm.shutdown()


class PromptNotFound(Exception):
    """`prompt_id` named a record this client has not registered."""


async def resolve_thread(request: ReplyRequest) -> Thread | None:
    """Find or create the thread this reply belongs to.

    Three ways in, in order:
      1. an explicit `thread_id`
      2. `client_id` + `channel` + `contact_id`
      3. neither - the call is stateless, history comes from `context.history`

    A `thread_id` that does not exist is not an error: the caller may be
    replaying, and refusing would lose the message. It falls through to (2), and
    then to (3).
    """
    if request.thread_id:
        thread = await thread_store.get(request.thread_id)
        if thread is not None:
            if request.client_id and thread.client_id != request.client_id:
                # Threads are isolated by client. Silently serving another
                # client's history would leak one customer's conversation.
                raise PermissionError(
                    f"thread {request.thread_id} belongs to a different client_id"
                )
            return thread
        logger.warning("thread_id %s not found - resolving by identity", request.thread_id)

    contact_id = request.contact_id or request.contact.phone
    if request.client_id and contact_id:
        return await thread_store.get_or_create(
            ThreadRef(
                client_id=request.client_id,
                channel=request.channel or request.context.get("channel") or "whatsapp",
                contact_id=str(contact_id),
            )
        )

    return None


async def resolve_prompt(request: ReplyRequest) -> PromptRecord | None:
    """Load the registered prompt record, if one was named.

    An inline `system_prompt` wins over the record for the instruction text, but
    the record is still loaded when named - its `tools_allowed`, tone and
    language remain in force unless separately overridden.
    """
    if not request.prompt_id:
        return None
    if not request.client_id:
        raise PromptNotFound("prompt_id requires client_id - prompts are scoped per client")

    record = await prompt_store.get(request.client_id, request.prompt_id)
    if record is None:
        raise PromptNotFound(
            f"client {request.client_id} has no prompt registered as '{request.prompt_id}'. "
            "Register it with POST /agent/prompt, or send system_prompt inline."
        )
    return record


def resolve_tools_allowed(request: ReplyRequest, record: PromptRecord | None) -> list[str] | None:
    """The effective tool whitelist.

    Per-request override, else the prompt record's list, else None meaning "no
    opinion" - which offers every configured tool.
    """
    if request.tools_allowed is not None:
        return request.tools_allowed
    if record is not None:
        return record.tools_allowed
    return None


def _to_schema_usage(usage: LLMUsage) -> Usage:
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
    )


def _build_messages(request: ReplyRequest, thread: Thread | None) -> list[dict[str, Any]]:
    """Prior turns then the new message.

    The stored thread is the history when one resolved; `context.history` is the
    fallback for callers that keep history on their own side and send it inline.

    An EMPTY thread falls back too. A thread is created on first contact, so it
    exists but holds nothing while the caller's own record of the conversation
    may hold plenty - after a redeploy with a fresh volume, that is every
    thread. Preferring an empty thread over supplied history would silently
    start every conversation again from nothing.
    """
    history = thread.turns if thread is not None and thread.turns else request.history

    messages: list[dict[str, Any]] = []
    for turn in history:
        content = turn.content.strip()
        if content:
            messages.append({"role": turn.role, "content": content})

    # Conversations must open with a user turn.
    while messages and messages[0]["role"] != "user":
        messages.pop(0)

    messages.append({"role": "user", "content": request.message})
    return messages


def _build_trace(*, system, messages, sent, request, resolved, usage, llm_seconds,
                 provider_calls):
    """What was actually sent, for the caller's dm_ai_logs row.

    Observed, never reconstructed: `system` and `messages` are the objects handed
    to the provider on the last turn.

    `sent` is the snapshot taken immediately before the last provider call - the
    system turn and the conversation as one array, captured before the tool loop
    could append anything to it. Both `messages` and `all_prompt_text` are
    rendered from it, so the structured and text forms of this row are always
    the same payload.
    """
    reported = (
        usage.input_tokens
        or usage.output_tokens
        or usage.cache_read_input_tokens
        or usage.cache_creation_input_tokens
    )
    payload = sent if sent is not None else [{"role": "system", "content": system}, *messages]

    return LLMTrace(
        system_prompt=system,
        # The turn being answered. The full context is in `messages` below.
        user_prompt=request.message,
        messages=_serialisable(payload),
        all_prompt_text=_safe_prompt_text(payload),
        llm_ms=int(llm_seconds * 1000),
        provider_calls=provider_calls,
        usage_source="provider" if reported else "unavailable",
        model_requested=request.model,
        model_resolved=resolved.model,
    )


def _safe_prompt_text(messages) -> str | None:
    """`render_prompt_text`, but never able to break a reply.

    Audit logging is reporting. A row that cannot be rendered is worth less than
    a reply that was successfully generated, so a failure here costs the text
    and nothing else - the same rule the gateway applies to dm_ai_logs writes.
    """
    try:
        return render_prompt_text(messages)
    except Exception:  # noqa: BLE001 - logging must never break the reply
        logger.warning("could not render all_prompt_text", exc_info=True)
        return None


def render_prompt_text(messages) -> str:
    """The payload as role-labelled text, for dm_ai_logs.all_prompt_text_type.

    A rendering of the array that was sent - not a second assembly of the
    prompt. It is called with the same snapshot that populates `trace.messages`,
    so the two can never disagree and neither can drift from what the provider
    received.

    Content that is not a string (Gemini parts, tool-call structures) is
    serialised rather than skipped: an approximation of a turn is worth more in
    an audit log than a silent gap. Nothing here can raise - see the caller.
    """
    blocks: list[str] = []
    for message in _serialisable(messages):
        role = str(message.get("role", "unknown")).upper()
        content = message.get("content")

        if content is None or content == "":
            rendered = ""
        elif isinstance(content, str):
            rendered = content
        else:
            rendered = json.dumps(content, ensure_ascii=False, default=str)

        # Tool calls are part of what the model was sent on the next turn, so an
        # assistant turn that carries them is incomplete without them.
        tool_calls = message.get("tool_calls")
        if tool_calls:
            rendered = (
                f"{rendered}\n[tool_calls] "
                f"{json.dumps(tool_calls, ensure_ascii=False, default=str)}"
            ).strip()

        blocks.append(f"--- {role} ---\n{rendered}")

    return "\n\n".join(blocks)


def _serialisable(messages):
    """Messages as plain JSON.

    Gemini's turns are SDK objects, not dicts, and json.dumps would choke on them
    downstream. Anything not already a mapping is rendered as text rather than
    dropped - a readable approximation beats an empty log.
    """
    out = []
    for message in messages:
        if isinstance(message, dict):
            out.append(message)
        else:
            out.append({"role": getattr(message, "role", "unknown"), "content": str(message)})
    return out


async def generate_reply(request: ReplyRequest) -> ReplyResponse:
    started = time.perf_counter()

    thread = await resolve_thread(request)
    record = await resolve_prompt(request)

    # Laravel's `text_llm` arrives as `request.model`. The resolver is the only
    # thing that turns it into a provider and a model id - see services/models.py.
    resolved = models.resolve(request.model)
    provider = llm.get_provider(resolved.provider)
    max_tokens = request.max_tokens or settings.max_tokens

    logger.info(
        "client %s | LLM Identifier: %s | Provider: %s | Model: %s (max_tokens=%s)",
        request.client_id,
        resolved.identifier or "(registry default)",
        resolved.provider,
        resolved.model,
        max_tokens,
    )

    system = build_system_prompt(request, record, provider=resolved.provider)
    messages = _build_messages(request, thread)
    available_tools = tool_registry.build_tools(request, resolve_tools_allowed(request, record))

    # -- RAG Pre-filter & Single Context Fetch --------------------------------
    # Evaluates user query with a cheap rule-based pre-filter (<1 ms, 0 extra LLM calls).
    # Obvious casual/non-knowledge queries skip RAG completely. Eligible queries run a single RAG
    # search; if relevant results above threshold are found, context is attached. If unhelpful
    # or unavailable, falls back gracefully to main LLM without context.
    user_query = str(request.message or request.prompt or "").strip()
    if user_query and "retrieved_context" not in request.context:
        try:
            from ..rag import service as rag_service
        except Exception:  # noqa: BLE001 - RAG module optional
            rag_service = None

        from .rag_filter import should_skip_rag

        if should_skip_rag(user_query):
            logger.info("RAG skipped: casual query ('%s') - disabling tools & using condensed system prompt to save tokens", user_query)
            available_tools = []
            system = "You are a helpful customer support assistant. Reply politely and briefly. Respond only with the final message to the customer."
        elif rag_service is None:
            logger.info("RAG unavailable: module not loaded - falling back to LLM")
        else:
            logger.info("RAG checked: query eligible ('%s')", user_query)
            try:
                rag_result = await rag_service.build_context(
                    query=user_query,
                    conversation_id=thread.thread_id if thread else None,
                )
                if rag_result.get("found") and rag_result.get("context"):
                    request.context["retrieved_context"] = rag_result["context"]
                    logger.info(
                        "RAG relevant: context attached (%s chunks)",
                        rag_result.get("chunks_used", 0),
                    )
                    # Rebuild system prompt to include retrieved context
                    system = build_system_prompt(request, record, provider=resolved.provider)
                else:
                    logger.info("RAG not relevant: falling back to LLM without context")
            except Exception as error:  # noqa: BLE001 - RAG failure must never break main LLM
                logger.warning("RAG unavailable: falling back to LLM (%s)", error)

    usage = LLMUsage()
    tool_calls: list[ToolCallRecord] = []
    handoff_requested = False
    result = None

    # Time spent INSIDE provider calls, so the AI log records the LLM request
    # duration rather than the whole pipeline. A tool loop makes several
    # round-trips for one reply; they are summed, and `provider_calls` says how
    # many there were.
    llm_seconds = 0.0
    provider_calls = 0

    # The exact payload of the most recent provider call, captured immediately
    # before it is made. Every provider prepends the system turn to `messages`
    # the same way, and the tool loop appends to `messages` as it runs, so a
    # snapshot is the only thing that still describes one specific request once
    # the loop has moved on. Read back by _build_trace for both trace.messages
    # and dm_ai_logs.all_prompt_text_type.
    sent: list[dict[str, Any]] | None = None

    for iteration in range(settings.max_tool_iterations + 1):
        sent = [{"role": "system", "content": system}, *messages]

        if settings.log_token_breakdown:
            token_diagnostics.log_prompt_breakdown(
                model=resolved.model,
                system=system,
                messages=messages,
                tools=available_tools,
                max_tokens=max_tokens,
                provider_call=provider_calls + 1,
            )

        call_started = time.perf_counter()
        try:
            result = await provider.complete(
                system=system,
                messages=messages,
                tools=available_tools or None,
                max_tokens=max_tokens,
                model=resolved.model,
            )
        except Exception:
            # The caller logs the failure and needs the duration of the attempt
            # that failed, so account for it before the exception propagates.
            llm_seconds += time.perf_counter() - call_started
            provider_calls += 1
            raise
        llm_seconds += time.perf_counter() - call_started
        provider_calls += 1
        usage = usage + result.usage

        if result.refused:
            return ReplyResponse(
                reply=settings.fallback_reply or DEFAULT_FALLBACK,
                thread_id=thread.thread_id if thread else None,
                tool_calls_made=[call.name for call in tool_calls],
                provider=provider.name,
                model=result.model,
                stop_reason="refusal",
                handoff_requested=handoff_requested,
                tool_calls=tool_calls,
                usage=_to_schema_usage(usage),
                latency_ms=int((time.perf_counter() - started) * 1000),
                trace=_build_trace(
                    system=system, messages=messages, sent=sent, request=request,
                    resolved=resolved,
                    usage=usage, llm_seconds=llm_seconds, provider_calls=provider_calls,
                ),
            )

        if not result.tool_calls:
            break

        if iteration == settings.max_tool_iterations:
            logger.warning(
                "tool iteration limit (%s) reached; answering with what we have",
                settings.max_tool_iterations,
            )
            break

        messages.append(result.assistant_message)

        outcomes: list[ToolOutcome] = []
        for call in result.tool_calls:
            content, is_error = await tool_registry.execute(call.name, call.arguments, request)

            if call.name == tool_registry.HANDOFF_TOOL and not is_error:
                handoff_requested = True

            tool_calls.append(
                ToolCallRecord(
                    name=call.name,
                    input=call.arguments,
                    ok=not is_error,
                    error=content if is_error else None,
                )
            )
            outcomes.append(ToolOutcome(call=call, content=content, is_error=is_error))

        messages.extend(provider.tool_result_messages(outcomes))

    reply = result.text if result else ""
    if not reply:
        logger.warning(
            "empty reply (stop_reason=%s)", result.stop_reason if result else "no_response"
        )
        reply = settings.fallback_reply or DEFAULT_FALLBACK

    # Persist the exchange only when a real answer was produced, so a provider
    # outage does not write a fallback apology into the conversation history.
    if thread is not None and request.persist and result is not None and result.text:
        await thread_store.append(
            thread,
            [
                HistoryTurn(role="user", content=request.message),
                HistoryTurn(role="assistant", content=reply),
            ],
        )

    return ReplyResponse(
        reply=reply,
        thread_id=thread.thread_id if thread else None,
        tool_calls_made=[call.name for call in tool_calls],
        provider=provider.name,
        model=result.model if result else resolved.model,
        stop_reason=result.stop_reason if result else None,
        handoff_requested=handoff_requested,
        tool_calls=tool_calls,
        usage=_to_schema_usage(usage),
        latency_ms=int((time.perf_counter() - started) * 1000),
        trace=_build_trace(
            system=system, messages=messages, sent=sent, request=request, resolved=resolved,
            usage=usage, llm_seconds=llm_seconds, provider_calls=provider_calls,
        ),
    )


# -- conversation intelligence ---------------------------------------------


def _strict_schema(model: type) -> dict[str, Any]:
    """Pydantic JSON schema, tightened for structured-output APIs."""
    schema = model.model_json_schema()

    def tighten(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
            node.pop("title", None)
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for value in node:
                tighten(value)

    tighten(schema)
    return schema


async def classify(request: ClassifyRequest) -> ClassifyResponse:
    """Structured analysis of an inbound message, for Laravel routing/reporting."""
    started = time.perf_counter()

    # Classification is an internal call, so it usually carries no identifier and
    # falls back to the registry's declared default. When Laravel does send one,
    # it is honoured exactly as on the reply path.
    resolved = models.resolve(getattr(request, "model", None))
    provider = llm.get_provider(resolved.provider)
    logger.info(
        "classify | LLM Identifier: %s | Provider: %s | Model: %s",
        resolved.identifier or "(registry default)",
        resolved.provider,
        resolved.model,
    )

    system = CLASSIFY_SYSTEM
    if request.labels:
        system += "\n\nUse one of these intent labels when it fits: " + ", ".join(request.labels)
    if request.context:
        system += "\n\nAccount context:\n" + json.dumps(
            request.context, ensure_ascii=False, default=str
        )

    data, result = await provider.complete_json(
        system=system,
        user=request.message,
        schema=_strict_schema(Classification),
        schema_name="classification",
        model=resolved.model,
    )

    return ClassifyResponse(
        classification=Classification.model_validate(data),
        provider=provider.name,
        model=result.model,
        usage=_to_schema_usage(result.usage),
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
