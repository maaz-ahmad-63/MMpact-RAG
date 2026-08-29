"""The LangChain agent. One LLM decides everything.

    user message
         |
         v
    ServiceChatModel.bind_tools([rag_search, …])     <- Laravel's model
         |
    +----+----------------------------+
    |                                 |
    v                                 v
 answers directly            emits a tool call
 (1 LLM call, 0 RAG)               |
                                   v
                          existing rag/service.search()
                                   |
                                   v
                          ToolMessage -> same LLM
                                   |
                                   v
                             final answer

THERE IS NO INTENT ANALYZER. No classification call, no `if intent == …`, no
second model asked "what does the user want". The model that answers is the
model that decides, and it decides by calling a tool or not calling one - which
is what tool-calling models are for.

Why this is cheaper
-------------------
The removed design spent one structured-output call on every request before any
work happened. Here a greeting costs exactly one call, and a knowledge question
costs two (decide-and-call, then answer) - the second only when the model
actually asked for a lookup.

`AgentExecutor` is not used because it does not exist in the installed
dependency set: `langchain-core` is present, `langchain` is not, and installing
it pulls in `langgraph` (measured - see chat_model.py). This loop is the
tool-calling contract those abstractions implement, over the same primitives.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..config import settings
from ..schemas import HistoryTurn, LLMTrace, ThreadRef, Usage
from ..services import agent as agent_service
from ..services import models
from ..services.prompts import build_system_prompt
from ..store import threads as thread_store
from . import lc_tools, store
from .config import graph_settings
from .chat_model import NATIVE_KEY, ServiceChatModel
from .observability import Tracer, new_run_id
from .prompts import AGENT_TOOL_RULES
from .schemas import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


async def _resolve_conversation(request: ChatRequest):
    """Conversation id and prior turns, from the EXISTING thread store.

    Unchanged from before and identical to services/agent.resolve_thread:
    threads are keyed by (client_id, channel, contact_id), history comes from
    the thread when it has turns and from `context.history` otherwise, and
    THREAD_MAX_TURNS still governs how many are retained.
    """
    thread = None
    if request.thread_id:
        thread = await thread_store.get(request.thread_id)
        if thread is not None and request.client_id and thread.client_id != request.client_id:
            raise PermissionError(
                f"thread {request.thread_id} belongs to a different client_id"
            )

    if thread is None:
        contact_id = request.contact_id or request.contact.phone
        if request.client_id and contact_id:
            thread = await thread_store.get_or_create(
                ThreadRef(
                    client_id=request.client_id,
                    channel=request.channel or request.context.get("channel") or "whatsapp",
                    contact_id=str(contact_id),
                )
            )

    turns = thread.turns if thread is not None and thread.turns else request.history
    conversation_id = request.conversation_id or (thread.thread_id if thread else None)
    return thread, conversation_id, turns


def _recent(turns, pairs: int):
    """The last `pairs` user turns and the last `pairs` assistant turns.

    Order is preserved: the survivors are returned in their original sequence,
    so the model still reads the conversation forwards. Counting per role rather
    than taking a flat tail is what makes "3 and 3" mean 3 and 3 even when a
    customer sends several messages in a row - see GRAPH_HISTORY_PAIRS.
    """
    if pairs <= 0 or not turns:
        return turns

    quota = {"user": pairs, "assistant": pairs}
    keep: set[int] = set()
    for turn in reversed(turns):
        role = getattr(turn, "role", "user")
        if quota.get(role, 0) > 0:
            quota[role] -= 1
            keep.add(id(turn))
        if not any(quota.values()):
            break

    selected = [turn for turn in turns if id(turn) in keep]

    # If the window opens on an assistant turn, pull in the user turn it was
    # replying to. Without this the caller below drops that assistant turn
    # entirely - a conversation must open with a user turn - and the reply the
    # follow-up question depends on is lost. Costs at most one extra turn, and
    # only in the case that would otherwise lose a whole exchange.
    if selected and getattr(selected[0], "role", "user") == "assistant":
        first = turns.index(selected[0])
        for index in range(first - 1, -1, -1):
            if getattr(turns[index], "role", "user") == "user":
                selected.insert(0, turns[index])
                break

    return selected


def _history_messages(turns) -> list[BaseMessage]:
    """Prior turns as LangChain messages, capped by GRAPH_HISTORY_PAIRS.

    A conversation must open with a user turn, so leading assistant turns are
    dropped - the same normalisation services/agent._build_messages applies.
    """
    turns = _recent(turns, graph_settings.graph_history_pairs)
    messages: list[BaseMessage] = []
    for turn in turns:
        content = (turn.content or "").strip()
        if not content:
            continue
        messages.append(
            HumanMessage(content) if turn.role == "user" else AIMessage(content)
        )
    while messages and not isinstance(messages[0], HumanMessage):
        messages.pop(0)
    return messages


def _trace_for(chat: ServiceChatModel, messages: list[BaseMessage], request: ChatRequest,
               resolved, tracer: Tracer) -> LLMTrace:
    """The dm_ai_logs payload: what was actually sent on the last call.

    Rendered from the SAME translation the provider received, so the structured
    and text forms of the row can never disagree with each other or with the
    request. Usage and timing are the run's totals, because one dm_ai_logs row
    describes the whole exchange rather than one round-trip.
    """
    system, native = chat._to_native(messages)  # noqa: SLF001 - same package
    payload = [{"role": "system", "content": system}, *native]
    return LLMTrace(
        system_prompt=system,
        user_prompt=request.message,
        messages=agent_service._serialisable(payload),  # noqa: SLF001
        all_prompt_text=agent_service._safe_prompt_text(payload),  # noqa: SLF001
        llm_ms=tracer.llm_ms,
        provider_calls=tracer.provider_calls,
        usage_source=tracer.usage_source,
        model_requested=request.model,
        model_resolved=resolved.model,
    )


async def run_chat(request: ChatRequest) -> ChatResponse:
    """Answer one message with the LangChain agent.

    Failure policy is unchanged: an exception inside the loop closes the run as
    'failed' and returns the configured fallback rather than a 500. The router
    still translates a missing prompt or an unknown model into 4xx before this
    is entered.
    """
    started = time.perf_counter()
    run_id = new_run_id()
    tracer = Tracer(run_id)

    # Laravel's `text_llm` -> provider + real model id. UNCHANGED: this is the
    # same resolver /agent/reply uses, and it is the only thing that chooses a
    # model. No model id appears anywhere in this package.
    resolved = models.resolve(request.model)
    thread, conversation_id, turns = await _resolve_conversation(request)
    record = await agent_service.resolve_prompt(request)

    logger.info(
        "REQUEST RECEIVED | run=%s client=%s conversation=%s | LLM Identifier: %s | "
        "Provider: %s | Model: %s | query=%r",
        run_id, request.client_id, conversation_id,
        resolved.identifier or "(registry default)", resolved.provider, resolved.model,
        request.message[:200],
    )

    await store.start_run(
        run_id=run_id,
        conversation_id=conversation_id,
        user_id=request.user_id,
        client_id=request.client_id,
        user_message=request.message,
        model_identifier=resolved.identifier,
        model_provider=resolved.provider,
        model_name=resolved.model,
    )

    # The CLIENT's own prompt, assembled by the existing composer - BASE_RULES,
    # tone, language, the facts block. AGENT_TOOL_RULES is appended: it says how
    # to decide about tools and nothing about the business.
    system = build_system_prompt(request, record, provider=resolved.provider)
    system = f"{system}\n\n{AGENT_TOOL_RULES}"

    messages: list[BaseMessage] = [SystemMessage(system)]
    messages.extend(_history_messages(turns))
    messages.append(HumanMessage(request.message))

    allowed = agent_service.resolve_tools_allowed(request, record)
    tools = lc_tools.build_tools(request, allowed)
    chat = ServiceChatModel(
        provider_name=resolved.provider,
        model_name=resolved.model,
        max_tokens=request.max_tokens or settings.max_tokens,
    )
    model = chat.bind_tools(tools) if tools else chat

    logger.info(
        "AGENT STARTED | run=%s tools=%s", run_id, [item.name for item in tools] or "none"
    )

    status = "completed"
    error_text: str | None = None
    answer = ""
    tool_calls_made: list[dict[str, Any]] = []
    handoff_requested = False
    sources: list[dict[str, Any]] = []
    reply: AIMessage | None = None

    try:
        with lc_tools.scope(request):
            # MAX_TOOL_ITERATIONS, the setting the reply path already honours.
            for iteration in range(settings.max_tool_iterations + 1):
                async with tracer.step(
                    "agent_llm" if iteration == 0 else "agent_llm_followup", step_type="llm"
                ) as step:
                    call_started = time.perf_counter()
                    reply = await model.ainvoke(messages)
                    latency_ms = int((time.perf_counter() - call_started) * 1000)

                    usage = reply.usage_metadata or {}
                    await tracer.llm_call(
                        node_name="agent_llm",
                        provider=resolved.provider,
                        model=reply.response_metadata.get("model") or resolved.model,
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        latency_ms=latency_ms,
                    )
                    step.note(
                        iteration=iteration,
                        tool_calls=[call["name"] for call in reply.tool_calls],
                        answered=not reply.tool_calls,
                        stop_reason=reply.response_metadata.get("stop_reason"),
                    )

                messages.append(reply)

                if not reply.tool_calls:
                    break

                if iteration == settings.max_tool_iterations:
                    logger.warning(
                        "run=%s tool iteration limit (%s) reached; answering with what "
                        "we have", run_id, settings.max_tool_iterations,
                    )
                    break

                for call in reply.tool_calls:
                    name = call["name"]
                    arguments = call.get("args") or {}
                    async with tracer.step("tool_executor", step_type="tool") as step:
                        tool_started = time.perf_counter()
                        chosen = next((item for item in tools if item.name == name), None)
                        if chosen is None:
                            # The model named a tool it was not given. Told, not
                            # crashed - it can recover in-conversation.
                            content = f"Unknown tool: {name}. Do not call it again."
                            ok = False
                        else:
                            try:
                                content = str(await chosen.ainvoke(arguments))
                                ok = True
                            except Exception as error:  # noqa: BLE001
                                logger.warning(
                                    "run=%s tool %s raised: %s", run_id, name, error,
                                    exc_info=True,
                                )
                                content = (
                                    f"The {name} tool failed. Do not retry it; tell the "
                                    "user you will check and follow up."
                                )
                                ok = False

                        tool_ms = int((time.perf_counter() - tool_started) * 1000)
                        step.note(tool=name, ok=ok, latency_ms=tool_ms)
                        await tracer.tool_call(
                            tool_name=name,
                            arguments=arguments,
                            result=content,
                            status="ok" if ok else "error",
                            latency_ms=tool_ms,
                            error=None if ok else content,
                        )

                    tool_calls_made.append({"name": name, "arguments": arguments, "ok": ok})
                    if name == "request_human_handoff" and ok:
                        handoff_requested = True

                    messages.append(
                        ToolMessage(
                            content=content,
                            tool_call_id=call.get("id") or name,
                            name=name,
                            additional_kwargs={} if ok else {"is_error": True},
                        )
                    )

            sources = lc_tools.collected_sources()
            answer = (reply.content if reply else "") or ""
            if isinstance(answer, list):  # some providers return content blocks
                answer = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in answer
                )
            answer = answer.strip()

    except Exception as error:  # noqa: BLE001 - a run failure is data, not a 500
        status = "failed"
        error_text = f"{type(error).__name__}: {error}"
        logger.exception("run=%s agent failed", run_id)

    if not answer:
        if status == "completed":
            logger.warning("run=%s empty reply - using the configured fallback", run_id)
        answer = settings.fallback_reply or agent_service.DEFAULT_FALLBACK

    # Persist only a real answer, so a provider outage does not write a fallback
    # apology into the conversation history. Same rule as /agent/reply.
    if thread is not None and request.persist and status == "completed" and answer:
        await thread_store.append(
            thread,
            [
                HistoryTurn(role="user", content=request.message),
                HistoryTurn(role="assistant", content=answer),
            ],
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    used_rag = any(call["name"] == lc_tools.RAG_TOOL_NAME for call in tool_calls_made)
    # `action` is DERIVED from what the model actually did - it is not a
    # classification and nothing was asked to produce it. The dashboard's
    # existing columns keep working without an intent analyzer behind them.
    action = (
        "human_handoff" if handoff_requested
        else "rag_search" if used_rag
        else "tool" if tool_calls_made
        else "direct_answer"
    )

    # The query the MODEL wrote when it called the tool - not one a classifier
    # produced. Surfaced so a developer can see what was actually searched.
    search_query = next(
        (
            str((call["arguments"] or {}).get("query") or "")
            for call in tool_calls_made
            if call["name"] == lc_tools.RAG_TOOL_NAME
        ),
        None,
    )

    trace = _trace_for(chat, messages, request, resolved, tracer)

    logger.info(
        "LLM RESPONSE | run=%s status=%s action=%s tools=%s sources=%s | "
        "tokens=%s/%s calls=%s | TOTAL LATENCY %sms",
        run_id, status, action, [c["name"] for c in tool_calls_made], len(sources),
        tracer.input_tokens, tracer.output_tokens, tracer.provider_calls, latency_ms,
    )

    await store.finish_run(
        run_id=run_id,
        status=status,
        intent=None,  # no classifier produces one any more
        action=action,
        confidence=None,
        requires_rag=used_rag,
        search_query=search_query,
        final_answer=answer,
        handoff_requested=handoff_requested,
        latency_ms=latency_ms,
        error=error_text,
        model_provider=resolved.provider,
        model_name=resolved.model,
        input_tokens=tracer.input_tokens,
        output_tokens=tracer.output_tokens,
    )

    return ChatResponse(
        run_id=run_id,
        reply=answer,
        conversation_id=conversation_id,
        thread_id=conversation_id,
        action=action,
        requires_rag=used_rag,
        search_query=search_query,
        provider=resolved.provider,
        model=resolved.model,
        sources=sources,
        tool_calls=[
            {"name": call["name"], "ok": call["ok"]} for call in tool_calls_made
        ],
        handoff_requested=handoff_requested,
        usage=Usage(
            input_tokens=tracer.input_tokens, output_tokens=tracer.output_tokens
        ),
        usage_by_node=tracer.usage_by_node,
        trace=trace,
        latency_ms=latency_ms,
        status=status,
        error=error_text,
    )
