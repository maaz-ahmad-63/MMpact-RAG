"""LangChain tools for the agent.

`rag_search` is a thin wrapper over the EXISTING retrieval layer - it calls
`app.rag.service.search()`, the same function that backs POST /api/rag/search.
No new vector store, no new embeddings, no new chunking, no SQL. The BGE-M3
model, the pgvector tables and the similarity threshold are all untouched.

The service's other tools (lookup_business_info, request_human_handoff,
block_contact, unblock_contact) are NOT redefined here. They already exist in
services/tools.py with their own availability gates, and `registry_tools()`
exposes them to the agent through that registry rather than duplicating them.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, tool
from pydantic import BaseModel, Field

from ..schemas import ReplyRequest
from ..services import tools as tool_registry
from .config import graph_settings

logger = logging.getLogger(__name__)

RAG_TOOL_NAME = "rag_search"

#: Per-request state for the tools, set by the executor before it runs them.
#: LangChain tools are plain callables with no request in scope, and threading a
#: ReplyRequest through every signature would change the @tool contract the
#: model sees. A ContextVar keeps the tool signature exactly as the model is
#: told it is, and is correct under concurrency - each asyncio task gets its own.
from contextvars import ContextVar

_current_request: ContextVar[ReplyRequest | None] = ContextVar(
    "graph_current_request", default=None
)
_last_sources: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "graph_last_sources", default=None
)


class _Scope:
    """Binds the request for the duration of one agent run."""

    def __init__(self, request: ReplyRequest) -> None:
        self._request = request
        self._tokens: list[Any] = []

    def __enter__(self) -> "_Scope":
        self._tokens = [
            _current_request.set(self._request),
            _last_sources.set([]),
        ]
        return self

    def __exit__(self, *_: Any) -> bool:
        _current_request.reset(self._tokens[0])
        _last_sources.reset(self._tokens[1])
        return False


def scope(request: ReplyRequest) -> _Scope:
    return _Scope(request)


def collected_sources() -> list[dict[str, Any]]:
    """Provenance from every rag_search call in this run, for the API response.

    Collected as a side effect rather than parsed back out of the tool's text:
    the model receives prose, and re-extracting document ids from prose would be
    a second, lossy source of truth for something the retrieval layer already
    returned structurally.
    """
    return list(_last_sources.get() or [])


class RagSearchInput(BaseModel):
    # Terse deliberately: this schema is serialised into EVERY provider call,
    # and a tool-calling turn makes two of them. The query-rewriting guidance
    # this used to spell out is stated once in AGENT_TOOL_RULES instead of once
    # per tool definition.
    query: str = Field(description="Specific search query, in the documents' words.")


@tool("rag_search", args_schema=RagSearchInput)
async def rag_search(query: str) -> str:
    """Search the knowledge base of uploaded company documents. Returns the
    passages that match, or says that nothing matched."""
    request = _current_request.get()
    if request is None:  # pragma: no cover - the executor always sets scope
        return "The knowledge base is not available for this request."

    # Imported here, not at module scope: app/rag pulls asyncpg and the
    # embedding stack, and a deployment with retrieval unconfigured must not
    # load them to answer a greeting.
    from ..rag import service as rag_service

    text = (query or "").strip()
    if not text:
        return "No search query was supplied. Provide a short, specific query."

    try:
        outcome = await rag_service.search(query=text)
    except Exception as error:  # noqa: BLE001 - the model gets a usable answer
        # Technical detail server-side; the model gets a plain instruction. A
        # raised exception here would end the run, when the honest outcome is a
        # reply that says the lookup failed.
        logger.warning("rag_search failed for %r: %s", text[:120], error, exc_info=True)
        return (
            "The knowledge-base search is temporarily unavailable. Do not retry it. "
            "Tell the user you cannot look that up right now and offer to follow up."
        )

    results = outcome.get("results") or []
    logger.info("TOOL rag_search | query=%r | results=%s", text[:120], len(results))

    if not results:
        return (
            f"No documents in the knowledge base matched '{text}'. "
            "Tell the user the available information does not cover this, and do "
            "not guess an answer."
        )

    # Budgeted, and trimmed by WHOLE passages. Retrieval already ranked these,
    # so stopping early drops the weakest - and half a passage is one the model
    # may go on to quote as though it were complete.
    budget = max(0, graph_settings.graph_rag_result_max_chars)
    sources = _last_sources.get()
    blocks: list[str] = []
    used = 0
    dropped = 0

    for position, item in enumerate(results, start=1):
        metadata = item.get("metadata") or {}
        title = metadata.get("source") or "Document"
        page = metadata.get("page")
        header = f"SOURCE {position}\nTitle: {title}"
        if page:
            header += f" (page {page})"
        block = f"{header}\nContent:\n{item['content']}"

        # `blocks and` - the top-ranked passage is always included, even if it
        # alone exceeds the budget. Returning nothing because the best match was
        # long would be a retrieval failure caused by a token setting.
        if blocks and used + len(block) > budget:
            dropped = len(results) - position + 1
            break

        blocks.append(block)
        used += len(block) + 2

        # Provenance is recorded only for passages actually QUOTED. A source the
        # model never saw must not appear in the answer's citations.
        if sources is not None:
            sources.append(
                {
                    "document_id": item.get("document_id"),
                    "chunk_id": item.get("chunk_id"),
                    "source": title,
                    "page": page,
                    "score": item.get("similarity"),
                }
            )

    if dropped:
        logger.info(
            "rag_search | %s of %s passages dropped at the %s-char budget "
            "(GRAPH_RAG_RESULT_MAX_CHARS)", dropped, len(results), budget,
        )

    return "\n\n".join(blocks)


def _registry_tool(name: str) -> BaseTool | None:
    """One of the service's EXISTING tools, presented to LangChain.

    Built from `services/tools.REGISTRY` so the description, the JSON schema and
    the availability gate all stay in one place. Executing goes back through
    `tool_registry.execute`, which already converts a failure into text the
    model can recover from rather than an exception.
    """
    entry = tool_registry.REGISTRY.get(name)
    if entry is None or not entry.available():
        return None

    async def _run(**arguments: Any) -> str:
        request = _current_request.get()
        if request is None:  # pragma: no cover
            return f"The {name} tool is not available for this request."
        content, is_error = await tool_registry.execute(name, arguments, request)
        logger.info("TOOL %s | ok=%s", name, not is_error)
        return content

    # The registry's own JSON schema, verbatim - langchain-core accepts a raw
    # dict, so the tool the model sees here is identical to the one it sees on
    # the /agent/reply path. One definition, two callers.
    return StructuredTool(
        name=entry.name,
        description=entry.description,
        args_schema=entry.parameters,
        coroutine=_run,
    )


def build_tools(request: ReplyRequest, allowed: list[str] | None) -> list[Any]:
    """The tools this request may use.

    `allowed` is the resolved whitelist - the per-request `tools_allowed` if the
    caller sent one, otherwise the prompt record's - applied exactly as
    services/tools.build_tools applies it on the /agent/reply path. None means
    the caller expressed no opinion, which offers every configured tool.

    An empty list means the model gets no tools at all and must answer directly.
    """
    if allowed is not None and not allowed:
        return []

    tools: list[Any] = []
    names = list(tool_registry.REGISTRY) if allowed is None else list(allowed)

    # rag_search is the LangChain-native one defined above, not the registry's
    # copy: the model needs the SOURCE-block format, which is specific to this
    # agent. The registry's version stays as it is for /agent/reply.
    if RAG_TOOL_NAME in names:
        entry = tool_registry.REGISTRY.get(RAG_TOOL_NAME)
        if entry is not None and entry.available():
            tools.append(rag_search)

    for name in names:
        if name == RAG_TOOL_NAME:
            continue
        built = _registry_tool(name)
        if built is not None:
            tools.append(built)
        elif name not in tool_registry.REGISTRY:
            logger.warning("tools_allowed names an unknown tool: %s", name)

    return tools


def definitions_for(tools: list[Any]) -> list[dict[str, Any]]:
    """Neutral {name, description, parameters} for logging and diagnostics."""
    out = []
    for item in tools:
        schema = getattr(item, "args_schema", None)
        out.append(
            {
                "name": item.name,
                "description": (item.description or "")[:160],
                "parameters": (
                    schema.model_json_schema()
                    if schema is not None and hasattr(schema, "model_json_schema")
                    else {}
                ),
            }
        )
    return out


def render_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, ensure_ascii=False, default=str)
