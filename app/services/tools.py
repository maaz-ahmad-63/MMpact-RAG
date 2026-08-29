"""Tool registry and dispatcher.

Adding a tool means writing one `Tool(...)` below and nothing else - the
function-calling loop in services/agent.py discovers everything through
`build_tools()` and `execute()`.

Two gates decide whether a tool is offered on a given request:

  1. `available()` - is the thing this tool talks to configured at all?
     A tool whose endpoint is blank is never advertised to the model, so it can
     never be called and then fail.
  2. `tools_allowed`  - the whitelist from the prompt record the client
     registered, or the per-request override. An empty list means no tools.

Definitions use the provider-neutral shape {"name", "description", "parameters"};
each LLM provider translates them into its own schema.

None of these tools carry business logic. They are adapters onto the gateway's
REST API (block/unblock) or a client application's own endpoints (knowledge,
handoff). The decision always belongs to the system on the other side.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..schemas import ReplyRequest
from . import gateway, laravel

logger = logging.getLogger(__name__)

ToolHandler = Callable[[dict[str, Any], ReplyRequest], Awaitable[Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler
    #: Whether the dependency this tool needs is configured.
    available: Callable[[], bool] = field(default=lambda: True)

    def definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# -- handlers ---------------------------------------------------------------


def _require_whatsapp_id(request: ReplyRequest) -> str:
    whatsapp_id = request.whatsapp_id
    if not whatsapp_id:
        raise ValueError(
            "context.whatsapp_id is required to block or unblock, and was not supplied"
        )
    return whatsapp_id


def _target_contact(tool_input: dict[str, Any], request: ReplyRequest) -> str:
    """Who to block. Defaults to the person in this conversation.

    The model may name a contact explicitly, but the common case - "this person
    is abusive" - should not require it to repeat a phone number back correctly.
    """
    explicit = str(tool_input.get("contact") or "").strip()
    if explicit:
        return explicit
    contact = request.contact.phone or request.contact_id
    if not contact:
        raise ValueError("no contact to act on: neither tool input nor context supplied one")
    return str(contact)


async def _block_contact(tool_input: dict[str, Any], request: ReplyRequest) -> Any:
    result = await gateway.block_contact(
        whatsapp_id=_require_whatsapp_id(request),
        contact=_target_contact(tool_input, request),
        reason=str(tool_input.get("reason") or "unspecified"),
        blocked_by=f"ai-agent:{request.client_id}" if request.client_id else "ai-agent",
    )
    return {"status": "blocked", **result}


async def _unblock_contact(tool_input: dict[str, Any], request: ReplyRequest) -> Any:
    result = await gateway.unblock_contact(
        whatsapp_id=_require_whatsapp_id(request),
        contact=_target_contact(tool_input, request),
        reason=str(tool_input.get("reason") or "unspecified"),
    )
    return {"status": "unblocked", **result}


async def _rag_search(tool_input: dict[str, Any], request: ReplyRequest) -> Any:
    """Search the document collection, through the EXISTING RAG service.

    Calls app.rag.service.search() in process - the same function POST
    /api/rag/search calls. No embedding, chunking or SQL happens here, and the
    threshold and top_k are the RAG layer's own.

    Imported inside the function, not at module scope: app/rag pulls asyncpg and
    the embedding stack, and services/tools.py is imported at boot by every
    deployment, including ones with no database.
    """
    from ..rag import service as rag_service

    query = str(tool_input.get("query") or "").strip()
    if not query:
        return {"status": "error", "message": "query is required"}

    try:
        outcome = await rag_service.search(
            query=query,
            conversation_id=request.thread_id or None,
        )
    except Exception as error:  # noqa: BLE001 - the model gets a usable answer
        logger.warning("rag_search tool failed: %s", error)
        return {
            "status": "unavailable",
            "message": (
                "The document search is unavailable right now. Do not retry it. "
                "Answer from what you were given, or say you will check and follow up."
            ),
            "query": query,
        }

    return {
        "status": "ok",
        "query": query,
        "found": outcome["found"],
        "results": [
            {
                "content": item["content"],
                "source": (item.get("metadata") or {}).get("source"),
                "page": (item.get("metadata") or {}).get("page"),
                "similarity": item["similarity"],
            }
            for item in outcome["results"]
        ],
    }


async def _lookup_business_info(tool_input: dict[str, Any], request: ReplyRequest) -> Any:
    return await laravel.fetch_knowledge(request.whatsapp_id, str(tool_input.get("query", "")))


async def _request_human_handoff(tool_input: dict[str, Any], request: ReplyRequest) -> Any:
    data = await laravel.request_handoff(
        request.whatsapp_id,
        request.contact.model_dump(),
        str(tool_input.get("reason", "unspecified")),
        str(tool_input.get("summary", "")),
    )
    return {"status": "handoff_recorded", **data}


# -- registry ---------------------------------------------------------------

BLOCK_TOOL = "block_contact"
UNBLOCK_TOOL = "unblock_contact"
RAG_TOOL = "rag_search"
KNOWLEDGE_TOOL = "lookup_business_info"
HANDOFF_TOOL = "request_human_handoff"

REGISTRY: dict[str, Tool] = {
    tool.name: tool
    for tool in [
        Tool(
            name=BLOCK_TOOL,
            description=(
                "Block a contact so this business can never message them again and their "
                "messages stop being answered. Call this when the contact is abusive or "
                "threatening, sends spam or scam content, or clearly asks never to be "
                "contacted again. Blocking is not reversible by the customer - be sure."
            ),
            parameters=_object(
                {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short reason code, e.g. 'abusive', 'spam', 'opt_out', 'fraud'."
                        ),
                    },
                    "contact": {
                        "type": "string",
                        "description": (
                            "Phone number to block. Omit to block the person you are "
                            "currently talking to, which is almost always what you want."
                        ),
                    },
                },
                ["reason"],
            ),
            handler=_block_contact,
            available=lambda: settings.block_tools_available,
        ),
        Tool(
            name=UNBLOCK_TOOL,
            description=(
                "Remove a block, so this contact can be messaged and answered again. "
                "Call this only when you have been explicitly told the block was a "
                "mistake or has been lifted."
            ),
            parameters=_object(
                {
                    "reason": {
                        "type": "string",
                        "description": "Why the block is being lifted.",
                    },
                    "contact": {
                        "type": "string",
                        "description": "Phone number to unblock. Omit for the current contact.",
                    },
                },
                ["reason"],
            ),
            handler=_unblock_contact,
            available=lambda: settings.block_tools_available,
        ),
        Tool(
            name=RAG_TOOL,
            description=(
                "Search the business's uploaded document collection - manuals, "
                "policies, contracts, brochures - for passages relevant to a "
                "question. Use it when the answer would be written down in a "
                "document rather than held in the business's live systems."
            ),
            parameters=_object(
                {"query": {"type": "string", "description": "What to search for."}},
                ["query"],
            ),
            handler=_rag_search,
            # Offered only when ENABLE_RAG_TOOL is on AND a DSN is configured.
            # Advertising a tool that always fails is worse than not offering
            # it - see config.rag_tool_available.
            available=lambda: settings.rag_tool_available,
        ),
        Tool(
            name=KNOWLEDGE_TOOL,
            description=(
                "Look up the business's own information - products, services, pricing, "
                "policies, opening hours, FAQ entries. Call this whenever the customer "
                "asks about a specific fact you were not given. Never guess a business "
                "fact you could look up here."
            ),
            parameters=_object(
                {
                    "query": {
                        "type": "string",
                        "description": "What to look up, in the business's own terms.",
                    }
                },
                ["query"],
            ),
            handler=_lookup_business_info,
            available=lambda: settings.knowledge_tool_available,
        ),
        Tool(
            name=HANDOFF_TOOL,
            description=(
                "Escalate the conversation to a human agent. Call this when the customer "
                "explicitly asks for a person, is angry or complaining, raises a billing "
                "or legal issue, or when you cannot resolve their request. Tell the "
                "customer a colleague will follow up after calling this."
            ),
            parameters=_object(
                {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short reason code, e.g. 'customer_requested', 'complaint', "
                            "'cannot_resolve'."
                        ),
                    },
                    "summary": {
                        "type": "string",
                        "description": "One or two sentences summarising what the human needs to know.",
                    },
                },
                ["reason", "summary"],
            ),
            handler=_request_human_handoff,
            available=lambda: settings.handoff_tool_available,
        ),
    ]
}


def available_tools() -> list[str]:
    """Tool names whose dependency is configured, for /health and the boot log."""
    return sorted(name for name, tool in REGISTRY.items() if tool.available())


def build_tools(request: ReplyRequest, allowed: list[str] | None) -> list[dict[str, Any]]:
    """Definitions for the tools this request may use.

    `allowed` is the resolved whitelist: the per-request `tools_allowed` if the
    caller sent one, otherwise the prompt record's. `None` means the caller
    expressed no opinion, in which case every configured tool is offered.
    """
    if allowed is not None and not allowed:
        return []

    names = REGISTRY.keys() if allowed is None else allowed
    definitions: list[dict[str, Any]] = []

    for name in names:
        tool = REGISTRY.get(name)
        if tool is None:
            logger.warning("tools_allowed names an unknown tool: %s", name)
            continue
        if not tool.available():
            logger.debug("tool %s requested but its dependency is not configured", name)
            continue
        definitions.append(tool.definition())

    return definitions


async def execute(name: str, tool_input: dict[str, Any], request: ReplyRequest) -> tuple[str, bool]:
    """Run a tool. Returns (result_text, is_error).

    Errors come back as tool results rather than exceptions, so the model can
    recover in-conversation instead of the whole reply failing.
    """
    tool = REGISTRY.get(name)
    if tool is None:
        return f"Unknown tool: {name}", True

    try:
        result = await tool.handler(tool_input, request)
        return json.dumps(result, ensure_ascii=False, default=str), False
    except Exception as error:  # noqa: BLE001 - surfaced to the model, logged here
        logger.warning("tool %s failed: %s", name, error)
        return (
            f"The {name} tool is unavailable right now. Do not retry it; "
            "tell the customer you will check and follow up.",
            True,
        )
