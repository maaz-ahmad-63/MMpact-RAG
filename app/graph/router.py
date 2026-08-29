"""Graph API: POST /api/v1/agent/chat and the run lookup.

ADDITIVE. /agent/reply, /agent/classify, /agent/thread/*, /agent/prompt/*,
/v1/chat/completions and /api/rag/* are untouched - a caller that never sends a
request here sees no difference at all.

Auth is the SERVICE'S EXISTING mechanism: `Depends(require_token)` on the
router, the same shared bearer every other authenticated route uses. No second
auth system, and no route here is public.

Error translation reuses app/routers/agent.py:_translate, so a Groq timeout
produces the same status code on this path as on the reply path.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..config import settings
from ..routers.agent import _translate
from ..security import require_token
from ..services.agent import PromptNotFound
from ..services.models import UnknownModel
from ..services.prompts import PromptUnavailable
from . import agent as lc_agent
from . import store
from .schemas import ChatRequest, ChatResponse, RunDetail, RunListResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/agent",
    tags=["agent-graph"],
    dependencies=[Depends(require_token)],
)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Answer a message through the LangGraph agent",
    responses={
        404: {"description": "prompt_id names a prompt this client has not registered"},
        422: {"description": "Empty message, unknown text_llm, or no resolvable prompt"},
        503: {"description": "No provider API key is configured"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """One LangChain agent decides whether to answer directly or search.

    The payload is the /agent/reply payload - `model` is Laravel's `text_llm`,
    and the prompt is `prompt_id` or `system_prompt` exactly as there. What
    differs is only how the reply is produced: the answering model is given a
    `rag_search` tool and decides for itself whether to use it. There is no
    intent-analysis call.

    The response carries `run_id`. Everything the graph did is readable at
    GET /api/v1/agent/runs/{run_id} and on the dashboard.
    """
    if not request.message.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="message must not be empty",
        )
    if not settings.api_key_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "No provider API key is configured. Set at least one of "
                "GROQ_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY or ANTHROPIC_API_KEY."
            ),
        )

    try:
        return await lc_agent.run_chat(request)
    except HTTPException:
        raise
    except PromptNotFound as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
    except (UnknownModel, PromptUnavailable) as error:
        # Both mean the caller under-specified the request. Answering on a house
        # default model or a house prompt is the one thing this service refuses.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(error)) from error
    except PermissionError as error:
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(error)) from error
    except Exception as error:  # noqa: BLE001
        logger.exception("graph chat failed")
        raise _translate(error) from error


# `/run` as an alias: the task names both, and they are the same operation.
router.add_api_route(
    "/run",
    chat,
    methods=["POST"],
    response_model=ChatResponse,
    summary="Alias of POST /api/v1/agent/chat",
)


@router.get(
    "/runs",
    response_model=RunListResponse,
    summary="Recent agent runs",
    responses={503: {"description": "Trace storage is not configured"}},
)
async def list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    conversation_id: str | None = None,
    user_id: str | None = None,
    run_status: str | None = Query(default=None, alias="status"),
    action: str | None = None,
) -> RunListResponse:
    try:
        runs, total = await store.list_runs(
            limit=limit, offset=offset, conversation_id=conversation_id,
            user_id=user_id, status=run_status, action=action,
        )
    except store.TraceStoreUnavailable as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    return RunListResponse(runs=runs, total=total, limit=limit, offset=offset)


@router.get(
    "/runs/{run_id}",
    response_model=RunDetail,
    summary="One run: every step, search, LLM call and tool call",
    responses={
        404: {"description": "No such run"},
        503: {"description": "Trace storage is not configured"},
    },
)
async def get_run(run_id: str) -> RunDetail:
    try:
        detail = await store.get_run(run_id)
    except store.TraceStoreUnavailable as error:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(error)) from error
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No run {run_id}")
    return RunDetail.model_validate(detail)


@router.get("/tools", summary="Tools this deployment offers the agent")
async def agent_tools() -> dict:
    """Diagnostic: what the model can actually call, and its exact descriptions.

    Replaces the old graph diagram. With the routing decision now made by the
    model from these descriptions, they ARE the routing logic - so being able to
    read the live ones matters more than a picture of a graph.
    """
    from ..schemas import ReplyRequest
    from . import lc_tools

    probe = ReplyRequest(message="", context={})
    return {"tools": lc_tools.definitions_for(lc_tools.build_tools(probe, None))}
