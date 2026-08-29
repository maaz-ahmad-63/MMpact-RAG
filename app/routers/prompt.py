"""Agent Prompt API - prompts and personas registered by client applications.

A prompt record is stored exactly as the client sent it. This service ships no
prompts of its own and interprets none of these fields beyond assembling them
into a system prompt at reply time.

Prompt ids are scoped per client, so two applications may both register
"support" without colliding. That is also why the read routes require
`client_id`: a bare prompt_id does not identify a record.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ..schemas import PromptRecord, PromptUpsertRequest
from ..security import require_token
from ..services import tools as tool_registry
from ..store import prompts as prompt_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent/prompt", tags=["prompt"], dependencies=[Depends(require_token)])


@router.post("", response_model=PromptRecord)
async def upsert(request: PromptUpsertRequest) -> PromptRecord:
    """Register or update a prompt.

    Request:
      {
        "client_id": "laravel-prod",
        "prompt_id": "support",
        "system_prompt": "You are the support agent for Acme…",
        "tone": "warm",
        "language": "English",
        "tools_allowed": ["block_contact", "request_human_handoff"],
        "rag_config": null
      }

    `tools_allowed` names tools from the dispatcher's registry. Unknown names are
    rejected here rather than silently ignored at reply time. An empty list means
    this prompt runs with no tools at all.

    `rag_config` is accepted and stored, but no retrieval layer reads it yet.
    """
    unknown = [name for name in request.tools_allowed if name not in tool_registry.REGISTRY]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Unknown tools in tools_allowed: {', '.join(unknown)}. "
                f"Registered tools: {', '.join(sorted(tool_registry.REGISTRY))}"
            ),
        )

    if not request.system_prompt.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="system_prompt must not be empty - this service has no default to fall back on",
        )

    record = await prompt_store.upsert(PromptRecord.model_validate(request.model_dump()))
    logger.info(
        "prompt %s/%s registered (tools=%s)",
        record.client_id,
        record.prompt_id,
        record.tools_allowed or "none",
    )
    return record


@router.get("/{prompt_id}", response_model=PromptRecord)
async def fetch(
    prompt_id: str,
    client_id: str = Query(description="The client that registered this prompt."),
) -> PromptRecord:
    """Read a prompt back."""
    record = await prompt_store.get(client_id, prompt_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} has no prompt '{prompt_id}'",
        )
    return record


@router.delete("/{prompt_id}")
async def delete(
    prompt_id: str,
    client_id: str = Query(description="The client that registered this prompt."),
) -> dict[str, object]:
    record = await prompt_store.get(client_id, prompt_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client {client_id} has no prompt '{prompt_id}'",
        )
    deleted = await prompt_store.delete(client_id, prompt_id)
    logger.info("prompt %s/%s deleted", client_id, prompt_id)
    return {"ok": True, "client_id": client_id, "prompt_id": prompt_id, "deleted": deleted}
