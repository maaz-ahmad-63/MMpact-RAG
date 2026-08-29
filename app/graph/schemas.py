"""Request/response models for the graph and dashboard APIs.

`ChatRequest` deliberately SUBCLASSES the existing ReplyRequest rather than
declaring a new payload. Everything a caller already sends to /agent/reply -
client_id, prompt_id/system_prompt, `model` (Laravel's text_llm), context,
tools_allowed, persist - means exactly the same thing here. A second request
vocabulary would be a second configuration system, which is the thing this task
forbids.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..schemas import LLMTrace, ReplyRequest, Usage

# -- actions ----------------------------------------------------------------

#: What a run turned out to be. DERIVED from what the model actually did - it is
#: not a classification and no model is asked to produce it. Kept because the
#: dashboard, the agent_runs.action column and any Laravel reporting already
#: read these values.
Action = Literal["direct_answer", "rag_search", "tool", "human_handoff"]


# -- chat -------------------------------------------------------------------


class ChatRequest(ReplyRequest):
    """A /agent/reply payload, answered through the graph instead of the loop."""

    conversation_id: str | None = Field(
        default=None,
        description=(
            "Groups runs in the dashboard. Defaults to the resolved thread_id, so "
            "callers that already use threads need send nothing."
        ),
    )
    user_id: str | None = Field(
        default=None, description="Whose conversation this is, for dashboard filtering."
    )
    rag_top_k: int | None = Field(
        default=None, ge=1, description="Overrides RAG_TOP_K for this run only."
    )
    rag_threshold: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Overrides RAG_SIMILARITY_THRESHOLD."
    )

    # Document scoping. OPT-IN and separate from `user_id`/`conversation_id`
    # above, which group runs in the dashboard and say nothing about which
    # documents to search. Send these only if you set the matching columns when
    # UPLOADING the documents - a filter that no document matches returns
    # nothing at all, which is indistinguishable from a bad query unless you
    # know it was applied.
    rag_user_id: str | None = Field(
        default=None, description="Restrict retrieval to documents uploaded with this user_id."
    )
    rag_agent_id: str | None = Field(
        default=None, description="Restrict retrieval to documents uploaded with this agent_id."
    )
    rag_conversation_id: str | None = Field(
        default=None,
        description="Restrict retrieval to documents uploaded with this conversation_id.",
    )
    rag_document_ids: list[str] | None = Field(
        default=None, description="Restrict retrieval to these document ids."
    )


class RetrievedChunk(BaseModel):
    content: str
    score: float
    document_id: str
    chunk_id: str
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    """The graph's answer, plus the trace handle for the dashboard."""

    run_id: str
    reply: str
    conversation_id: str | None = None
    thread_id: str | None = None

    #: Always null now: nothing classifies intent. The field stays so a Laravel
    #: client that reads it keeps parsing this response.
    intent: str | None = None
    #: Derived from what the model did - see Action.
    action: str | None = None
    confidence: float | None = None
    requires_rag: bool = False
    search_query: str | None = None

    provider: str | None = None
    model: str | None = None

    sources: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    handoff_requested: bool = False

    #: Tokens for the WHOLE run - every provider call the graph made, summed.
    #: This is what dm_ai_logs.prompt_tokens/completion_tokens/total_tokens must
    #: be written from: a graph run costs 2-3 calls, and reporting only the
    #: answer call would under-bill every retrieval question.
    usage: Usage | None = None
    #: The same tokens split per node, so an expensive run is attributable.
    #: Belongs in dm_ai_logs.metadata, not in a column.
    usage_by_node: list[dict[str, Any]] = Field(default_factory=list)
    #: Same shape /agent/reply returns, so an existing dm_ai_logs writer needs
    #: no change: system_prompt, user_prompt, messages, all_prompt_text, llm_ms.
    trace: LLMTrace | None = None

    latency_ms: int | None = None
    status: str = "completed"
    error: str | None = None


# -- observability read models ---------------------------------------------


class RunSummary(BaseModel):
    run_id: str
    conversation_id: str | None = None
    user_id: str | None = None
    client_id: str | None = None
    user_message: str | None = None
    intent: str | None = None
    action: str | None = None
    confidence: float | None = None
    status: str
    model_provider: str | None = None
    model_name: str | None = None
    used_rag: bool = False
    latency_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None


class StepRecord(BaseModel):
    step_number: int
    node_name: str
    step_type: str
    status: str
    latency_ms: int | None = None
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RagResultRecord(BaseModel):
    document_id: str | None = None
    chunk_id: str | None = None
    score: float | None = None
    source: str | None = None
    content: str | None = None


class RagSearchRecord(BaseModel):
    query: str
    top_k: int | None = None
    threshold: float | None = None
    results_count: int = 0
    latency_ms: int | None = None
    rewritten: bool = False
    results: list[RagResultRecord] = Field(default_factory=list)


class LlmCallRecord(BaseModel):
    node_name: str
    provider: str | None = None
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int | None = None
    status: str = "ok"
    error: str | None = None


class ToolCallRecordRow(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None
    status: str = "ok"
    latency_ms: int | None = None
    error: str | None = None


class RunDetail(BaseModel):
    run: RunSummary
    final_answer: str | None = None
    steps: list[StepRecord] = Field(default_factory=list)
    rag_searches: list[RagSearchRecord] = Field(default_factory=list)
    llm_calls: list[LlmCallRecord] = Field(default_factory=list)
    tool_calls: list[ToolCallRecordRow] = Field(default_factory=list)


class RunListResponse(BaseModel):
    runs: list[RunSummary] = Field(default_factory=list)
    total: int = 0
    limit: int = 50
    offset: int = 0


class OverviewResponse(BaseModel):
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    rag_requests: int = 0
    direct_answers: int = 0
    tool_calls: int = 0
    human_handoffs: int = 0
    average_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    window_hours: int = 24


class DashboardUnavailable(BaseModel):
    ok: bool = False
    detail: str
