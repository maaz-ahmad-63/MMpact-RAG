from typing import Any, Literal

from pydantic import BaseModel, Field


class HistoryTurn(BaseModel):
    """One prior turn of the conversation.

    Supplied inline by the caller (`context.history`) or read back from a stored
    thread. Both paths produce the same shape.
    """

    role: Literal["user", "assistant"]
    content: str
    at: str | None = None


class Contact(BaseModel):
    phone: str | None = None
    push_name: str | None = None
    name: str | None = None


# -- thread API -------------------------------------------------------------


class ThreadRef(BaseModel):
    """Identity of a conversation.

    Threads are keyed by (client_id, channel, contact_id). Two client apps
    talking to the same phone number get two separate threads - neither can see
    the other's history.
    """

    client_id: str = Field(description="Which application owns this conversation.")
    channel: str = Field(default="whatsapp", description="whatsapp, web, sms, …")
    contact_id: str = Field(description="The other party, in that channel's terms.")


class Thread(BaseModel):
    thread_id: str
    client_id: str
    channel: str
    contact_id: str
    created_at: str
    updated_at: str
    turns: list[HistoryTurn] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


# -- prompt API -------------------------------------------------------------


class PromptRecord(BaseModel):
    """A prompt/persona a client app registered.

    Nothing about it is interpreted by this service beyond assembling it into a
    system prompt. There is no prompt library here to fall back on: if a client
    has not registered one and does not send `system_prompt` inline, the request
    is refused.
    """

    client_id: str
    prompt_id: str
    system_prompt: str = Field(description="The instruction text, verbatim from the client.")
    tone: str | None = None
    language: str | None = None
    tools_allowed: list[str] = Field(
        default_factory=list,
        description="Tool names this prompt may use. Empty means no tools.",
    )
    rag_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Reserved. Stored and echoed back, but no retrieval layer reads it yet - "
            "the rag_search tool reports that RAG is not enabled."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class PromptUpsertRequest(PromptRecord):
    pass


# -- reply API --------------------------------------------------------------


class ReplyRequest(BaseModel):
    """Payload sent by any client application, directly or via a gateway.

    Nothing here is specific to one caller. `client_id` says which application is
    asking; the prompt is either a `prompt_id` it registered earlier or
    `system_prompt` sent inline. Everything in `context` is that client's own
    data and is treated as authoritative - this service applies no business
    rules and holds no configuration of its own.

    The older `{ prompt, message, context }` shape still validates: `prompt` is
    accepted as inline instruction text when no `prompt_id`/`system_prompt` is
    given.
    """

    client_id: str | None = Field(
        default=None,
        description="Which application is calling. Required to use threads or prompt records.",
    )
    thread_id: str | None = Field(
        default=None, description="Existing thread. Omit and supply channel+contact_id to resolve one."
    )
    channel: str | None = Field(default=None, description="Used with contact_id to resolve a thread.")
    contact_id: str | None = Field(default=None, description="Used with channel to resolve a thread.")

    prompt_id: str | None = Field(
        default=None, description="A prompt this client registered via POST /agent/prompt."
    )
    system_prompt: str | None = Field(
        default=None, description="Inline instruction text, for clients that skip registration."
    )
    prompt: str | None = Field(
        default=None,
        description="Legacy alias for system_prompt. Used only when neither of the above is set.",
    )

    tools_allowed: list[str] | None = Field(
        default=None,
        description="Overrides the prompt record's tools_allowed for this one call.",
    )

    model: str | None = Field(
        default=None,
        description=(
            "Model the caller wants, in its own vocabulary. Resolved through "
            "the LLM registry; an unattributable name is refused rather than silently "
            "answered on the service default."
        ),
    )
    max_tokens: int | None = Field(
        default=None, ge=1, description="Caps this reply's length. Falls back to MAX_TOKENS."
    )

    message: str = Field(description="The message to answer.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Free-form context supplied by the calling application.",
    )

    persist: bool = Field(
        default=True,
        description="Write this exchange back to the thread. False for one-off/test calls.",
    )

    # Convenience accessors for the well-known context keys.
    @property
    def whatsapp_id(self) -> str | None:
        value = self.context.get("whatsapp_id")
        return str(value) if value is not None else None

    @property
    def purpose(self) -> str | None:
        return self.context.get("purpose")

    @property
    def account_name(self) -> str | None:
        return self.context.get("account_name")

    @property
    def ai_settings(self) -> dict[str, Any]:
        value = self.context.get("ai_settings") or {}
        return value if isinstance(value, dict) else {}

    @property
    def history(self) -> list[HistoryTurn]:
        raw = self.context.get("history") or []
        turns: list[HistoryTurn] = []
        for item in raw:
            try:
                turns.append(HistoryTurn.model_validate(item))
            except Exception:  # noqa: BLE001 - skip malformed turns, never fail the reply
                continue
        return turns

    @property
    def contact(self) -> Contact:
        raw = self.context.get("contact") or {}
        try:
            return Contact.model_validate(raw)
        except Exception:  # noqa: BLE001
            return Contact()


class ToolCallRecord(BaseModel):
    name: str
    input: dict[str, Any]
    ok: bool
    error: str | None = None


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class LLMTrace(BaseModel):
    """Exactly what was sent to the model, for the caller's audit log.

    This service holds no database. The gateway owns the Postgres connection and
    the organisation/agent/conversation ids, so it writes dm_ai_logs - but only
    this service knows the assembled prompt, the provider timing and the token
    usage. Returning them here is what lets one complete row be written without
    either side guessing.

    Everything is observed, never reconstructed: `system_prompt` is the string
    handed to the provider, and `messages` is the array as sent.
    """

    system_prompt: str | None = None
    #: The final user turn - the message actually being answered.
    user_prompt: str | None = None
    #: The full structured conversation as sent, system turn included.
    messages: list[dict[str, Any]] = Field(default_factory=list)
    #: The same array rendered as role-labelled text, for dm_ai_logs.
    #: all_prompt_text_type. Same source object as `messages` above - it is a
    #: rendering of what was sent, never a second assembly of the prompt.
    all_prompt_text: str | None = None
    #: Time inside provider calls only, excluding prompt assembly, thread I/O
    #: and tool execution. This is the LLM request duration, not the pipeline's.
    llm_ms: int | None = None
    #: How many provider round-trips this reply took. More than one means the
    #: model called tools; usage below is the sum across all of them.
    provider_calls: int = 0
    #: 'provider' when the API reported usage, 'unavailable' when it did not.
    usage_source: str = "unavailable"
    #: What the caller asked for, and what the registry resolved it to.
    model_requested: str | None = None
    model_resolved: str | None = None


class ReplyResponse(BaseModel):
    """Response returned to the caller.

    `reply` is the only field a caller must read; the rest is telemetry the
    calling application may choose to persist.
    """

    reply: str
    thread_id: str | None = None
    tool_calls_made: list[str] = Field(
        default_factory=list, description="Names of the tools that ran, in order."
    )
    provider: str | None = None
    model: str | None = None
    stop_reason: str | None = None
    handoff_requested: bool = False
    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list, description="The same calls with arguments and outcomes."
    )
    usage: Usage | None = None
    latency_ms: int | None = None
    #: Prompt/timing detail for the caller's AI log. See LLMTrace.
    trace: LLMTrace | None = None


class ClassifyRequest(BaseModel):
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    labels: list[str] | None = Field(
        default=None,
        description="Intent labels Laravel wants the message sorted into. Optional.",
    )
    model: str | None = Field(
        default=None,
        description=(
            "Laravel's `text_llm`. Resolved through the LLM registry exactly as on "
            "the reply path. Omitted, the registry's declared default is used."
        ),
    )


class Classification(BaseModel):
    intent: str
    sentiment: Literal["positive", "neutral", "negative"]
    urgency: Literal["low", "medium", "high"]
    language: str
    requires_human: bool
    summary: str


class ClassifyResponse(BaseModel):
    classification: Classification
    provider: str | None = None
    model: str | None = None
    usage: Usage | None = None
    latency_ms: int | None = None
