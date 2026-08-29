from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Service configuration. Everything business-related comes from Laravel at
    request time - only infrastructure and model defaults live here."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    # -- logging to disk ----------------------------------------------------
    # Blank disables it and only stdout is used. Docker already rotates stdout,
    # but at 10 MB x 5 a busy day rolls off within hours, and a post-mortem
    # needs the morning it rolled off. One file per container so gateway.log
    # and ai-agent.log can be read and archived independently.
    log_dir: str = ""
    log_file_name: str = "ai-agent.log"
    log_file_max_mb: int = 20
    log_file_keep: int = 10

    # Shared secret every caller presents as `Authorization: Bearer <token>`.
    agent_api_token: str = ""

    # -- persistence --------------------------------------------------------
    # Threads and prompt records, as JSON files. No database is involved; see
    # app/store/base.py for how to swap the backend.
    data_dir: str = "./storage"
    # Turns retained per thread. Working memory for the model - the calling
    # application keeps the durable message history in its own system.
    thread_max_turns: int = 40

    # -- generation limits --------------------------------------------------
    # Not model names: caps that apply whichever model the registry resolves to.
    max_tokens: int = 2000
    max_tool_iterations: int = 4

    # -- provider request bounds --------------------------------------------
    #
    # Every vendor SDK here (groq, openai, anthropic) defaults to a 600-second
    # timeout with 2 automatic retries. That is up to THIRTY MINUTES on one
    # /agent/reply, and it is not theoretical: a provider that accepts the
    # connection and then stalls produces exactly that, holding an asyncio task
    # and its whole prompt in memory the entire time.
    #
    # The gateway gives up on us at AI_TIMEOUT_MS (60s by default) and moves on,
    # so anything past that is work nobody is waiting for - it can only queue up
    # behind itself until the container runs out of memory. Bounded below that
    # ceiling so the failure is OURS to log and report, with one retry for a
    # genuine transient rather than the SDK's silent two.
    llm_timeout_s: float = 45.0
    llm_max_retries: int = 1

    # Log a per-request breakdown of what the prompt is made of (system vs
    # history vs tools) immediately before each provider call. Diagnostic only:
    # it changes nothing that is sent, and the numbers are estimates - see
    # services/token_diagnostics.py. Verbose, so off unless asked for.
    log_token_breakdown: bool = False

    # WHICH MODEL RUNS IS NOT CONFIGURED HERE.
    #
    # Laravel sends `text_llm`; app/services/models.py resolves it against
    # app/services/llm_registry.json into a provider and a real model id. There
    # is deliberately no LLM_PROVIDER, no *_MODEL and no MODEL_ALIASES: a model
    # name in the environment is a model nobody can see from the dashboard that
    # chose it.
    #
    # LLM_REGISTRY_PATH may point at a replacement registry file. It is a PATH,
    # not a model name, and is read directly by services/models.py.

    # -- API keys -----------------------------------------------------------
    # The ONLY provider configuration that belongs in .env. Each is looked up by
    # the `key_env` its provider declares in the registry.
    groq_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    # GEMINI_API_KEY is the name Google's own SDK and docs use; GOOGLE_API_KEY is
    # accepted as a synonym because it is what many deployments already export.
    # Read `gemini_key` rather than either of these directly.
    gemini_api_key: str = ""
    google_api_key: str = ""

    # -- per-provider generation tuning -------------------------------------
    groq_temperature: float = 0.3
    # Only accepted by some Groq models (e.g. gpt-oss). Left empty = not sent.
    groq_reasoning_effort: str = ""

    gemini_temperature: float = 0.3
    # Gemini 3.x models always reason before answering, and those thinking tokens
    # are billed against `max_output_tokens`. Left unbounded, a modest cap can be
    # consumed entirely by reasoning and the reply comes back truncated or empty.
    # A small budget keeps short replies predictable. Set to -1 to let the model
    # decide (Google's dynamic default). 0 is rejected by Gemini 3 - reasoning
    # cannot be switched off on those models.
    gemini_thinking_budget: int = 128

    # -- Anthropic generation tuning ----------------------------------------
    thinking_mode: Literal["adaptive", "disabled"] = "adaptive"
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"

    # -- Laravel back-channel (read-only, tools only) -----------------------
    # NO ROUTE IS ASSUMED: both endpoints default to empty. A tool is only
    # offered to the model when its endpoint is explicitly configured.
    laravel_base_url: str = ""
    laravel_api_token: str = ""
    laravel_timeout_s: float = 10.0
    knowledge_api: str = ""
    handoff_api: str = ""
    enable_knowledge_tool: bool = True
    enable_handoff_tool: bool = True

    # -- retrieval tool -----------------------------------------------------
    # ENABLE_RAG_TOOL=false stops the agent OFFERING knowledge-base search, on
    # both the /agent/reply tool loop and the LangChain agent - one switch, not
    # two. Every reply is then answered from the prompt, the conversation and
    # the caller's context alone.
    #
    # It does NOT touch the /api/rag/* routes: uploading, listing and POST
    # /api/rag/search keep working, so a corpus can be ingested and tested
    # before the agent is allowed to use it, and turned off without losing it.
    # To disable the whole RAG layer instead, leave RAG_DATABASE_URL unset.
    enable_rag_tool: bool = True

    # -- OpenClaw Gateway back-channel (block/unblock tools) ----------------
    # The only route from the agent to WhatsApp, and it goes through the
    # gateway's REST API. Blank = the block tools are not offered to the model.
    gateway_base_url: str = ""
    gateway_token: str = ""
    gateway_timeout_s: float = 10.0
    enable_block_tools: bool = True

    # -- OpenClaw integration (OpenAI-compatible provider) ------------------
    # OpenClaw calls this service as a model provider; these control how it
    # identifies itself and which prompt is applied to OpenClaw-routed turns.
    openai_compat_model_name: str = "whatsapp-agent"
    # OpenClaw cannot carry client_id/prompt_id through the OpenAI schema, so
    # they are configured. One of OPENCLAW_PROMPT / OPENCLAW_PROMPT_ID must be
    # set for /v1/chat/completions to work - there is no default prompt.
    openclaw_client_id: str = "openclaw"
    openclaw_prompt: str = ""
    openclaw_prompt_id: str = ""

    # -- Reply shaping ------------------------------------------------------
    default_max_words: int = 120
    fallback_reply: str = ""

    @property
    def gemini_key(self) -> str:
        """The resolved Gemini key: GEMINI_API_KEY, else GOOGLE_API_KEY."""
        return self.gemini_api_key or self.google_api_key

    def api_key_for(self, env_name: str) -> str:
        """The configured value of an API-key variable, by its .env name.

        Keyed on the name the registry declares (`providers.<name>.key_env`), so
        adding a provider to the registry needs no branch here.
        """
        name = str(env_name or "").strip().lower()
        if name in {"gemini_api_key", "google_api_key"}:
            return self.gemini_key
        return str(getattr(self, name, "") or "")

    @property
    def configured_keys(self) -> list[str]:
        """Which provider keys are set. For /health and boot logging."""
        present = []
        if self.groq_api_key:
            present.append("GROQ_API_KEY")
        if self.openai_api_key:
            present.append("OPENAI_API_KEY")
        if self.gemini_key:
            present.append("GEMINI_API_KEY")
        if self.anthropic_api_key:
            present.append("ANTHROPIC_API_KEY")
        return present

    @property
    def api_key_configured(self) -> bool:
        """Is ANY provider key set? Without one, nothing can be answered."""
        return bool(self.configured_keys)

    @property
    def rag_tool_available(self) -> bool:
        """Whether the agent may search documents.

        The DSN is still required: advertising a tool whose backend answers 503
        is worse than not offering it. `rag_settings` is read lazily and behind
        a guard because app/rag is optional - a deployment without it loses the
        tool, not the service.
        """
        if not self.enable_rag_tool:
            return False
        try:
            from .rag.config import rag_settings
        except Exception:  # noqa: BLE001 - RAG absent means the tool is absent
            return False
        return bool(rag_settings.database_configured)

    @property
    def knowledge_tool_available(self) -> bool:
        return self.enable_knowledge_tool and bool(self.knowledge_api)

    @property
    def handoff_tool_available(self) -> bool:
        return self.enable_handoff_tool and bool(self.handoff_api)

    @property
    def block_tools_available(self) -> bool:
        return self.enable_block_tools and bool(self.gateway_base_url)

    @property
    def tools_available(self) -> bool:
        return (
            self.knowledge_tool_available
            or self.handoff_tool_available
            or self.block_tools_available
            or self.rag_tool_available
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
