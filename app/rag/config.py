"""Configuration for the RAG layer only.

A separate Settings class rather than new fields on app/config.py: that file is
read by the agent, the providers and the stores, and this task must not change
what any of them see. Both classes read the same .env - pydantic-settings is
happy to have two - so operators still configure one file.

Nothing here has a secret as a default, and no DSN, key or model path is written
into source. `RAG_DATABASE_URL` falls back to `DATABASE_URL` because most
deployments already export the latter for the rest of the stack.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Distance metric -> (pgvector operator, opclass for the index).
# Cosine is the default because BGE-M3 is trained for cosine/inner-product
# retrieval and its vectors are normalised, which makes `1 - distance` a
# similarity in [0, 1] that a threshold can be reasoned about.
METRICS: dict[str, tuple[str, str]] = {
    "cosine": ("<=>", "vector_cosine_ops"),
    "l2": ("<->", "vector_l2_ops"),
    "inner_product": ("<#>", "vector_ip_ops"),
}


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # -- database -----------------------------------------------------------
    # Blank = the RAG routes report 503 and the rest of the service is unaffected.
    rag_database_url: str = ""
    database_url: str = ""
    rag_db_pool_min: int = 1
    rag_db_pool_max: int = 5
    rag_db_timeout_s: float = 30.0
    # Schema the two tables live in. Separate from anything the gateway owns.
    rag_db_schema: str = "public"

    # -- provider selection -------------------------------------------------
    # local | cloudflare
    embedding_provider: Literal["local", "cloudflare"] = "local"
    reranker_provider: Literal["local", "cloudflare"] = "local"

    # -- cloudflare workers ai ----------------------------------------------
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    cloudflare_embedding_model: str = "@cf/baai/bge-m3"
    cloudflare_reranker_model: str = "@cf/baai/bge-reranker-base"
    cloudflare_timeout_s: float = 30.0

    # -- embeddings ---------------------------------------------------------
    embedding_model: str = "BAAI/bge-m3"
    # auto  - HTTP when EMBEDDING_API_BASE is set, otherwise local
    # http  - an OpenAI-compatible /embeddings endpoint (TEI, Infinity, vLLM,
    #         llama.cpp, Ollama's OpenAI surface): the self-hosted case
    # local - sentence-transformers in this process (needs the extra install)
    embedding_backend: Literal["auto", "http", "local"] = "auto"
    # Either of these turns the HTTP backend on:
    #   EMBEDDING_API_URL  - the full endpoint, for a server with its own path
    #                        and its own request shape (e.g. .../embed)
    #   EMBEDDING_API_BASE - an OpenAI-compatible base; "/embeddings" is
    #                        appended
    # URL wins when both are set.
    embedding_api_base: str = ""
    embedding_api_url: str = ""
    embedding_api_key: str = ""
    # How the request body is written. Response shapes are auto-detected and
    # need no setting; only the request has to be declared.
    #   openai - {"model": "...", "input": ["a", "b"]}
    #   texts  - {"texts": ["a", "b"]}
    embedding_api_format: Literal["openai", "texts"] = "openai"
    # Which header carries the key. "Authorization" is sent as "Bearer <key>";
    # any other name (X-API-Key, x-token, …) is sent as the bare value, because
    # that is what servers using a custom header expect.
    embedding_api_key_header: str = "Authorization"
    embedding_timeout_s: float = 120.0
    # How many chunks go to the model in one call. Ingestion of a large PDF is
    # thousands of chunks; one call per chunk would be pathological.
    embedding_batch_size: int = 32
    # Device for the local backend: cpu, cuda, mps. Blank = let the library pick.
    embedding_device: str = ""
    # The stored vector width. It MUST equal what the model actually returns -
    # this is not an assumption, it is checked against a live embedding at
    # startup and on every ingest, and a mismatch is refused loudly rather than
    # written into the column. 1024 is BGE-M3's dense width.
    rag_embedding_dim: int = 1024

    # -- chunking -----------------------------------------------------------
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 150
    # Fragments below this are folded into the neighbouring chunk instead of
    # being stored: a 20-character chunk retrieves noise and costs a row.
    rag_min_chunk_size: int = 100

    # -- retrieval ----------------------------------------------------------
    rag_top_k: int = 5
    rag_max_top_k: int = 50
    rag_similarity_threshold: float = 0.70
    rag_distance_metric: Literal["cosine", "l2", "inner_product"] = "cosine"
    # Characters. Caps what /api/rag/context hands to a prompt, so retrieval
    # can never blow the model's context window on its own.
    rag_context_max_chars: int = 6000

    # -- hybrid search settings ---------------------------------------------
    rag_vector_candidates: int = 20
    rag_keyword_candidates: int = 20
    rag_rrf_k: int = 60

    # -- reranker settings --------------------------------------------------
    rag_rerank_enabled: bool = True
    rag_rerank_candidates: int = 20
    rag_rerank_top_k: int = 5
    rag_rerank_model: str = "BAAI/bge-reranker-base"

    # -- llm answer pipeline (/api/rag/ask) ---------------------------------
    # auto (picks first available with key) | groq | gemini | openai | anthropic | ollama | mock
    rag_llm_provider: str = "auto"
    rag_llm_model: str = ""
    rag_llm_temperature: float = 0.2
    rag_llm_max_tokens: int = 1024
    rag_max_history_turns: int = 10
    rag_default_system_prompt: str = ""

    # LLM API keys (read from .env)
    groq_api_key: str = ""
    gemini_api_key: str = ""
    google_api_key: str = ""
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # -- uploads ------------------------------------------------------------
    rag_max_upload_mb: int = 25
    # Where the original file is kept, so a document can be re-chunked later
    # without asking the client to upload it again. Blank = do not keep it.
    rag_storage_dir: str = "./storage/rag_documents"

    @field_validator("embedding_provider", "reranker_provider", mode="before")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        if isinstance(value, str):
            val = value.strip().lower()
            if val in ("local", "cloudflare"):
                return val
            raise ValueError(f"Invalid provider '{value}'. Must be 'local' or 'cloudflare'.")
        return value

    @field_validator("rag_chunk_overlap")
    @classmethod
    def _overlap_fits(cls, value: int, info) -> int:
        size = info.data.get("rag_chunk_size", 800)
        if value < 0:
            raise ValueError("RAG_CHUNK_OVERLAP must not be negative")
        if size and value >= size:
            raise ValueError(
                f"RAG_CHUNK_OVERLAP ({value}) must be smaller than RAG_CHUNK_SIZE "
                f"({size}); an overlap at or above the size never advances."
            )
        return value

    def validate_cloudflare_config(self, for_feature: str = "inference") -> None:
        """Validate that Cloudflare credentials are configured when needed."""
        missing = []
        if not self.cloudflare_account_id.strip():
            missing.append("CLOUDFLARE_ACCOUNT_ID")
        if not self.cloudflare_api_token.strip():
            missing.append("CLOUDFLARE_API_TOKEN")
        if missing:
            raise ValueError(
                f"Cloudflare {for_feature} requested, but required environment "
                f"variable(s) missing: {', '.join(missing)}. "
                f"Please set them in your environment or .env file."
            )

    @property
    def dsn(self) -> str:
        """The Postgres DSN, preferring the RAG-specific one."""
        return (self.rag_database_url or self.database_url).strip()

    @property
    def database_configured(self) -> bool:
        return bool(self.dsn)

    @property
    def embedding_endpoint(self) -> str:
        """The exact URL that is POSTed to, or "" when none is configured."""
        if self.embedding_api_url.strip():
            return self.embedding_api_url.strip()
        base = self.embedding_api_base.strip().rstrip("/")
        return f"{base}/embeddings" if base else ""

    @property
    def resolved_backend(self) -> Literal["http", "local", "cloudflare"]:
        if self.embedding_provider == "cloudflare":
            return "cloudflare"
        if self.embedding_backend != "auto":
            return self.embedding_backend
        return "http" if self.embedding_endpoint else "local"

    @property
    def distance_operator(self) -> str:
        return METRICS[self.rag_distance_metric][0]

    @property
    def index_opclass(self) -> str:
        return METRICS[self.rag_distance_metric][1]

    def public_state(self) -> dict[str, object]:
        """What /api/rag/health may report. Names and numbers, never secrets."""
        active_embedding_model = (
            self.cloudflare_embedding_model
            if self.embedding_provider == "cloudflare"
            else self.embedding_model
        )
        active_reranker_model = (
            self.cloudflare_reranker_model
            if self.reranker_provider == "cloudflare"
            else self.rag_rerank_model
        )
        return {
            "database_configured": self.database_configured,
            "embedding_provider": self.embedding_provider,
            "reranker_provider": self.reranker_provider,
            "cloudflare_configured": bool(
                self.cloudflare_account_id.strip() and self.cloudflare_api_token.strip()
            ),
            "cloudflare_embedding_model": self.cloudflare_embedding_model,
            "cloudflare_reranker_model": self.cloudflare_reranker_model,
            "embedding_model": active_embedding_model,
            "embedding_backend": self.resolved_backend,
            "embedding_endpoint": self.embedding_endpoint or None,
            "embedding_api_format": self.embedding_api_format,
            # The header's NAME, never its value.
            "embedding_api_key_header": self.embedding_api_key_header,
            "embedding_api_key_set": bool(self.embedding_api_key),
            "embedding_dim": self.rag_embedding_dim,
            "chunk_size": self.rag_chunk_size,
            "chunk_overlap": self.rag_chunk_overlap,
            "top_k": self.rag_top_k,
            "similarity_threshold": self.rag_similarity_threshold,
            "distance_metric": self.rag_distance_metric,
            "vector_candidates": self.rag_vector_candidates,
            "keyword_candidates": self.rag_keyword_candidates,
            "rrf_k": self.rag_rrf_k,
            "rerank_enabled": self.rag_rerank_enabled,
            "rerank_candidates": self.rag_rerank_candidates,
            "rerank_top_k": self.rag_rerank_top_k,
            "rerank_model": active_reranker_model,
            "llm_provider": self.rag_llm_provider,
            "llm_model": self.rag_llm_model,
            "llm_temperature": self.rag_llm_temperature,
            "llm_max_tokens": self.rag_llm_max_tokens,
            "groq_configured": bool(self.groq_api_key),
            "gemini_configured": bool(self.gemini_api_key or self.google_api_key),
            "openai_configured": bool(self.openai_api_key),
            "anthropic_configured": bool(self.anthropic_api_key),
        }


@lru_cache
def get_rag_settings() -> RagSettings:
    return RagSettings()


rag_settings = get_rag_settings()
