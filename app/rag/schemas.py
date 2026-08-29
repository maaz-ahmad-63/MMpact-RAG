"""Request and response models for the RAG API.

Separate from app/schemas.py on purpose: that module is the agent's contract
with its clients, and nothing in this task should change what a caller sees
there. Every model below carries descriptions and examples, so /docs documents
these routes without a word of hand-written prose.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChunkMetadata(BaseModel):
    """What travels with every chunk. `page` is null for DOCX and TXT, which
    have no page until they are rendered - an invented page number would be a
    citation the reader cannot check."""

    document_id: str
    chunk_index: int
    source: str = Field(description="The original filename.")
    page: int | None = Field(default=None, description="1-based PDF page, else null.")


class DocumentSummary(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int = 0
    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None
    updated_at: str | None = None


class UploadResponse(BaseModel):
    """The result of ingesting one file."""

    document_id: str
    filename: str
    file_type: str
    chunks: int = Field(description="Chunks created, embedded and stored.")
    characters: int = Field(description="Characters of text after cleaning.")
    pages: int | None = Field(default=None, description="Pages read, PDFs only.")
    embedding_model: str
    embedding_dim: int
    took_ms: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="The natural-language question.")
    top_k: int | None = Field(
        default=None, ge=1, description="How many chunks to return. Defaults to RAG_TOP_K."
    )
    threshold: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description=(
            "Minimum similarity a chunk must reach to be returned. Defaults to "
            "RAG_SIMILARITY_THRESHOLD. 0 disables the filter."
        ),
    )
    # Optional scoping. Absent = search everything, which is the right default
    # for a service whose callers do not all partition their documents.
    user_id: str | None = None
    agent_id: str | None = None
    conversation_id: str | None = None
    document_ids: list[str] | None = Field(
        default=None, description="Restrict the search to these documents."
    )

    @field_validator("query")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be empty or whitespace")
        return value

    model_config = {
        "json_schema_extra": {
            "example": {"query": "What is the refund policy?", "top_k": 5}
        }
    }


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    content: str
    similarity: float = Field(description="Higher is closer. Cosine: [0, 1].")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="source, page and chunk_index, plus whatever the uploader attached.",
    )


class SearchResponse(BaseModel):
    query: str
    found: bool = Field(
        description="False when nothing reached the similarity threshold. The "
                    "caller - not this service - decides what to do about that."
    )
    results: list[SearchResult] = Field(default_factory=list)
    threshold: float
    top_k: int
    took_ms: int


class ContextRequest(SearchRequest):
    max_chars: int | None = Field(
        default=None, ge=1,
        description="Cap on the assembled context. Defaults to RAG_CONTEXT_MAX_CHARS.",
    )


class ContextSource(BaseModel):
    document_id: str
    source: str
    page: int | None = None
    chunk_index: int | None = None
    similarity: float


class ContextResponse(BaseModel):
    """Retrieved text, assembled and ready to paste into a prompt.

    This endpoint calls NO model and touches no agent code: it retrieves,
    formats and returns. The existing agent decides whether to use it.
    """

    query: str
    found: bool
    context: str = Field(description="Empty string when nothing passed the threshold.")
    sources: list[ContextSource] = Field(default_factory=list)
    chunks_used: int = 0
    took_ms: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentSummary]
    total: int
    limit: int
    offset: int


class DeleteResponse(BaseModel):
    ok: bool = True
    document_id: str
    chunks_deleted: int


class RagHealthResponse(BaseModel):
    ok: bool
    configuration: dict[str, Any]
    database: dict[str, Any]
    embeddings: dict[str, Any]
