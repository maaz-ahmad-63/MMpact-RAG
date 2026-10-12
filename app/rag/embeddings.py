"""BGE-M3 embeddings, behind one interface and one loaded instance.

Two backends, chosen by configuration and never by the caller:

  http  - POST to a self-hosted embedding server. The model runs where the GPU
          is, this process stays small, and `httpx` is already a dependency.
          Two request dialects, chosen by EMBEDDING_API_FORMAT:
            openai - {"model": ..., "input": [...]} to {base}/embeddings, which
                     text-embeddings-inference, Infinity, vLLM, llama.cpp and
                     Ollama all speak
            texts  - {"texts": [...]} to EMBEDDING_API_URL, for a server with
                     its own shape
          The key goes in EMBEDDING_API_KEY_HEADER (Authorization as a Bearer
          token; any other name as the bare value).
  local - sentence-transformers in this process. Convenient for a single-box
          deployment; costs ~2 GB of RAM and a slow first load.

Loading, once
-------------
`get_embedder()` returns a process-wide singleton behind an asyncio lock, so a
burst of concurrent requests during startup loads the model once rather than
once per request. `warmup()` is what makes that happen at boot instead of inside
somebody's first upload.

Dimension is measured, not assumed
----------------------------------
`dimension` is read from a real embedding of a probe string the first time the
embedder initialises, and `verify_dimension()` compares it with the width of the
pgvector column. A mismatch is refused with both numbers named, because the
alternative - Postgres rejecting every insert with `expected 1024 dimensions` -
is a far worse way to find out.

Blocking work never runs on the event loop: sentence-transformers is CPU-bound
and goes through asyncio.to_thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Sequence

from .cloudflare_ai import (
    CloudflareAIError,
    CloudflareAuthError,
    CloudflareModelError,
    CloudflareQuotaExhaustedError,
    EmbeddingError,
    embed_texts_cloudflare,
)
from .config import RagSettings, rag_settings

logger = logging.getLogger(__name__)

_PROBE = "dimension probe"


class Embedder:
    """One loaded embedding model (or one client to a hosted one)."""

    def __init__(self, settings: RagSettings | None = None) -> None:
        self._settings = settings or rag_settings
        self._model = None  # sentence-transformers instance, local backend only
        self._client = None  # httpx.AsyncClient, http / cloudflare backend
        self._dimension: int | None = None
        self._lock = asyncio.Lock()

    # -- properties ---------------------------------------------------------

    @property
    def backend(self) -> str:
        return self._settings.resolved_backend

    @property
    def model_name(self) -> str:
        if self.backend == "cloudflare":
            return self._settings.cloudflare_embedding_model
        return self._settings.embedding_model

    @property
    def dimension(self) -> int | None:
        """The width the model actually returns; None until first use."""
        return self._dimension

    @property
    def ready(self) -> bool:
        return self._dimension is not None

    # -- lifecycle ----------------------------------------------------------

    async def load(self) -> int:
        """Initialise the backend and measure the vector width. Idempotent."""
        if self._dimension is not None:
            return self._dimension

        async with self._lock:
            if self._dimension is not None:  # won the race while waiting
                return self._dimension

            started = time.perf_counter()
            if self.backend == "local":
                await self._load_local()
            elif self.backend == "cloudflare":
                self._open_cloudflare_client()
            else:
                self._open_http_client()

            vectors = await self._embed_raw([_PROBE])
            if not vectors or not vectors[0]:
                raise EmbeddingError(
                    f"The embedding backend returned nothing for a probe string "
                    f"(model={self.model_name}, backend={self.backend})"
                )
            self._dimension = len(vectors[0])
            logger.info(
                "rag: embedding model ready (model=%s backend=%s dim=%s in %.1fs)",
                self.model_name, self.backend, self._dimension,
                time.perf_counter() - started,
            )
            return self._dimension

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._model = None
        self._dimension = None

    async def _load_local(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise EmbeddingError(
                "The local embedding backend needs `sentence-transformers` "
                "(pip install sentence-transformers). Alternatively set "
                "EMBEDDING_API_BASE to a self-hosted embedding server and the "
                "HTTP backend is used instead - no model is loaded in-process."
            ) from error

        device = self._settings.embedding_device or None

        def _build():
            return SentenceTransformer(self._settings.embedding_model, device=device)

        logger.info(
            "rag: loading %s locally (device=%s) - first run downloads the weights",
            self._settings.embedding_model, device or "auto",
        )
        self._model = await asyncio.to_thread(_build)

    def _open_http_client(self) -> None:
        import httpx

        endpoint = self._settings.embedding_endpoint
        if not endpoint:
            raise EmbeddingError(
                "EMBEDDING_BACKEND=http but neither EMBEDDING_API_URL nor "
                "EMBEDDING_API_BASE is set"
            )

        headers = {"accept": "application/json"}
        key = self._settings.embedding_api_key
        if key:
            name = self._settings.embedding_api_key_header.strip() or "Authorization"
            # Authorization has a scheme; a custom header (X-API-Key and its
            # relatives) carries the bare key, and prefixing it with "Bearer "
            # would simply make every request a 401.
            headers[name] = f"Bearer {key}" if name.lower() == "authorization" else key

        # The full URL is passed per request rather than as a base_url, so a
        # server whose path is not /embeddings needs no special case here.
        self._client = httpx.AsyncClient(
            timeout=self._settings.embedding_timeout_s, headers=headers
        )

    def _open_cloudflare_client(self) -> None:
        import httpx

        self._settings.validate_cloudflare_config("embeddings")
        headers = {
            "Authorization": f"Bearer {self._settings.cloudflare_api_token.strip()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MMpact-RAG-Inference-Client/1.0",
        }
        self._client = httpx.AsyncClient(
            timeout=self._settings.cloudflare_timeout_s, headers=headers
        )

    # -- embedding ----------------------------------------------------------

    async def embed_text(self, text: str) -> list[float]:
        """One string to one vector."""
        vectors = await self.embed_texts([text])
        return vectors[0]

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Many strings to many vectors, in batches of EMBEDDING_BATCH_SIZE.

        Batching is the whole point: a 200-page PDF is thousands of chunks, and
        one request per chunk turns a ten-second ingest into ten minutes of
        round-trips.
        """
        items = list(texts)
        if not items:
            return []
        if any(not str(text).strip() for text in items):
            raise EmbeddingError("Cannot embed an empty string")

        await self.load()

        batch_size = max(1, self._settings.embedding_batch_size)
        vectors: list[list[float]] = []
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            vectors.extend(await self._embed_raw(batch))

        if len(vectors) != len(items):
            raise EmbeddingError(
                f"The embedding backend returned {len(vectors)} vectors for "
                f"{len(items)} inputs"
            )
        if self._dimension is not None:
            wrong = [len(vector) for vector in vectors if len(vector) != self._dimension]
            if wrong:
                raise EmbeddingError(
                    f"The backend returned vectors of width {sorted(set(wrong))} "
                    f"where {self._dimension} was expected"
                )
        return vectors

    async def _embed_raw(self, batch: list[str]) -> list[list[float]]:
        if self.backend == "local":
            return await self._embed_local(batch)
        if self.backend == "cloudflare":
            return await self._embed_cloudflare(batch)
        return await self._embed_http(batch)

    async def _embed_cloudflare(self, batch: list[str]) -> list[list[float]]:
        if self._client is None:
            self._open_cloudflare_client()
        try:
            return await embed_texts_cloudflare(
                batch, self._settings, client=self._client
            )
        except (CloudflareQuotaExhaustedError, CloudflareAuthError, CloudflareAIError) as error:
            if isinstance(error, EmbeddingError):
                raise
            raise EmbeddingError(str(error)) from error

    async def _embed_local(self, batch: list[str]) -> list[list[float]]:
        model = self._model
        if model is None:
            raise EmbeddingError("The local embedding model is not loaded")

        def _run():
            # Normalised so cosine distance and inner product agree, and so the
            # similarity reported to callers is a clean [0, 1].
            return model.encode(
                batch, normalize_embeddings=True, convert_to_numpy=True
            ).tolist()

        try:
            return await asyncio.to_thread(_run)
        except Exception as error:  # noqa: BLE001 - one failure mode for callers
            raise EmbeddingError(f"Local embedding failed: {error}") from error

    async def _embed_http(self, batch: list[str]) -> list[list[float]]:
        import httpx

        client = self._client
        if client is None:
            self._open_http_client()
            client = self._client

        if self._settings.embedding_api_format == "texts":
            payload: dict = {"texts": batch}
        else:
            payload = {"model": self._settings.embedding_model, "input": batch}

        endpoint = self._settings.embedding_endpoint
        try:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as error:
            # The body carries the reason (unknown model, input too long) and is
            # not sensitive; the Authorization header is never logged.
            raise EmbeddingError(
                f"The embedding server returned {error.response.status_code}: "
                f"{error.response.text[:300]}"
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            raise EmbeddingError(
                f"Could not reach the embedding server at {endpoint}: {error}"
            ) from error

        return _vectors_from(body, len(batch))

    async def verify_dimension(self, expected: int) -> None:
        """Refuse to run when the model and the column disagree."""
        actual = await self.load()
        if actual != expected:
            raise EmbeddingError(
                f"Embedding width mismatch: {self.model_name} produces {actual}-d "
                f"vectors but the rag_chunks.embedding column is {expected}-d. "
                f"Set RAG_EMBEDDING_DIM={actual} and re-run the migration "
                f"(existing rows must be re-embedded)."
            )


def _vectors_from(body: object, expected_count: int) -> list[list[float]]:
    """Read vectors out of an OpenAI-shaped response.

    Three dialects are read without configuration, because the response is
    unambiguous where the request is not:

        {"data": [{"embedding": [...]}, ...]}   OpenAI, vLLM, Ollama
        {"vectors": [[...], ...]}               the BGE-M3 service in .env
        {"embeddings": [[...], ...]}            some TEI builds
        [[...], ...]                            bare TEI
    """
    if isinstance(body, list):
        rows = body
    elif isinstance(body, dict) and isinstance(body.get("data"), list):
        rows = [item.get("embedding") if isinstance(item, dict) else item for item in body["data"]]
    elif isinstance(body, dict) and isinstance(body.get("vectors"), list):
        rows = body["vectors"]
    elif isinstance(body, dict) and isinstance(body.get("embeddings"), list):
        rows = body["embeddings"]
    else:
        raise EmbeddingError(f"Unrecognised embedding response shape: {type(body).__name__}")

    vectors = [[float(value) for value in row] for row in rows if row]
    if len(vectors) != expected_count:
        raise EmbeddingError(
            f"The embedding server returned {len(vectors)} vectors for {expected_count} inputs"
        )
    return vectors


# -- process-wide singleton -------------------------------------------------

_embedder: Embedder | None = None
_singleton_lock = asyncio.Lock()


async def get_embedder() -> Embedder:
    """The one Embedder for this process, loading it on first use."""
    global _embedder
    if _embedder is not None and _embedder.ready:
        return _embedder

    async with _singleton_lock:
        if _embedder is None:
            _embedder = Embedder()
    await _embedder.load()
    return _embedder


async def warmup() -> bool:
    """Load the model at startup so no request pays for it. Never raises."""
    try:
        await get_embedder()
        return True
    except Exception as error:  # noqa: BLE001 - a cold model must not stop boot
        logger.warning(
            "rag: embedding model not ready at startup (%s). "
            "The RAG routes will retry on first use; the rest of the service is unaffected.",
            error,
        )
        return False


async def shutdown() -> None:
    global _embedder
    if _embedder is not None:
        await _embedder.close()
        _embedder = None
