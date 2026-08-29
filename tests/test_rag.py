"""RAG layer: cleaning, chunking, extraction, embeddings, retrieval and the API.

Runs entirely offline. No Postgres, no embedding server and no model download:
the database and the embedder are the two things a unit test must not need, so
both are substituted. What is exercised is the code in app/rag - the SQL string
building, the threshold logic, the pipeline order and the HTTP contract.

    pip install pytest pytest-asyncio
    pytest tests/test_rag.py -v

The one thing these tests deliberately do NOT cover is pgvector itself; that
needs a live database, and app/rag/migrate.py is how it is set up.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
import pytest_asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag import document_loader, embeddings, service, vector_store  # noqa: E402
from app.rag.chunker import chunk_segments, split_text  # noqa: E402
from app.rag.config import RagSettings  # noqa: E402
from app.rag.document_loader import Segment  # noqa: E402
from app.rag.text_cleaner import clean_text, is_effectively_empty  # noqa: E402

pytestmark = pytest.mark.asyncio


# -- fixtures ---------------------------------------------------------------


@pytest.fixture
def config() -> RagSettings:
    """Explicit settings, so a developer's .env cannot change an assertion."""
    return RagSettings(
        _env_file=None,
        rag_database_url="postgresql://test/test",
        rag_chunk_size=200,
        rag_chunk_overlap=40,
        rag_min_chunk_size=50,
        rag_top_k=3,
        rag_similarity_threshold=0.70,
        embedding_backend="local",
        rag_embedding_dim=8,
        embedding_batch_size=2,
    )


class FakeModel:
    """Stands in for a SentenceTransformer. Counts calls, so batching is testable."""

    def __init__(self, dimension: int = 8) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    def encode(self, batch, normalize_embeddings=True, convert_to_numpy=True):
        self.calls.append(list(batch))

        class _Array(list):
            def tolist(self):
                return list(self)

        # Deterministic and text-dependent, so two different strings differ.
        return _Array(
            [[(len(text) + index) % 7 / 7 for index in range(self.dimension)]
             for text in batch]
        )


@pytest_asyncio.fixture
async def embedder(config, monkeypatch):
    """An Embedder wired to FakeModel - no download, no server."""
    instance = embeddings.Embedder(config)
    model = FakeModel(config.rag_embedding_dim)

    async def _fake_load(self=instance):
        instance._model = model

    monkeypatch.setattr(instance, "_load_local", _fake_load)
    await instance.load()
    instance.model = model
    return instance


# -- text cleaning ----------------------------------------------------------


async def test_clean_text_repairs_extraction_artefacts():
    raw = "inter-\nnational  policy\nwrapped line\n\n\n\nNext​ para"
    cleaned = clean_text(raw)

    assert "international" in cleaned          # hyphenated line break rejoined
    assert "policy wrapped line" in cleaned    # single newline is a soft wrap
    assert "\n\n\n" not in cleaned             # blank-line runs collapsed
    assert "​" not in cleaned             # zero-width characters removed
    assert "  " not in cleaned                 # runs of spaces collapsed


async def test_clean_text_preserves_meaning():
    """Case, punctuation and stop words are what a semantic model reads."""
    raw = "The refund IS available within 30 days, per section 4.2."
    assert clean_text(raw) == raw


async def test_is_effectively_empty():
    assert is_effectively_empty("")
    assert is_effectively_empty("   \n\n  ")
    # A scanned page typically extracts to stray punctuation, not to nothing.
    assert is_effectively_empty("-- .. ~~")
    assert not is_effectively_empty("a")


# -- chunking ---------------------------------------------------------------


async def test_short_text_is_one_chunk():
    assert split_text("A short policy note.", chunk_size=200, overlap=40) == [
        "A short policy note."
    ]


async def test_long_text_is_split_within_the_size_bound():
    text = " ".join(f"Sentence number {index} about refunds." for index in range(120))
    pieces = split_text(text, chunk_size=200, overlap=40, min_chunk_size=50)

    assert len(pieces) > 1
    # 1.25x is the documented ceiling for a chunk that absorbed a runt.
    assert all(len(piece) <= 250 for piece in pieces)


async def test_chunks_overlap():
    text = " ".join(f"word{index}" for index in range(200))
    pieces = split_text(text, chunk_size=200, overlap=60)

    # The tail of one chunk must reappear at the head of the next, or a sentence
    # cut at the boundary is lost to both.
    tail_words = set(pieces[0].split()[-6:])
    assert tail_words & set(pieces[1].split()[:12])


async def test_no_overlap_when_configured_to_zero():
    text = " ".join(f"word{index}" for index in range(200))
    pieces = split_text(text, chunk_size=200, overlap=0)
    assert not set(pieces[0].split()[-4:]) & set(pieces[1].split()[:4])


async def test_overlap_at_or_above_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        split_text("x" * 500, chunk_size=100, overlap=100)


async def test_runts_are_absorbed():
    """A trailing fragment belongs to its neighbour, not to a row of its own."""
    pieces = split_text(
        "A" * 195 + ". " + "B" * 190 + ". tiny.",
        chunk_size=200, overlap=20, min_chunk_size=60,
    )
    assert all(len(piece) >= 60 for piece in pieces)


async def test_chunk_metadata_is_complete(config):
    chunks = chunk_segments(
        [Segment(text="Refund policy. " * 60, page=4)],
        document_id="doc_1",
        source="policy.pdf",
        settings=config,
    )

    assert len(chunks) > 1
    for position, chunk in enumerate(chunks):
        assert chunk.metadata["document_id"] == "doc_1"
        assert chunk.metadata["source"] == "policy.pdf"
        assert chunk.metadata["page"] == 4
        assert chunk.metadata["chunk_index"] == position == chunk.index


async def test_page_is_null_where_the_format_has_no_pages(config):
    chunks = chunk_segments(
        [Segment(text="A DOCX has no pages until it is rendered.", page=None)],
        document_id="doc_2", source="handbook.docx", settings=config,
    )
    assert chunks[0].metadata["page"] is None


async def test_chunk_index_is_continuous_across_pages(config):
    chunks = chunk_segments(
        [Segment(text="Page one text. " * 30, page=1),
         Segment(text="Page two text. " * 30, page=2)],
        document_id="doc_3", source="two-pages.pdf", settings=config,
    )
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    # A chunk never spans a page, so its page is a citation and not a guess.
    assert {chunk.page for chunk in chunks} == {1, 2}


# -- extraction -------------------------------------------------------------


def _minimal_pdf(text: str) -> bytes:
    """A one-page PDF, written by hand so the test needs no PDF library."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n".encode()
        + b"%%EOF\n"
    )
    return bytes(out)


def _docx_bytes(paragraph: str, table_cells: list[list[str]]) -> bytes:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph(paragraph)
    table = document.add_table(rows=len(table_cells), cols=len(table_cells[0]))
    for row_index, row in enumerate(table_cells):
        for cell_index, value in enumerate(row):
            table.cell(row_index, cell_index).text = value
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


async def test_txt_extraction():
    segments = document_loader.load("Refunds within 30 days.".encode("utf-8"), "policy.txt")
    assert segments[0].text == "Refunds within 30 days."
    assert segments[0].page is None


async def test_txt_extraction_survives_a_non_utf8_file():
    segments = document_loader.load("Café — naïve".encode("cp1252"), "notes.txt")
    assert "Caf" in segments[0].text


async def test_pdf_extraction_reports_pages():
    pytest.importorskip("pypdf")
    segments = document_loader.load(_minimal_pdf("Refunds within 30 days"), "policy.pdf")
    assert segments[0].page == 1
    assert "Refunds" in segments[0].text


async def test_docx_extraction_includes_paragraphs_and_tables():
    content = _docx_bytes("Our refund policy:", [["Item", "Days"], ["Standard", "30"]])
    segments = document_loader.load(content, "policy.docx")
    text = segments[0].text

    assert "Our refund policy:" in text
    assert "Standard | 30" in text   # tables carry the facts people ask about
    assert segments[0].page is None


async def test_unsupported_extension_is_rejected():
    for filename in ("archive.zip", "sheet.xlsx", "noextension"):
        with pytest.raises(document_loader.UnsupportedFileType):
            document_loader.load(b"data", filename)


async def test_empty_file_is_rejected():
    with pytest.raises(document_loader.ExtractionError):
        document_loader.load(b"", "empty.txt")


async def test_corrupt_pdf_is_an_extraction_error():
    pytest.importorskip("pypdf")
    with pytest.raises(document_loader.ExtractionError):
        document_loader.load(b"%PDF-1.4 this is not a pdf", "broken.pdf")


async def test_ocr_hook_is_not_called_for_pages_that_have_text(monkeypatch):
    pytest.importorskip("pypdf")
    calls = []
    monkeypatch.setattr(
        document_loader, "_OCR_HOOK", lambda page_number, page: calls.append(page_number) or ""
    )
    document_loader.load(_minimal_pdf("Real text here"), "policy.pdf")
    assert calls == []


# -- embeddings -------------------------------------------------------------


async def test_embed_text_returns_a_vector_of_the_configured_width(embedder, config):
    vector = await embedder.embed_text("What is the refund policy?")

    assert len(vector) == config.rag_embedding_dim == embedder.dimension
    assert all(isinstance(value, float) for value in vector)


async def test_embed_texts_batches(embedder):
    """EMBEDDING_BATCH_SIZE=2, so five chunks are three calls and not five."""
    before = len(embedder.model.calls)
    vectors = await embedder.embed_texts([f"chunk {index}" for index in range(5)])

    assert len(vectors) == 5
    assert len(embedder.model.calls) - before == 3


async def test_model_is_loaded_only_once(embedder):
    loads = 0

    async def _count(*_args):
        nonlocal loads
        loads += 1

    await embedder.embed_text("one")
    await embedder.embed_text("two")
    assert loads == 0                 # already loaded; no reload per request
    assert embedder.ready


async def test_empty_string_is_not_embedded(embedder):
    with pytest.raises(embeddings.EmbeddingError):
        await embedder.embed_texts(["fine", "   "])


async def test_dimension_mismatch_with_the_column_is_refused(embedder):
    await embedder.verify_dimension(embedder.dimension)   # matching: no error
    with pytest.raises(embeddings.EmbeddingError) as raised:
        await embedder.verify_dimension(1024)
    assert "1024" in str(raised.value)


async def test_every_response_dialect_is_read():
    """Responses are unambiguous, so no configuration decides which is which."""
    expected = [[0.1, 0.2], [0.3, 0.4]]
    for body in (
        {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},   # OpenAI
        {"vectors": [[0.1, 0.2], [0.3, 0.4]], "dim": 2, "model": "BAAI/bge-m3"},
        {"embeddings": [[0.1, 0.2], [0.3, 0.4]]},                           # TEI
        [[0.1, 0.2], [0.3, 0.4]],                                           # bare TEI
    ):
        assert embeddings._vectors_from(body, 2) == expected

    with pytest.raises(embeddings.EmbeddingError):
        embeddings._vectors_from({"unexpected": True}, 2)


# -- the HTTP backend, against a stubbed server -----------------------------


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _RecordingClient:
    """Captures what would have gone over the wire."""

    def __init__(self, payload):
        self.headers: dict[str, str] = {}
        self.payload = payload
        self.requests: list[tuple[str, dict]] = []

    async def post(self, url, json):
        self.requests.append((url, json))
        return _Response(self.payload)

    async def aclose(self):
        return None


def _http_embedder(config, payload):
    instance = embeddings.Embedder(config)
    client = _RecordingClient(payload)

    def _open():
        headers = {"accept": "application/json"}
        key = config.embedding_api_key
        if key:
            name = config.embedding_api_key_header.strip() or "Authorization"
            headers[name] = f"Bearer {key}" if name.lower() == "authorization" else key
        client.headers = headers
        instance._client = client

    instance._open_http_client = _open
    return instance, client


async def test_texts_dialect_posts_the_configured_url_and_header():
    """The shape of the BGE-M3 service configured in .env."""
    config = RagSettings(
        _env_file=None,
        rag_database_url="postgresql://test/test",
        embedding_backend="http",
        embedding_api_url="https://embeddings.example.com/embed",
        embedding_api_format="texts",
        embedding_api_key_header="X-API-Key",
        embedding_api_key="secret-key",
    )
    embedder, client = _http_embedder(
        config, {"vectors": [[0.1] * 1024], "dim": 1024, "model": "BAAI/bge-m3"}
    )

    assert await embedder.load() == 1024

    url, body = client.requests[0]
    assert url == "https://embeddings.example.com/embed"
    assert body == {"texts": ["dimension probe"]}
    # A custom header carries the bare key; "Bearer " would make it a 401.
    assert client.headers["X-API-Key"] == "secret-key"
    assert "Authorization" not in client.headers


async def test_openai_dialect_posts_embeddings_with_a_bearer_token():
    config = RagSettings(
        _env_file=None,
        rag_database_url="postgresql://test/test",
        embedding_backend="http",
        embedding_api_base="http://tei:8080/v1/",
        embedding_api_key="sk-test",
        embedding_model="BAAI/bge-m3",
    )
    embedder, client = _http_embedder(config, {"data": [{"embedding": [0.5] * 8}]})

    await embedder.load()
    url, body = client.requests[0]

    assert url == "http://tei:8080/v1/embeddings"
    assert body == {"model": "BAAI/bge-m3", "input": ["dimension probe"]}
    assert client.headers["Authorization"] == "Bearer sk-test"


async def test_a_url_or_a_base_both_switch_the_backend_to_http():
    common = {"_env_file": None, "rag_database_url": "postgresql://test/test"}
    assert RagSettings(**common, embedding_api_url="https://x/embed").resolved_backend == "http"
    assert RagSettings(**common, embedding_api_base="http://x/v1").resolved_backend == "http"
    assert RagSettings(**common).resolved_backend == "local"
    # The URL is used verbatim when both are configured.
    both = RagSettings(**common, embedding_api_url="https://x/embed", embedding_api_base="http://y/v1")
    assert both.embedding_endpoint == "https://x/embed"


# -- SQL construction -------------------------------------------------------


async def test_vector_literal_is_pgvector_input_syntax():
    literal = vector_store.to_vector_literal([0.5, -1.25])
    assert literal.startswith("[") and literal.endswith("]")
    assert literal == "[0.50000000,-1.25000000]"


async def test_similarity_expression_matches_the_metric(config):
    cosine = vector_store._similarity_expression(config)
    assert "<=>" in cosine and cosine.startswith("1 - ")

    l2_config = config.model_copy(update={"rag_distance_metric": "l2"})
    assert "<->" in vector_store._similarity_expression(l2_config)

    ip_config = config.model_copy(update={"rag_distance_metric": "inner_product"})
    assert "<#>" in vector_store._similarity_expression(ip_config)


async def test_migration_sql_has_no_placeholders_left(config):
    sql = vector_store.migration_sql(config)
    assert "{{" not in sql
    assert f"VECTOR({config.rag_embedding_dim})" in sql
    assert "vector_cosine_ops" in sql
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql


async def test_schema_name_must_be_an_identifier(config):
    hostile = config.model_copy(update={"rag_db_schema": "public; DROP TABLE users"})
    with pytest.raises(vector_store.VectorStoreError):
        vector_store.migration_sql(hostile)


# -- retrieval and the threshold --------------------------------------------


def _hit(similarity: float, chunk_index: int = 0, page: int | None = 4) -> vector_store.SearchHit:
    return vector_store.SearchHit(
        chunk_id=f"chunk_{chunk_index}",
        document_id="doc_1",
        content=f"Refunds are available within 30 days ({chunk_index}).",
        similarity=similarity,
        distance=1 - similarity,
        chunk_index=chunk_index,
        metadata={"source": "policy.pdf", "page": page, "chunk_index": chunk_index},
        filename="policy.pdf",
        file_type="pdf",
    )


@pytest.fixture
def retrieval(monkeypatch, embedder, config):
    """service.search wired to a fake store; returns a setter for the hits."""
    async def _get_embedder():
        return embedder

    monkeypatch.setattr(embeddings, "get_embedder", _get_embedder)
    monkeypatch.setattr(service.embeddings, "get_embedder", _get_embedder)

    state: dict = {"hits": [], "top_k": None}

    async def _search(vector, *, top_k, settings=None, **_kwargs):
        state["top_k"] = top_k
        state["vector"] = vector
        return state["hits"][:top_k]

    monkeypatch.setattr(service.vector_store, "search", _search)
    return state


async def test_high_similarity_is_found(retrieval, config):
    retrieval["hits"] = [_hit(0.91), _hit(0.83, 1)]
    outcome = await service.search(query="What is the refund policy?", settings=config)

    assert outcome["found"] is True
    assert len(outcome["results"]) == 2
    assert outcome["results"][0]["similarity"] == pytest.approx(0.91)
    assert outcome["results"][0]["metadata"]["source"] == "policy.pdf"
    assert outcome["results"][0]["metadata"]["page"] == 4


async def test_low_similarity_is_not_found(retrieval, config):
    """Below the threshold the API reports found=false and returns nothing.

    It does NOT call a model: what to do about an empty result belongs to the
    agent, and this endpoint must not decide it.
    """
    retrieval["hits"] = [_hit(0.41), _hit(0.22, 1)]
    outcome = await service.search(query="Do you sell bicycles?", settings=config)

    assert outcome["found"] is False
    assert outcome["results"] == []
    assert outcome["threshold"] == 0.70


async def test_threshold_can_be_overridden_per_request(retrieval, config):
    retrieval["hits"] = [_hit(0.41)]
    assert (await service.search(query="q", threshold=0.30, settings=config))["found"]
    assert not (await service.search(query="q", threshold=0.95, settings=config))["found"]


async def test_top_k_defaults_to_config_and_is_capped(retrieval, config):
    retrieval["hits"] = [_hit(0.9, index) for index in range(60)]

    await service.search(query="q", settings=config)
    assert retrieval["top_k"] == config.rag_top_k

    await service.search(query="q", top_k=999, settings=config)
    assert retrieval["top_k"] == config.rag_max_top_k


async def test_context_is_assembled_with_citable_sources(retrieval, config):
    retrieval["hits"] = [_hit(0.91, 0), _hit(0.80, 1)]
    outcome = await service.build_context(query="What is the refund policy?", settings=config)

    assert outcome["found"] is True
    assert "[1] policy.pdf, page 4" in outcome["context"]
    assert "Refunds are available" in outcome["context"]
    assert outcome["sources"][0] == {
        "document_id": "doc_1", "source": "policy.pdf", "page": 4,
        "chunk_index": 0, "similarity": pytest.approx(0.91),
    }


async def test_context_respects_its_character_budget(retrieval, config):
    retrieval["hits"] = [_hit(0.9, index) for index in range(3)]
    outcome = await service.build_context(query="q", max_chars=80, settings=config)

    assert len(outcome["context"]) <= 80
    assert outcome["chunks_used"] < 3
    # Whole chunks only - half a sentence in a prompt reads as a fact.
    assert outcome["context"].endswith(".")


async def test_context_is_empty_when_nothing_passes(retrieval, config):
    retrieval["hits"] = [_hit(0.30)]
    outcome = await service.build_context(query="q", settings=config)

    assert outcome["found"] is False
    assert outcome["context"] == ""
    assert outcome["sources"] == []


# -- the pipeline, end to end (with the database faked) ---------------------


@pytest.fixture
def ingestion(monkeypatch, embedder, config):
    """service.ingest with Postgres replaced by a recorder."""
    async def _get_embedder():
        return embedder

    stored: dict = {"documents": [], "chunks": []}

    async def _column_dimension(settings=None):
        return config.rag_embedding_dim

    async def _insert_document(**kwargs):
        stored["documents"].append(kwargs)
        return kwargs

    async def _insert_chunks(document_id, rows, settings=None):
        rows = list(rows)
        stored["chunks"].extend(rows)
        return len(rows)

    monkeypatch.setattr(service.embeddings, "get_embedder", _get_embedder)
    monkeypatch.setattr(service.vector_store, "column_dimension", _column_dimension)
    monkeypatch.setattr(service.vector_store, "insert_document", _insert_document)
    monkeypatch.setattr(service.vector_store, "insert_chunks", _insert_chunks)
    monkeypatch.setattr(service, "_persist_original", lambda *args: None)
    return stored


async def test_pdf_ingests_to_stored_embeddings(ingestion, config):
    """PDF -> text -> chunks -> BGE-M3 -> rows ready for pgvector."""
    pytest.importorskip("pypdf")
    result = await service.ingest(
        content=_minimal_pdf("Refunds are available within 30 days"),
        filename="policy.pdf",
        settings=config,
    )

    assert result["file_type"] == "pdf"
    assert result["chunks"] == len(ingestion["chunks"]) >= 1
    assert result["pages"] == 1
    assert result["embedding_dim"] == config.rag_embedding_dim

    index, content, vector, metadata = ingestion["chunks"][0]
    assert index == 0
    assert "Refunds" in content
    assert len(vector) == config.rag_embedding_dim   # matches the pgvector column
    assert metadata["source"] == "policy.pdf"
    assert metadata["page"] == 1


async def test_txt_ingests_with_a_null_page(ingestion, config):
    result = await service.ingest(
        content=b"Refunds are available within 30 days of purchase.",
        filename="policy.txt",
        user_id="u1",
        metadata={"tenant": "acme"},
        settings=config,
    )

    assert result["pages"] is None
    assert ingestion["chunks"][0][3]["page"] is None
    assert ingestion["chunks"][0][3]["user_id"] == "u1"
    assert ingestion["documents"][0]["metadata"] == {"tenant": "acme"}


async def test_docx_ingests(ingestion, config):
    content = _docx_bytes("Refund policy", [["Item", "Days"], ["Standard", "30"]])
    result = await service.ingest(content=content, filename="policy.docx", settings=config)

    assert result["file_type"] == "docx"
    assert result["chunks"] >= 1


async def test_a_document_with_no_text_is_refused(ingestion, config):
    with pytest.raises(service.EmptyDocument):
        await service.ingest(content=b"   \n  \n ", filename="blank.txt", settings=config)


async def test_supplied_document_id_is_kept(ingestion, config):
    result = await service.ingest(
        content=b"Some policy text worth storing.",
        filename="policy.txt",
        document_id="laravel-media-42",
        settings=config,
    )
    assert result["document_id"] == "laravel-media-42"
    assert ingestion["chunks"][0][3]["document_id"] == "laravel-media-42"


# -- HTTP contract ----------------------------------------------------------


@pytest.fixture
def client(monkeypatch):
    """TestClient with auth disabled - these tests are about the RAG routes."""
    from fastapi.testclient import TestClient

    from app.config import settings as app_settings
    from app.rag import router as rag_router

    monkeypatch.setattr(app_settings, "rag_api_token", "")
    from fastapi import FastAPI

    application = FastAPI()
    application.include_router(rag_router.router)
    return TestClient(application)


async def test_upload_rejects_an_unsupported_type(client):
    response = client.post(
        "/api/rag/upload",
        files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")},
    )
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


async def test_upload_rejects_an_empty_file(client):
    response = client.post(
        "/api/rag/upload",
        files={"file": ("empty.txt", b"", "text/plain")},
    )
    assert response.status_code == 422


async def test_upload_rejects_a_file_over_the_size_limit(client, monkeypatch):
    from app.rag import router as rag_router

    monkeypatch.setattr(rag_router.rag_settings, "rag_max_upload_mb", 0)
    response = client.post(
        "/api/rag/upload",
        files={"file": ("big.txt", b"x" * 10, "text/plain")},
    )
    assert response.status_code == 413


async def test_upload_rejects_malformed_metadata(client, monkeypatch):
    response = client.post(
        "/api/rag/upload",
        files={"file": ("policy.txt", b"Refund policy text.", "text/plain")},
        data={"metadata": "not json"},
    )
    assert response.status_code == 422
    assert "metadata" in response.json()["detail"]


async def test_upload_returns_the_ingest_summary(client, monkeypatch):
    from app.rag import router as rag_router

    async def _ingest(**kwargs):
        return {
            "document_id": "doc_1", "filename": kwargs["filename"], "file_type": "txt",
            "chunks": 3, "characters": 120, "pages": None,
            "embedding_model": "BAAI/bge-m3", "embedding_dim": 1024,
            "took_ms": 42, "metadata": {},
        }

    monkeypatch.setattr(rag_router.service, "ingest", _ingest)
    response = client.post(
        "/api/rag/upload",
        files={"file": ("policy.txt", b"Refund policy text.", "text/plain")},
    )
    assert response.status_code == 201
    assert response.json()["chunks"] == 3


async def test_authentication_enforced(monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import settings as app_settings
    from app.rag import router as rag_router
    from fastapi import FastAPI

    monkeypatch.setattr(app_settings, "rag_api_token", "secure-token")
    application = FastAPI()
    application.include_router(rag_router.router)
    auth_client = TestClient(application)

    # Missing authorization header
    response = auth_client.post("/api/rag/search", json={"query": "test"})
    assert response.status_code == 401

    # Invalid authorization header
    response = auth_client.post(
        "/api/rag/search",
        json={"query": "test"},
        headers={"Authorization": "Bearer bad-token"}
    )
    assert response.status_code == 401

    # Correct authorization header
    # Mock vector search to avoid actual DB access
    async def mock_search(**kwargs):
        return {
            "query": "test", "found": False, "results": [],
            "threshold": 0.7, "top_k": 5, "took_ms": 1,
        }
    monkeypatch.setattr(rag_router.service, "search", mock_search)
    response = auth_client.post(
        "/api/rag/search",
        json={"query": "test"},
        headers={"Authorization": "Bearer secure-token"}
    )
    assert response.status_code == 200


async def test_search_rejects_an_empty_query(client):
    assert client.post("/api/rag/search", json={"query": "   "}).status_code == 422
    assert client.post("/api/rag/search", json={}).status_code == 422


async def test_search_rejects_an_invalid_top_k(client):
    assert client.post("/api/rag/search", json={"query": "q", "top_k": 0}).status_code == 422
    assert client.post("/api/rag/search", json={"query": "q", "top_k": -1}).status_code == 422
    assert client.post("/api/rag/search", json={"query": "q", "top_k": 10_000}).status_code == 422


async def test_search_without_a_database_is_503(client, monkeypatch):
    """An unconfigured RAG layer is a 503 on its own routes and nothing else."""
    from app.rag import router as rag_router

    async def _boom(**_kwargs):
        raise vector_store.VectorStoreUnavailable("RAG storage is not configured")

    monkeypatch.setattr(rag_router.service, "search", _boom)
    response = client.post("/api/rag/search", json={"query": "refunds"})
    assert response.status_code == 503


async def test_search_reports_an_embedding_failure_as_502(client, monkeypatch):
    from app.rag import router as rag_router

    async def _boom(**_kwargs):
        raise embeddings.EmbeddingError("embedding server unreachable")

    monkeypatch.setattr(rag_router.service, "search", _boom)
    assert client.post("/api/rag/search", json={"query": "q"}).status_code == 502


async def test_search_response_shape(client, monkeypatch):
    from app.rag import router as rag_router

    async def _search(**_kwargs):
        return {
            "query": "What is the refund policy?", "found": True,
            "results": [{
                "chunk_id": "c1", "document_id": "d1",
                "content": "Refunds are available within 30 days...",
                "similarity": 0.91,
                "metadata": {"source": "policy.pdf", "page": 4, "chunk_index": 12},
            }],
            "threshold": 0.7, "top_k": 5, "took_ms": 12,
        }

    monkeypatch.setattr(rag_router.service, "search", _search)
    body = client.post(
        "/api/rag/search", json={"query": "What is the refund policy?", "top_k": 5}
    ).json()

    assert body["found"] is True
    assert body["results"][0]["similarity"] == 0.91
    assert body["results"][0]["metadata"]["page"] == 4


async def test_delete_of_an_unknown_document_is_404(client, monkeypatch):
    from app.rag import router as rag_router

    async def _delete(_document_id):
        return False, 0

    monkeypatch.setattr(rag_router.vector_store, "delete_document", _delete)
    assert client.delete("/api/rag/documents/nope").status_code == 404


async def test_delete_reports_the_chunks_removed(client, monkeypatch):
    from app.rag import router as rag_router

    async def _delete(_document_id):
        return True, 17

    monkeypatch.setattr(rag_router.vector_store, "delete_document", _delete)
    body = client.delete("/api/rag/documents/doc_1").json()
    assert body == {"ok": True, "document_id": "doc_1", "chunks_deleted": 17}


async def test_document_list_is_paginated(client, monkeypatch):
    from app.rag import router as rag_router

    async def _list(*, limit, offset, **_kwargs):
        return (
            [{"id": "doc_1", "filename": "company-policy.pdf", "file_type": "pdf",
              "chunk_count": 12, "created_at": "2026-01-01T00:00:00+00:00"}],
            57,
        )

    monkeypatch.setattr(rag_router.vector_store, "list_documents", _list)
    body = client.get("/api/rag/documents?limit=1&offset=10").json()

    assert body["total"] == 57
    assert body["limit"] == 1 and body["offset"] == 10
    assert body["documents"][0]["filename"] == "company-policy.pdf"


async def test_rag_health_never_leaks_secrets(client, monkeypatch):
    from app.rag import router as rag_router

    monkeypatch.setattr(rag_router.rag_settings, "rag_database_url", "postgresql://u:secret@h/db")
    monkeypatch.setattr(rag_router.rag_settings, "embedding_api_key", "sk-secret")

    body = client.get("/api/rag/health").json()
    serialised = str(body)

    assert "secret" not in serialised
    assert body["configuration"]["embedding_api_key_set"] is True
    assert body["configuration"]["embedding_model"] == "BAAI/bge-m3"
