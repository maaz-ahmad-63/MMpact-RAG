# MMpact RAG Service

MMpact RAG is a production-grade document intelligence, hybrid retrieval, and semantic search platform powered by FastAPI, PostgreSQL, pgvector, and multi-provider inference (Local PyTorch or Cloudflare Workers AI). It provides a secure, robust backend for AI agents and enterprise SaaS applications to ingest documents (PDF, DOCX, TXT), perform hybrid retrieval (Dense pgvector + Sparse Full-Text Search + Reciprocal Rank Fusion + Cross-Encoder Reranking), and generate grounded answers.

---

## Architecture

```
Client / Frontend UI / AI Agents
        |
        | HTTP REST API (FastAPI)
        v
+--------------------------------------------------------------------+
|                         MMpact RAG Backend                         |
|                                                                    |
|  [Document Ingestion]      [Hybrid Retrieval & Scoring]            |
|  - PDF / DOCX / TXT        - Dense pgvector Cosine Search          |
|  - Text Cleaning           - PostgreSQL Full-Text Search (tsvector)|
|  - Semantic Chunking       - Reciprocal Rank Fusion (RRF k=60)     |
|                            - Answerability Slot Evaluation         |
+--------------------------------------------------------------------+
        |                                              |
        v                                              v
+---------------------------+        +-------------------------------+
|    Inference Providers    |        |        Persistent Stores      |
|                           |        |                               |
| 1. LOCAL Mode:            |        | 1. PostgreSQL + pgvector      |
|    - BAAI/bge-m3 (1024-d) |        |    - rag_documents            |
|    - BAAI/bge-reranker    |        |    - rag_chunks (vector col)  |
| 2. CLOUDFLARE Mode:       |        | 2. MongoDB (Optional)         |
|    - @cf/baai/bge-m3      |        |    - Search audit & telemetry |
|    - @cf/baai/bge-reranker|        +-------------------------------+
+---------------------------+
```

---

## Inference Providers: LOCAL vs CLOUDFLARE

MMpact RAG supports two distinct inference execution modes without breaking existing workflows:

### 1. LOCAL Mode (Default)
- **Embeddings**: In-process PyTorch / `sentence-transformers` using `BAAI/bge-m3`. Produces 1024-dimensional dense vectors normalized to unit L2 norm.
- **Reranker**: In-process `CrossEncoder` using `BAAI/bge-reranker-base`.
- **Ideal For**: Local development, offline workstations, or dedicated GPU instances.
- **Footprint**: Requires ~2 GB of memory and downloads model weights to disk on first run.

### 2. CLOUDFLARE Mode (Edge Inference)
- **Embeddings**: Cloudflare Workers AI `@cf/baai/bge-m3` REST API.
- **Reranker**: Cloudflare Workers AI `@cf/baai/bge-reranker-base` REST API.
- **Zero Local ML Download**: Starts instantly without downloading multi-gigabyte models or initializing PyTorch into process memory.
- **Ideal For**: Production deployment on cloud providers with constrained RAM (such as Render 512 MB instances), edge deployments, or serverless containers.
- **Security**: Cloudflare credentials (`CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`) remain securely isolated on the backend and are **never** logged or exposed to the frontend.

---

## Configuration

Set the inference provider and credentials in your `.env` file:

```env
# ---------------------------------------------------------------------------
# Inference Provider Selection
# Options: local | cloudflare (defaults to local)
# ---------------------------------------------------------------------------
EMBEDDING_PROVIDER=local
RERANKER_PROVIDER=local

# ---------------------------------------------------------------------------
# Cloudflare Workers AI Credentials
# Required when EMBEDDING_PROVIDER=cloudflare or RERANKER_PROVIDER=cloudflare
# ---------------------------------------------------------------------------
CLOUDFLARE_ACCOUNT_ID=your_cloudflare_account_id_here
CLOUDFLARE_API_TOKEN=your_cloudflare_api_token_here
CLOUDFLARE_EMBEDDING_MODEL=@cf/baai/bge-m3
CLOUDFLARE_RERANKER_MODEL=@cf/baai/bge-reranker-base
CLOUDFLARE_TIMEOUT_S=30.0

# ---------------------------------------------------------------------------
# Database & Vector Settings
# ---------------------------------------------------------------------------
RAG_DATABASE_URL=postgresql://user:password@host:5432/rag_db
RAG_EMBEDDING_DIM=1024
RAG_DISTANCE_METRIC=cosine
RAG_SIMILARITY_THRESHOLD=0.70
```

---

## Embedding & Reranker Compatibility

### Embedding Space Compatibility
- **1024-d Dense Width**: Both local `BAAI/bge-m3` and Cloudflare `@cf/baai/bge-m3` generate 1024-dimensional vectors.
- **L2 Vector Normalization**: Both implementations strictly enforce L2 normalization ($\|v\|_2 = 1.0$), ensuring that pgvector's cosine distance operator `<=>` ($1 - u \cdot v$) produces valid $[0, 1]$ similarities.
- **Migration Rule**: Before switching an existing database populated with local embeddings to Cloudflare embeddings, test compatibility using `scripts/test_cloudflare.py --compare-local`. If divergence is detected, run the transactional migration procedure:
  ```bash
  python scripts/reindex_embeddings.py
  ```
  This re-embeds all stored chunks in batches using the new provider so incompatible vector spaces are never mixed in search.

### Reranker Compatibility
- Both providers pass candidates through MMpact RAG's **Answerability Slot Evaluation Layer** (which scans for targeted slots such as course codes, semesters, fees, and types).
- If the Cloudflare cross-encoder returns scores in logit space ($< 0.0$ or $> 1.0$), scores are automatically transformed via sigmoid $\sigma(x) = \frac{1}{1 + e^{-x}}$ to ensure consistent $[0, 1]$ evidence scoring and candidate ranking.

---

## Diagnostic & Testing Tools

### 1. Run Comprehensive Unit Tests
```bash
# Test Cloudflare inference adapters, dimension validation, and quota handling:
pytest tests/test_cloudflare_inference.py -v

# Run the full test suite:
pytest tests/
```

### 2. Verify Cloudflare Model Endpoints (CLI Diagnostic Tool)
```bash
# Verify live endpoints with configured credentials:
python scripts/test_cloudflare.py

# Offline / CI mock verification:
python scripts/test_cloudflare.py --mock

# Compare local vs Cloudflare cosine fidelity:
python scripts/test_cloudflare.py --compare-local
```

### 3. Safe Re-indexing & Vector Migration Procedure
```bash
# Dry run (inspect chunks without updating database):
python scripts/reindex_embeddings.py --dry-run

# Re-index all database chunks using active provider:
python scripts/reindex_embeddings.py

# Re-index chunks for a specific tenant:
python scripts/reindex_embeddings.py --tenant-id default
```

---

## Deployment on Render

MMpact RAG is engineered to deploy seamlessly on Render using Cloudflare Workers AI for inference:

1. **Create Web Service on Render**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
2. **Configure Render Environment Variables**:
   ```env
   EMBEDDING_PROVIDER=cloudflare
   RERANKER_PROVIDER=cloudflare
   CLOUDFLARE_ACCOUNT_ID=<your-account-id>
   CLOUDFLARE_API_TOKEN=<your-api-token>
   RAG_DATABASE_URL=<your-postgresql-url-with-pgvector>
   RAG_API_TOKEN=<your-service-api-token>
   ```
3. **Benefits on Render**:
   - **Low Memory Footprint**: Uses ~70–90 MB RAM instead of ~2.2 GB RAM.
   - **Fast Cold Start**: Boots in ~1–2 seconds; does not trigger PyTorch model downloads.
   - **Controlled Quota Handling**: If Cloudflare rate limits or quota is exceeded (HTTP 429), the API returns a controlled error without crashing the server.

---

## API References & Examples

### 1. Root Health Check
- **Endpoint**: `GET /` or `GET /health`
- **Auth Required**: No
- **Response**:
  ```json
  {
    "status": "healthy",
    "service": "MMpact RAG Service"
  }
  ```

### 2. RAG Diagnostic Health
- **Endpoint**: `GET /api/rag/health`
- **Description**: Verifies database connection, column dimensions, active provider (`local` vs `cloudflare`), and model availability.
- **Example Curl**:
  ```bash
  curl -X GET http://localhost:8000/api/rag/health \
    -H "Authorization: Bearer <your-rag-api-token>"
  ```

### 3. Upload & Ingest Document
- **Endpoint**: `POST /api/rag/upload`
- **Request Format**: `multipart/form-data`
  - `file`: PDF, DOCX, or TXT file (Required).
  - `document_id`: Optional custom ID. Autogenerated if omitted.
  - `metadata`: Optional JSON string (e.g. `{"tenant":"acme"}`).
- **Example Curl**:
  ```bash
  curl -X POST http://localhost:8000/api/rag/upload \
    -H "Authorization: Bearer <your-rag-api-token>" \
    -F "file=@course_pack.pdf" \
    -F "metadata={\"department\":\"CSE\"}"
  ```

### 4. Hybrid Search & Reranking
- **Endpoint**: `POST /api/rag/search`
- **Description**: Performs dense pgvector retrieval, PostgreSQL Full-Text Search, Reciprocal Rank Fusion (RRF), and cross-encoder reranking.
- **Request Body**:
  ```json
  {
    "query": "What is the course code for Mobile Application Development?",
    "top_k": 5,
    "threshold": 0.50
  }
  ```

### 5. Ask Pipeline (/api/rag/ask)
- **Endpoint**: `POST /api/rag/ask`
- **Description**: Grounded question answering pipeline with citation attribution and source verification.
- **Request Body**:
  ```json
  {
    "query": "What semester is Mobile Application Development offered in?",
    "stream": false
  }
  ```

---

## Frontend Web Application

The frontend is an enterprise SaaS dashboard built with React 19, TypeScript, Vite, and Tailwind CSS.

### Running Frontend Locally
```bash
cd frontend
npm install
npm run dev
```
Access the dashboard at `http://localhost:5173`.
