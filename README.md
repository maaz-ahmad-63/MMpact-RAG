# Client-Agnostic AI Agent Service

LLM reasoning for **any** client application. It handles model calls, reasoning,
response generation, tool execution and conversation intelligence.

**Groq is the default provider.** Google Gemini and Anthropic are available by
switching one env var, or per request via the model alias map.

**No caller is assumed.** Laravel, a Shopify app, an internal tool and a curl
one-liner all take exactly the same path — nothing branches on who is asking.
`client_id` only ever selects *which* records to read; it never selects
behaviour.

The service holds no prompts, personas, configuration or business rules of its
own. Every request carries its own prompt (inline or by `prompt_id`) and its own
context. It contains no WhatsApp logic, no QR generation and no session
management; the only way it can affect WhatsApp is through the gateway's REST
API.

The one thing it does persist is what clients explicitly wrote through the Thread
and Prompt APIs, namespaced per `client_id`. Two client applications never share
a record.

## Requirements

- Python 3.11+
- A Groq API key (<https://console.groq.com/keys>)

## Setup

```bash
cp .env.example .env      # set GROQ_API_KEY and AGENT_API_TOKEN
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

---

## The three APIs

| Group | Purpose |
|---|---|
| `/agent/thread` | Conversation state for one contact on one channel |
| `/agent/prompt` | Prompts and personas a client registers for itself |
| `/agent/reply` | The single entrypoint that produces an actual LLM reply |

Auth on all of them: `Authorization: Bearer <AGENT_API_TOKEN>`.

A minimal end-to-end run, from any application:

```bash
# 1. register the persona you want used
curl -X POST localhost:8000/agent/prompt -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "client_id": "acme-app", "prompt_id": "support",
  "system_prompt": "You are the support agent for Acme. …",
  "tone": "warm", "language": "English",
  "tools_allowed": ["block_contact", "request_human_handoff"] }'

# 2. ask for a reply — the thread is resolved or created for you
curl -X POST localhost:8000/agent/reply -H "$AUTH" -H 'Content-Type: application/json' -d '{
  "client_id": "acme-app", "channel": "whatsapp", "contact_id": "923001234567",
  "prompt_id": "support", "message": "do you deliver on sundays?" }'
```

---

## Thread API

Threads are keyed by `client_id` + `channel` + `contact_id`. The id is derived
from that triple, so a caller that only knows who it is talking to resolves the
same thread every time without tracking an id — and two applications messaging
the same phone number can never land on the same conversation.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/agent/thread` | Create or get. Idempotent |
| `GET` | `/agent/thread/{id}?client_id=` | History and state |
| `POST` | `/agent/thread/{id}/reset?client_id=` | Clear the turns, keep the thread |
| `DELETE` | `/agent/thread/{id}?client_id=` | Remove it entirely |

```json
POST /agent/thread
{ "client_id": "acme-app", "channel": "whatsapp", "contact_id": "923001234567" }
```
```json
{
  "thread_id": "th_094a410f5553c0a881c8d1632fc80772",
  "client_id": "acme-app", "channel": "whatsapp", "contact_id": "923001234567",
  "created_at": "2026-07-27T12:04:29Z", "updated_at": "2026-07-27T12:04:29Z",
  "turns": [], "metadata": {}
}
```

`client_id` is required on the read routes and checked against the stored record.
Asking for another client's thread returns **404, not 403** — whether it exists is
not something a caller should be able to probe for.

Only the last `THREAD_MAX_TURNS` (default 40) are retained. This is working
memory for the model, not an archive; the calling application keeps the durable
message history in its own system.

---

## Prompt API

A prompt record is stored exactly as submitted. Nothing here is hardcoded and
nothing is interpreted beyond assembling it into a system prompt at reply time.
Prompt ids are scoped per client, so two applications may both register
`"support"` without colliding — which is also why the read routes require
`client_id`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/agent/prompt` | Register or update |
| `GET` | `/agent/prompt/{prompt_id}?client_id=` | Read it back |
| `DELETE` | `/agent/prompt/{prompt_id}?client_id=` | Remove it |

```json
POST /agent/prompt
{
  "client_id": "acme-app",
  "prompt_id": "support",
  "system_prompt": "You are the support agent for Acme…",
  "tone": "warm",
  "language": "English",
  "tools_allowed": ["block_contact", "request_human_handoff"],
  "rag_config": null
}
```

- `tools_allowed` scopes which tools this prompt may use. Unknown names are
  rejected here (`422`) rather than silently ignored at reply time. An empty list
  means this prompt runs with no tools at all.
- `rag_config` is **reserved**. It is accepted, stored and echoed back, but no
  retrieval layer reads it yet — see [Tools](#tools).
- `metadata.base_rules: false` switches off the built-in safety envelope for
  full control of the system prompt.

---

## `POST /agent/reply`

The only endpoint that produces a reply.

```json
{
  "client_id": "acme-app",
  "channel": "whatsapp",
  "contact_id": "923001234567",
  "prompt_id": "support",
  "message": "do you deliver on sundays?",
  "context": {
    "whatsapp_id": 10,
    "account_name": "Acme Store",
    "ai_settings": { "max_words": 80 },
    "contact": { "phone": "+15551234567", "push_name": "Sam" }
  }
}
```

```json
{
  "reply": "We don't deliver on Sundays, but Saturday orders arrive Monday morning.",
  "thread_id": "th_094a410f5553c0a881c8d1632fc80772",
  "tool_calls_made": ["lookup_business_info"],
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "stop_reason": "stop",
  "handoff_requested": false,
  "tool_calls": [{ "name": "lookup_business_info", "input": { "query": "delivery days" }, "ok": true, "error": null }],
  "usage": { "input_tokens": 812, "output_tokens": 41, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0 },
  "latency_ms": 1840
}
```

### Resolving the pieces

| Field | How it resolves |
|---|---|
| **thread** | `thread_id` if given; else `client_id`+`channel`+`contact_id`; else stateless, history from `context.history` |
| **prompt** | inline `system_prompt` wins; else the `prompt_id` record's `system_prompt` |
| **tools** | request `tools_allowed` wins; else the record's; else every configured tool |
| **tone / language** | the record's, else `context.ai_settings` |
| **model** | `model` (Laravel's `text_llm`) through the LLM registry; else the registry's declared default |
| **max_tokens** | `max_tokens`; else `MAX_TOKENS` |

### Choosing the model per request — the LLM registry

**Laravel is the single source of truth.** It sends one string, `text_llm`, and
that is the only input to model selection. There is no `LLM_PROVIDER`, no
`*_MODEL` and no `MODEL_ALIASES`: a model name in the environment is a model
nobody can see from the dashboard that chose it.

The translation lives in exactly one place — `app/services/llm_registry.json`:

```json
{
  "default": "groq_llama_3_3_70b",
  "providers": {
    "groq":   { "key_env": "GROQ_API_KEY",   "fallback": "groq_llama_3_3_70b",
                "prefixes": ["groq", "llama", "compound"] },
    "openai": { "key_env": "OPENAI_API_KEY", "fallback": "openai_gpt_4o_mini",
                "prefixes": ["openai", "gpt"] }
  },
  "models": {
    "groq_gpt_oss_120b":  { "provider": "groq",   "model": "openai/gpt-oss-120b" },
    "groq_llama_3_3_70b": { "provider": "groq",   "model": "llama-3.3-70b-versatile" },
    "groq_compound":      { "provider": "groq",   "model": "compound" },
    "openai_gpt_5":       { "provider": "openai", "model": "gpt-5" }
  }
}
```

`app/services/models.py` resolves against it, and every AI path goes through
that one function:

```
text_llm "groq_gpt_oss_120b"  ->  provider "groq",   model "openai/gpt-oss-120b"
text_llm "openai_gpt_5"       ->  provider "openai", model "gpt-5"
```

Each resolution is logged:

```
LLM Identifier: groq_gpt_oss_120b | Provider: groq | Model: openai/gpt-oss-120b
```

**Adding a model your dashboard offers is an edit to that JSON file and a
restart.** No code change. `LLM_REGISTRY_PATH` points at a replacement file for
deployments that would rather not rebuild the image — it is a *path*, never a
model name.

Providers are built lazily and cached, so one conversation can run on Gemini
while the next runs on Groq. Each entry pins its provider, so the registry cannot
route a name at a backend with no key configured.

#### What ships registered

`groq_gpt_oss_120b`, `groq_gpt_oss_20b`, `groq_llama_3_3_70b`,
`groq_llama_3_1_8b`, `groq_compound`, `groq_compound_mini`, `openai_gpt_5`,
`openai_gpt_4o`, `openai_gpt_4o_mini`, `gemini_3_flash`, `gemini_3_flash_lite`,
`gemini_3_pro`, `gemini_2_5_pro`, `gemini_flash`, `gemini_flash_lite`,
`gemini_pro`, `anthropic_claude_opus_5`, `anthropic_claude_sonnet_5`,
`anthropic_claude_haiku_4_5`, plus the older `llama_3_3` spellings.

`GET /health` lists them live under `llm_registry.identifiers`.

Prefer Gemini's `_latest` pointers in production: a pinned id eventually retires.
`gemini-2.5-flash` and `gemini-2.5-flash-lite` are deliberately **absent** —
Google now answers them with *"no longer available to new users"*.

#### An identifier the registry has never seen

Two things happen, in this order:

1. The vendor is read off the front of the name (`openai_gpt_6_turbo` → the
   `openai` provider, via that provider's `prefixes`) and its declared
   `fallback` model is used, so WhatsApp keeps answering.
2. It is logged as a **warning** naming the exact JSON entry to add.

The model id is never reconstructed. Turning `groq_llama_3_3_70b` into
`llama-3.3-70b-versatile` requires knowing the `-versatile` suffix; inventing it
produces a name that fails at the provider with a far more confusing error.

A name with no recognisable vendor at all is **refused** with `422` rather than
answered on something nobody chose.

A `thread_id` that does not exist is not an error — the caller may be replaying,
and refusing would lose the message. It falls through to identity resolution.

Set `"persist": false` for one-off or test calls that should not be written to
the thread. The exchange is also skipped when the provider failed and the reply
is the fallback text, so an outage does not write an apology into the history.

### Failure modes

| Situation | Response |
|---|---|
| No prompt resolvable at all | `422` — **never** a house default |
| `prompt_id` this client never registered | `404` |
| `thread_id` owned by a different `client_id` | `403` |
| Model provider unreachable / rate limited | `503` / `429` |

`context` is free-form: any keys beyond the recognised ones (`whatsapp_id`,
`account_name`, `purpose`, `ai_settings`, `contact`, `history`, `message_id`) are
rendered into the system prompt as authoritative reference facts, so a client can
extend the payload without a code change here.

### `POST /agent/classify`

```json
{ "message": "this is the third time my order is late", "labels": ["order_status", "complaint", "sales"] }
```

Returns structured conversation intelligence for routing and reporting:

```json
{
  "classification": {
    "intent": "complaint",
    "sentiment": "negative",
    "urgency": "high",
    "language": "en",
    "requires_human": true,
    "summary": "Repeat late-delivery complaint; customer is frustrated."
  },
  "provider": "groq",
  "model": "llama-3.3-70b-versatile",
  "usage": { "...": "..." },
  "latency_ms": 720
}
```

Constrained JSON output validated against the Pydantic schema before it returns,
so the shape is guaranteed to the caller.

### `GET /health`

Unauthenticated liveness probe reporting provider, model, tool availability and
whether an API key is configured.

---

## Prompt resolution

The calling application owns the prompt. Resolution order:

1. `system_prompt` sent inline on the request
2. the `system_prompt` of the record named by `prompt_id`
3. **nothing** → `422`

There is no built-in prompt library. Earlier versions shipped `support` /
`sales` / `marketing` templates used when the caller sent a blank prompt; those
are gone. Answering a real customer with a house default is exactly the guess
this service is not allowed to make.

The final system prompt is assembled from: base rules → account name → resolved
instruction → tone → language → length limit → extra instructions → reference
facts from `context`.

`BASE_RULES` — channel etiquette, "do not invent facts", prompt-injection defence
— is the one piece of text this service adds on its own initiative. It is not a
persona and states no business policy, but a client that wants total control can
switch it off per prompt record with `metadata.base_rules: false`.

---

## Tools

Tools are thin adapters onto the gateway's REST API or a client application's own
endpoints. They hold no business logic — the decision always belongs to the
system on the other side.

Two gates decide whether a tool is offered on a given request:

1. **configured** — a tool whose dependency is blank is never advertised, so it
   can never be called and then fail;
2. **allowed** — the `tools_allowed` whitelist from the prompt record, or the
   per-request override.

| Tool | Needs | Calls | Purpose |
|---|---|---|---|
| `block_contact` | `GATEWAY_BASE_URL` | `POST /whatsapp/block` | Block an abusive/spamming contact mid-conversation |
| `unblock_contact` | `GATEWAY_BASE_URL` | `POST /whatsapp/unblock` | Lift a block |
| `lookup_business_info` | `KNOWLEDGE_API` | `GET …?q=` | Products, pricing, policies, hours, FAQ |
| `request_human_handoff` | `HANDOFF_API` | `POST` | Escalate to a human agent |
| `rag_search` | — | — | **Stub.** Always reports "not enabled" |

### Blocking, from the agent

This is how the model itself can decide mid-conversation that a contact should be
blocked and have it actually take effect. The path is deliberately long:

```
agent  ──►  gateway REST  ──►  the client server that owns the number
```

The agent never touches Baileys, never opens a WhatsApp session, and never calls
a client application's API directly. It asks the gateway; the gateway — which
owns WhatsApp and owns enforcement — forwards to the number's owner, which is
where the block is actually stored. `context.whatsapp_id` is required for these
two tools; without it they return an error to the model rather than guessing.

The contact defaults to the person in the current conversation, so the model does
not have to repeat a phone number back correctly to block the right person.

### RAG

The `rag_search` **tool** is still a stub, deliberately. A retrieval layer now
exists as a standalone HTTP API — see [RAG layer](#rag-layer-apirag) below — but
nothing in the agent calls it yet, so `rag_config` is still stored and not read,
and a client that lists `rag_search` in `tools_allowed` still gets a clear
`{"status": "not_enabled"}` rather than an empty result set the model might read
as "no matches". Wiring the tool to the new layer is a separate, deliberate
change to [app/services/tools.py](app/services/tools.py); until it is made, the
agent's behaviour is exactly what it was.

### Adding a tool

One `Tool(...)` entry in [app/services/tools.py](app/services/tools.py) — name,
description, JSON-schema parameters, an async handler, and an `available`
predicate. The dispatcher and the reply loop pick it up with no other changes.

### Behaviour

Every call logs the URL, payload, response status and any error body — same as
the gateway's connector. A tool failure is returned to the model as a tool result
(not raised), so it can recover mid-conversation instead of the whole reply
failing. When `request_human_handoff` succeeds, `handoff_requested: true` comes
back in the response so the caller can flag the conversation.

The tool loop is bounded by `MAX_TOOL_ITERATIONS`; on exhaustion the service
answers with what it has rather than looping.

Disable a group globally with `ENABLE_BLOCK_TOOLS=false`,
`ENABLE_KNOWLEDGE_TOOL=false` or `ENABLE_HANDOFF_TOOL=false`.

---

## Storage

Threads and prompt records are JSON files under `DATA_DIR`, one per record,
written atomically. No database, no ORM, no added dependency.

```
storage/
  threads/th_094a410f….json
  prompts/acme-app__support.json
```

Everything goes through `JsonStore` in
[app/store/base.py](app/store/base.py), so swapping in SQLite, Redis or Postgres
means writing one class — no caller changes. Ids are percent-escaped before they
become filenames, so a `client_id` of `../../etc` cannot escape the data
directory.

---

## Providers

Backends live in [app/services/llm/](app/services/llm/) behind a neutral
interface ([base.py](app/services/llm/base.py)). The agent loop never sees
provider-specific types, so adding another backend is one file plus one enum value.

### Groq (default)

| Env | Default | Notes |
|---|---|---|
| `GROQ_API_KEY` | — | From <https://console.groq.com/keys>. The only Groq setting that is configuration. |
| `GROQ_TEMPERATURE` | `0.3` | Low keeps replies consistent; raise for more varied marketing copy. |

Which Groq model runs comes from `text_llm` via the registry — see
[Choosing the model per request](#choosing-the-model-per-request--the-llm-registry).
If you use tools, point your identifiers at tool-capable models.
| `GROQ_REASONING_EFFORT` | *(empty)* | Only accepted by some models (e.g. `gpt-oss`). Omitted from the request when empty. |

Groq's API is OpenAI-compatible, so tools are sent as
`{"type": "function", ...}` and results return as `{"role": "tool", ...}`
messages. `/agent/classify` uses JSON mode with the schema in the prompt, then
validates against the Pydantic model.

Because most Groq models have no separate reasoning channel, the system prompt
automatically appends a final-answer-only instruction to keep reasoning out of
the customer's reply.

### Google Gemini (opt-in)

```ini
GEMINI_API_KEY=...          # GOOGLE_API_KEY is accepted as a synonym
GEMINI_TEMPERATURE=0.3
GEMINI_THINKING_BUDGET=128
```

Get a key at <https://aistudio.google.com/apikey>. Uses the `google-genai` SDK —
the one Google currently recommends, not the older `google-generativeai`.

No key is needed unless a `text_llm` actually resolves to Gemini. To route a
conversation there, point its identifier at the `gemini` provider in the
registry — nothing in `.env` selects a provider.

**`GEMINI_THINKING_BUDGET` matters more than it looks.** Gemini 3.x always
reasons before answering, and those tokens are charged against
`max_output_tokens`. Left uncapped, a 400-token limit gets spent thinking and the
customer receives a sentence cut in half — I measured exactly that. A budget of
128 fixes it. Set `-1` to let the model decide; `0` is rejected, since reasoning
cannot be switched off on Gemini 3.

Make sure `max_tokens` comfortably exceeds the budget — if it doesn't, the reply
comes back empty with `stop_reason: MAX_TOKENS`, and the service logs a warning
saying so.

Full capability parity with the other providers: system prompts, multi-turn
history, tool calling, JSON mode, temperature and token limits.

### Anthropic (opt-in)

Set `ANTHROPIC_API_KEY` and point an identifier at the `anthropic` provider in
the registry. `THINKING_MODE` (`adaptive` | `disabled`) and `EFFORT`
(`low`…`max`) then apply. `temperature` is not sent on this path — Opus 4.8
rejects it; tone is steered through the prompt instead.

### Shared

| Env | Default | Notes |
|---|---|---|
| `MAX_TOKENS` | `2000` | Ample for a WhatsApp-length reply. |
| `MAX_TOOL_ITERATIONS` | `4` | Loop bound; on exhaustion the service answers with what it has. |

---

## Error handling

| Condition | Result |
|---|---|
| `GROQ_API_KEY` not set | `503` with an explicit message, before any upstream call |
| No prompt resolvable | `422` — never a house default |
| `prompt_id` not registered by this client | `404` |
| `thread_id` owned by another client | `403` |
| Model refusal (Anthropic `stop_reason: refusal`) | `FALLBACK_REPLY` (or a safe default) is returned, `stop_reason: "refusal"` |
| Empty model output | Same fallback, logged with the stop reason |
| Rate limit | `429` to the caller |
| Provider unreachable | `503` |
| Provider 5xx | `502` |

The gateway logs a failed AI call upstream and sends nothing, rather than sending
a broken message.

---

## OpenAI-compatible surface

`POST /v1/chat/completions` lets OpenClaw use this service as a model provider.
OpenClaw cannot carry `client_id` or `prompt_id` through the OpenAI schema, so
they come from configuration: `OPENCLAW_CLIENT_ID`, plus **one of**
`OPENCLAW_PROMPT` (inline text) or `OPENCLAW_PROMPT_ID` (a registered record).

With neither set the endpoint returns `503`. That is deliberate: with the
built-in templates gone there is no default persona to answer with, and a blank
prompt is a configuration error rather than a silent fallback.

---

## RAG layer (`/api/rag`)

Document ingestion and vector search, added as a **separate module** in
[app/rag/](app/rag/). It is additive in the strict sense: no existing route,
prompt, tool, LLM call, thread or response shape changes because it exists, and
nothing in `app/services/` imports it. Deleting the four RAG lines from
[app/main.py](app/main.py) removes the feature completely.

```
PDF / DOCX / TXT
      ↓  document_loader.py     extract (per page for PDFs)
      ↓  text_cleaner.py        repair extraction artefacts
      ↓  chunker.py             boundary-aware chunks + overlap
      ↓  embeddings.py          BGE-M3, batched, loaded once
      ↓  vector_store.py        bulk COPY into Postgres
   pgvector
      ↓  vector_store.search    similarity ranked BY THE DATABASE
   top-k chunks  →  /search (raw) or /context (prompt-ready)
```

### This service had no database

Threads and prompt records are JSON files ([Storage](#storage)); there was no
Postgres connection, no ORM and no migration system to extend. So the RAG layer
brings its own: its own `RAG_DATABASE_URL`, its own asyncpg pool and its own
settings class ([app/rag/config.py](app/rag/config.py)), rather than pushing a
DSN into the configuration that every existing code path reads. **With
`RAG_DATABASE_URL` unset, `/api/rag/*` answers 503 and everything else runs
exactly as before.**

### Turning retrieval off

```bash
ENABLE_RAG_TOOL=false
```

The agent stops being offered `rag_search` on **both** `/agent/reply` and
`/api/v1/agent/chat` - one switch, not two - and answers from the prompt, the
conversation and the caller's context alone.

The `/api/rag/*` routes are unaffected: upload, list, delete and
`POST /api/rag/search` keep working, so a corpus can be ingested and tested
before the agent is allowed to use it, and switched off later without losing it.
To disable the whole RAG layer instead, leave `RAG_DATABASE_URL` unset.

The tool is offered only when `ENABLE_RAG_TOOL` is on **and** a DSN is
configured - advertising a tool whose backend answers 503 is worse than not
offering it. `GET /api/v1/agent/tools` shows what the model is actually given.

### Token budget

Per-call overhead is ~714 tokens (BASE_RULES + your prompt + tool rules + tool
definitions), and a retrieval turn adds at most `GRAPH_RAG_RESULT_MAX_CHARS`.

```bash
GRAPH_RAG_RESULT_MAX_CHARS=2600   # cap on what rag_search hands the model
```

Trimmed by whole passages, never mid-content, and the top-ranked passage always
survives however long it is. Retrieval is untouched - the same chunks are found
and the same scores reach the trace; this only bounds how many are quoted into
the conversation. `sources` reflects what the model actually saw.

Two further levers: send `"tools_allowed": ["rag_search"]` to drop tools a given
conversation will never use (~111 tokens for the block/unblock pair), and note
that `AGENT_TOOL_RULES` is sent on every provider call - a test fails if it
grows past 800 characters.

### Setup

```bash
pip install -r requirements.txt      # adds asyncpg, pypdf, python-docx, python-multipart

# 1. a Postgres with pgvector available
docker run -d --name pgvector -e POSTGRES_PASSWORD=pg -p 5432:5432 pgvector/pgvector:pg16

# 2. an embedding server. Nothing is loaded in this process; the model runs
#    where the GPU is. Two request dialects are supported - see below.

# 3. point the service at both, in .env
#      RAG_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres
#      EMBEDDING_API_URL=https://your-bge-m3-host/embed
#      EMBEDDING_API_FORMAT=texts
#      EMBEDDING_API_KEY_HEADER=X-API-Key
#      EMBEDDING_API_KEY=…

# 4. create the tables (idempotent; adds nothing but rag_documents/rag_chunks)
python -m app.rag.migrate            # --sql prints it instead of running it

# 5. start the service as usual
uvicorn app.main:app --reload
```

#### Which embedding server

The request body is the only thing that differs between servers, so it is the
only thing that has to be declared. Responses are auto-detected — `{"data":
[{"embedding": …}]}`, `{"vectors": …}`, `{"embeddings": …}` and a bare list of
lists are all read.

| Your server | Settings |
| --- | --- |
| OpenAI-compatible (TEI, Infinity, vLLM, llama.cpp, Ollama) | `EMBEDDING_API_BASE=http://host:8080/v1` — `/embeddings` is appended, key sent as `Authorization: Bearer …` |
| Its own path and shape, e.g. `POST /embed {"texts": […]}` | `EMBEDDING_API_URL=https://host/embed`, `EMBEDDING_API_FORMAT=texts`, `EMBEDDING_API_KEY_HEADER=X-API-Key` |

`EMBEDDING_API_KEY_HEADER` decides how the key is sent: `Authorization` gets a
`Bearer ` prefix, and any other header name carries the bare value — which is
what servers using a custom header expect, and prefixing it would make every
request a 401.

To run BGE-M3 **inside** this process instead, leave both URL settings empty,
`pip install sentence-transformers`, and set `EMBEDDING_BACKEND=local`. The model
is loaded once at startup and reused; it is never loaded per request.

`GET /api/rag/health` says whether the layer is usable and why not if it is not.
It reports names and numbers only - never a DSN, never a key.

### Endpoints

All of them sit behind the same `Authorization: Bearer $AGENT_API_TOKEN` as
`/agent/*`, and all appear in `/docs` with full request and response schemas.
A ready-made Postman collection is in [postman/](postman/) — import
`ai-agent-rag.postman_collection.json`, set `agent_api_token`, and run the
requests in order.

| Route | Purpose |
| --- | --- |
| `POST /api/rag/documents/upload` | Ingest a PDF, DOCX or TXT (multipart) |
| `POST /api/rag/search` | Vector similarity search, with scores |
| `POST /api/rag/context` | The same retrieval, assembled for a prompt |
| `GET /api/rag/documents` | List documents (paginated) |
| `GET /api/rag/documents/{id}` | One document record |
| `DELETE /api/rag/documents/{id}` | Delete a document, its chunks and embeddings |
| `GET /api/rag/health` | Configuration and readiness |

#### Upload

`file` is the only required field. `user_id`, `agent_id`, `conversation_id` and
`document_id` are optional - supply them only if you want to filter searches by
them later.

```bash
curl -X POST http://localhost:8000/api/rag/documents/upload \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -F "file=@company-policy.pdf" \
  -F "user_id=42" \
  -F 'metadata={"tenant":"acme"}'
```

```json
{
  "document_id": "doc_9f2c1b7e5a4d4c1fa1b2c3d4e5f60718",
  "filename": "company-policy.pdf",
  "file_type": "pdf",
  "chunks": 34,
  "characters": 21877,
  "pages": 12,
  "embedding_model": "BAAI/bge-m3",
  "embedding_dim": 1024,
  "took_ms": 2841,
  "metadata": {"tenant": "acme"}
}
```

`400` unsupported type · `413` over `RAG_MAX_UPLOAD_MB` · `422` empty or
unreadable · `502` embedding backend failed · `503` storage not configured.

#### Search

```bash
curl -X POST http://localhost:8000/api/rag/search \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'
```

```json
{
  "query": "What is the refund policy?",
  "found": true,
  "results": [
    {
      "chunk_id": "0f1c…",
      "document_id": "doc_9f2c…",
      "content": "Refunds are available within 30 days…",
      "similarity": 0.91,
      "metadata": {"source": "company-policy.pdf", "page": 4, "chunk_index": 12}
    }
  ],
  "threshold": 0.7,
  "top_k": 5,
  "took_ms": 38
}
```

When nothing reaches `RAG_SIMILARITY_THRESHOLD`:

```json
{"query": "Do you sell bicycles?", "found": false, "results": [], "threshold": 0.7, "top_k": 5, "took_ms": 31}
```

**No model is called in either case.** What to do about `found: false` is the
calling agent decision, and this endpoint does not make it.

#### Context

```bash
curl -X POST http://localhost:8000/api/rag/context \
  -H "Authorization: Bearer $AGENT_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the refund policy?", "top_k": 5}'
```

```json
{
  "query": "What is the refund policy?",
  "found": true,
  "context": "[1] company-policy.pdf, page 4\nRefunds are available within 30 days…",
  "sources": [
    {"document_id": "doc_9f2c…", "source": "company-policy.pdf", "page": 4, "chunk_index": 12, "similarity": 0.91}
  ],
  "chunks_used": 3,
  "took_ms": 41
}
```

Whole chunks only, capped at `RAG_CONTEXT_MAX_CHARS` - a chunk is dropped rather
than truncated, because half a sentence in a prompt reads as a fact that was
never written.

### How the existing agent can use this later

Nothing is wired today, on purpose. When it should be, there are two options and
both are small:

1. **Give `rag_search` a body.** `_rag_search` in
   [app/services/tools.py](app/services/tools.py) currently returns
   `{"status": "not_enabled"}`. Replacing that body with
   `await app.rag.service.search(query=…, agent_id=request.whatsapp_id)` - an
   in-process call, no HTTP - and flipping `available` on its registry entry
   makes retrieval a tool the model can choose. The stored `rag_config` on the
   prompt record is the natural place to read `top_k` and a threshold from.
2. **Retrieve before the model runs.** Call `POST /api/rag/context` from the
   caller (Laravel, the gateway) and pass the returned `context` string in the
   existing `context` field of `/agent/reply`. This needs no change to this
   service at all.

Option 2 changes nothing here; option 1 is a deliberate edit to the tool layer,
which this task did not make.

### Configuration

Every value is an environment variable; the RAG block in
[.env.example](.env.example) is the annotated list.

| Variable | Default | Notes |
| --- | --- | --- |
| `RAG_DATABASE_URL` | — | Falls back to `DATABASE_URL`. Unset ⇒ 503 on `/api/rag/*` only |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | |
| `EMBEDDING_BACKEND` | `auto` | `http` when `EMBEDDING_API_BASE` is set, else `local` |
| `EMBEDDING_API_BASE` | — | OpenAI-compatible base, e.g. `http://embeddings:8080/v1` |
| `EMBEDDING_API_URL` | — | Full endpoint, for a server with its own path. Wins over the base |
| `EMBEDDING_API_FORMAT` | `openai` | `openai` ⇒ `{"model", "input"}`; `texts` ⇒ `{"texts"}` |
| `EMBEDDING_API_KEY_HEADER` | `Authorization` | Any other name sends the bare key |
| `EMBEDDING_BATCH_SIZE` | `32` | Chunks per model call |
| `RAG_EMBEDDING_DIM` | `1024` | Checked against a live embedding; a mismatch is refused |
| `RAG_CHUNK_SIZE` | `800` | Characters |
| `RAG_CHUNK_OVERLAP` | `150` | Must be smaller than the size |
| `RAG_TOP_K` | `5` | `RAG_MAX_TOP_K` (50) caps what a request may ask for |
| `RAG_SIMILARITY_THRESHOLD` | `0.70` | Below it, `found: false`. **Measure it** — see below |
| `RAG_DISTANCE_METRIC` | `cosine` | Also sets the index opclass - re-migrate if changed |
| `RAG_MAX_UPLOAD_MB` | `25` | |

### Calibrating the threshold

`RAG_SIMILARITY_THRESHOLD` is the one setting that must be measured rather than
copied, because the scale depends on the model. BGE-M3 dense scores sit in a
narrow band. Measured against a live BGE-M3 for the query *"What is the refund
policy?"*:

```
0.595  "Refunds are available within 30 days of purchase…"   <- the answer
0.431  "Shipping is free on orders above 5000 PKR."
0.416  "The bicycle repair shop next door closes at noon…"
0.396  "Our office is open Monday to Friday, 9am to 5pm."
```

A correct match scores around 0.6, not 0.9, and unrelated text still scores
about 0.4. The `0.70` above is a placeholder default: left as it is with this
model, **every** search returns `found: false`. Around `0.50` separates the two
clusters. Embed a handful of real questions against your own documents and put
the threshold between the two groups before trusting it.

### Schema and indexing

[app/rag/migrations/001_rag_tables.sql](app/rag/migrations/001_rag_tables.sql)
creates `rag_documents` and `rag_chunks` (`embedding VECTOR(RAG_EMBEDDING_DIM)`),
plus `CREATE EXTENSION IF NOT EXISTS vector` - a no-op when pgvector is already
installed. Every statement is `IF NOT EXISTS`; nothing existing is dropped,
recreated or altered.

Similarity is computed **in Postgres**: `ORDER BY embedding <=> $1::vector LIMIT
$k`. Chunks are never pulled into Python to be compared. The migration builds an
HNSW index with the opclass matching `RAG_DISTANCE_METRIC` (IVFFlat where the
server predates HNSW). Both index types are *approximate*: they trade a little
recall for speed and only pay off past roughly ten thousand chunks - below that
an exact scan is both faster and perfectly recalled, which is why the index is
created but not depended on.

### Tests

```bash
pytest tests/test_rag.py -v
```

56 tests, fully offline: no Postgres, no embedding server, no model download.
They cover cleaning, chunking (size, overlap, metadata, runts), PDF/DOCX/TXT
extraction and rejection, batching, the dimension check, the threshold in both
directions, context assembly, and every endpoint status code.

### Not included

- **OCR for scanned PDFs.** The extension point exists -
  `document_loader.set_ocr_hook()` is called only for pages that produced no
  text, so installing a backend cannot affect the normal PDF path - but no OCR
  engine is installed.
- **Re-ingestion of an existing `document_id`.** Uploading the same id twice is
  a primary-key conflict; delete first.
- **Hybrid (keyword + vector) search and reranking.**

## LangChain agent (`/api/v1/agent`) and developer dashboard

An **additional** path to a reply, not a replacement. `POST /agent/reply` still
runs the tool loop in `app/services/agent.py` and is unchanged.

**There is no intent analyzer.** One LLM - the one Laravel chose - is given a
`rag_search` tool and decides for itself whether to answer directly or look
something up. LangGraph is not used and is not imported.

```
POST /api/v1/agent/chat        # same payload as /agent/reply, + run_id in the response
GET  /api/v1/agent/runs        # recent runs
GET  /api/v1/agent/runs/{id}   # one run: steps, tool calls, LLM calls
GET  /api/v1/agent/tools       # the tools the model is offered, with descriptions

GET  /api/v1/dashboard/ui      # the dashboard page
GET  /api/v1/dashboard/overview|runs|runs/{id}|rag|tools|models|logs|status
```

### The flow

```
user message
     |
     v
ServiceChatModel.bind_tools([rag_search, ...])      <- Laravel's text_llm
     |
 +---+---------------------------+
 |                               |
 v                               v
answers directly          emits a tool call
(1 LLM call, 0 RAG)             |
                                v
                   existing rag/service.search()
                                v
                   ToolMessage -> the SAME model
                                v
                          final answer
```

Cost: a greeting is **one** LLM call. A knowledge question is **two** - decide,
then answer - and the second only happens because the model asked for a lookup.

### Why `langchain-core` only

Measured rather than assumed:

```
pip install --dry-run langchain langchain-groq
-> Would install groq-0.37.1 langchain-1.3.15 langchain-core-1.5.6
                 langchain-groq-1.1.3 langgraph-1.2.11
```

`langchain` depends on `langgraph`, and `langchain-groq` would replace the
pinned `groq` 0.15.0 that `/agent/reply` runs on. So `app/graph/chat_model.py`
adapts the **existing** `services/llm` providers to LangChain's `BaseChatModel`
instead. That keeps their bounded 45s timeouts, Groq reasoning-effort, Gemini
thinking-budget, refusal handling and `require_model()` guard - all of which a
stock integration would discard.

`AgentExecutor` is not used because it is not in `langchain-core`; the loop in
`app/graph/agent.py` is the tool-calling contract it implements.

### What it reuses

| Concern | Owner - unchanged |
|---|---|
| Which model runs | `services/models.resolve(text_llm)` - Laravel's choice |
| Providers | `services/llm/*`, via the `ServiceChatModel` adapter |
| System prompt | `services/prompts.build_system_prompt` - the client's own |
| Retrieval | `rag/service.search(...)`, called in process |
| Other tools | `services/tools.REGISTRY`, schemas passed through verbatim |
| Conversations | `store/threads.py`, `THREAD_MAX_TURNS` |
| Database | the pool `rag/vector_store.py` already owns |
| Auth | `security.require_token` |

**No model id appears anywhere in `app/graph/`** - a test asserts it. Change
`text_llm` in Laravel and both calls of a run move to the new model.

### Setup

```bash
pip install -r requirements.txt
python -m app.graph.migrate          # trace tables; idempotent and additive
```

Both optional. Without `langchain-core` the routes are not mounted and every
existing endpoint is unaffected. Without a database the agent still runs - runs
are logged instead of stored, and the dashboard says so.

### dm_ai_logs

The response carries `trace` and `usage` in the same shape `/agent/reply`
returns, so an existing `dm_ai_logs` writer needs no change. `usage` is the
**run** total across every provider call, not just the last one.

The package directory is still named `graph` because `python -m app.graph.migrate`,
the dashboard URLs and the deployed trace tables all live under that name.
