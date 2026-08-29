"""LangChain agent, RAG tool and developer observability.

The model decides
-----------------
There is NO intent analyzer. `agent.py` binds `rag_search` to the chat model
Laravel chose and lets that one model decide whether to answer directly or look
something up. A greeting costs one LLM call; a knowledge question costs two.

The package is still named `graph` for one reason: `python -m app.graph.migrate`,
the dashboard URLs and the trace tables are already deployed under that name,
and renaming them would break a running system for no behavioural gain.
LangGraph is NOT used and is not imported anywhere in it.

Original notes follow.


Added the same way app/rag/ was: as a package that the rest of the service does
not import. The wiring is two `include_router` lines and one lifespan call in
app/main.py. Remove them and every existing endpoint behaves exactly as before.

What this package OWNS
----------------------
The graph (state, nodes, routing), the intent-analysis prompt, the run/step
trace tables, and the developer dashboard.

What this package REUSES and must never reimplement
---------------------------------------------------
    model choice      services/models.resolve(text_llm)      <- Laravel's choice
    LLM calls         services/llm.get_provider(...)
    answer generation services/agent.generate_reply(...)
    retrieval         rag/service.search(...)   (in-process, not over HTTP)
    tools             services/tools.execute(...)
    handoff           the existing request_human_handoff tool -> HANDOFF_API
    conversations     store/threads.py
    database          the pool rag/vector_store.py already owns
    auth              security.require_token

Degradation
-----------
Observability writes are best-effort. With no RAG_DATABASE_URL the graph still
runs end to end and the trace is logged rather than persisted; the dashboard
then reports that storage is unconfigured instead of failing. A chat request is
never lost because a trace could not be written.
"""
