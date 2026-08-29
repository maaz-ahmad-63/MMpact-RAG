"""Configuration for the graph layer only.

A separate Settings class, for the same reason app/rag/config.py is one:
app/config.py is read by the agent, the providers and the stores, and adding
fields there changes what all of them see. Both read the same .env, so an
operator still configures one file.

Nothing here chooses a model. Which model runs is Laravel's `text_llm`, resolved
by services/models.py, and no setting in this file can override that.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class GraphSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # The tool-iteration ceiling is MAX_TOOL_ITERATIONS from app/config.py -
    # the same setting /agent/reply already honours. It is not duplicated here.
    #
    # GRAPH_INTENT_HISTORY_TURNS, GRAPH_PROMPT_KNOWLEDGE,
    # GRAPH_KNOWLEDGE_MAX_CHARS and GRAPH_MAX_REWRITES were read by the intent
    # analyzer and the rewrite cycle. Both are gone, so the settings are gone
    # with them; `extra="ignore"` means a .env that still lists them is fine.
    #
    # -- conversation history ----------------------------------------------
    #
    # How many prior exchanges the model is given: GRAPH_HISTORY_PAIRS=3 means
    # the last 3 user turns and the last 3 assistant turns.
    #
    # Counted PER ROLE rather than as a flat "last 6 turns", because a flat
    # slice is not what "3 and 3" means on a chat channel. A customer who sends
    # "hi", "are you there?", "hello??" produces three consecutive user turns,
    # and a last-6 window would then hand the model 4 user turns and 2 assistant
    # ones - dropping the assistant reply that gave the conversation its
    # meaning. Keeping each role's own last N is stable under that.
    #
    # 0 disables this cap and the model gets whatever the source supplies.
    #
    # Distinct from THREAD_MAX_TURNS, which is a RETENTION limit applied when a
    # thread is written to disk. This one applies at the agent boundary, so it
    # bounds inline `context.history` from Laravel too - which THREAD_MAX_TURNS
    # cannot, because those turns were never stored here.
    graph_history_pairs: int = 3

    # -- token budget ------------------------------------------------------
    #
    # Cap on the text `rag_search` hands back to the model. This is the biggest
    # VARIABLE cost in a run: RAG_TOP_K x RAG_CHUNK_SIZE is 5 x 800 = 4000
    # characters by default, roughly 1000 tokens, and it is carried on the
    # answer call on top of the system prompt and the tool definitions.
    #
    # Distinct from RAG_CONTEXT_MAX_CHARS, which bounds POST /api/rag/context
    # for callers assembling their own prompt. Retrieval itself is untouched -
    # the same chunks are found and the same scores are recorded on the trace;
    # this only bounds how many of them are quoted into the conversation.
    #
    # Trimmed by WHOLE chunks: half a passage is a passage the model may quote
    # as if it were complete.
    graph_rag_result_max_chars: int = 2600


@lru_cache
def get_graph_settings() -> GraphSettings:
    return GraphSettings()


graph_settings = get_graph_settings()
