"""The LangChain agent: the model decides whether to search.

Runs offline. The LLM is a scripted fake plugged into the real
`services.llm.get_provider` cache, so the production resolver and the real
ServiceChatModel translation stay in the code path; retrieval is a stub over the
existing `rag.service.search`; trace storage reports itself unconfigured.

    pytest tests/test_agent_langchain.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings as app_settings  # noqa: E402
from app.graph import agent as lc_agent  # noqa: E402
from app.graph import store  # noqa: E402
from app.graph.schemas import ChatRequest  # noqa: E402
from app.services import llm, models  # noqa: E402
from app.services.llm.base import LLMProvider, LLMResult, ToolCall, Usage  # noqa: E402

pytestmark = pytest.mark.asyncio


class ScriptedProvider(LLMProvider):
    """Returns a queued turn per call, and records every model id it was given."""

    name = "groq"

    def __init__(self) -> None:
        #: Each entry is either a string (the final answer) or a list of
        #: (tool_name, arguments) the model "decides" to call.
        self.turns: list[Any] = ["Here is the answer."]
        self.models_used: list[str] = []
        self.tools_seen: list[list[str]] = []
        self.systems_seen: list[str] = []
        self.messages_seen: list[list[dict[str, Any]]] = []
        self.fail = False
        self._call = 0

    async def complete(self, *, system, messages, tools=None, max_tokens=None, model=None):
        model_id = self.require_model(model)
        self.models_used.append(model_id)
        self.tools_seen.append([entry["name"] for entry in (tools or [])])
        self.systems_seen.append(system)
        self.messages_seen.append(list(messages))
        if self.fail:
            raise RuntimeError("provider exploded")

        turn = self.turns[min(self._call, len(self.turns) - 1)]
        self._call += 1

        if isinstance(turn, list):
            calls = [
                ToolCall(id=f"call_{index}", name=name, arguments=arguments)
                for index, (name, arguments) in enumerate(turn)
            ]
            return LLMResult(
                text="", tool_calls=calls, model=model_id, stop_reason="tool_calls",
                assistant_message={"role": "assistant", "tool_calls": "native-blob"},
                usage=Usage(input_tokens=100, output_tokens=20),
            )
        return LLMResult(
            text=turn, model=model_id, stop_reason="stop",
            usage=Usage(input_tokens=150, output_tokens=30),
        )

    def tool_result_messages(self, outcomes):
        return [
            {"role": "tool", "tool_call_id": outcome.call.id, "content": outcome.content}
            for outcome in outcomes
        ]

    async def complete_json(self, *, system, user, schema, schema_name="response",
                            max_tokens=1024, model=None):  # pragma: no cover
        raise AssertionError(
            "complete_json was called - that was the intent analyzer's entry point "
            "and must never happen on this path"
        )

    @property
    def calls(self) -> int:
        return self._call


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A registry for Laravel's `text_llm` to resolve against."""
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps({
            "default": "groq_model_a",
            "providers": {"groq": {"key_env": "groq_api_key",
                                   "fallback": "groq_model_a", "prefixes": ["groq"]}},
            "models": {"groq_model_a": {"provider": "groq", "model": "model-A"},
                       "groq_model_b": {"provider": "groq", "model": "model-B"}},
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_REGISTRY_PATH", str(path))
    models.reload()
    yield
    monkeypatch.delenv("LLM_REGISTRY_PATH", raising=False)
    models.reload()


@pytest.fixture
def provider(monkeypatch, registry):
    fake = ScriptedProvider()
    monkeypatch.setattr(llm, "_providers", {"groq": fake})
    return fake


@pytest.fixture(autouse=True)
def no_trace_storage(monkeypatch):
    monkeypatch.setattr(store, "storage_configured", lambda: False)


@pytest.fixture
def retrieval(monkeypatch):
    """Stub over the EXISTING rag.service.search."""
    from app.rag import service as rag_service

    calls: list[str] = []
    plan: dict[str, Any] = {"results": [], "error": None}

    async def fake_search(*, query, top_k=None, threshold=None, **kwargs):
        calls.append(query)
        if plan["error"] is not None:
            raise plan["error"]
        return {"query": query, "found": bool(plan["results"]), "results": plan["results"],
                "threshold": 0.5, "top_k": 5, "took_ms": 3}

    monkeypatch.setattr(rag_service, "search", fake_search)

    # Pin BOTH availability gates. A test must not depend on the developer's
    # .env - ENABLE_RAG_TOOL=false there would otherwise silently turn most of
    # this file into a test of the disabled path.
    monkeypatch.setattr("app.rag.config.rag_settings.rag_database_url",
                        "postgresql://stub/stub")
    monkeypatch.setattr(app_settings, "enable_rag_tool", True)
    return {"calls": calls, "plan": plan}


def chunk(text="Cancellations are free within 24 hours.", score=0.88, index=0):
    return {"chunk_id": f"c-{index}", "document_id": "doc-1", "content": text,
            "similarity": score,
            "metadata": {"source": "policy.pdf", "page": 4, "chunk_index": index}}


def make_request(message: str, model: str | None = "groq_model_a", **extra) -> ChatRequest:
    payload = {
        "client_id": "laravel-prod",
        "system_prompt": "You are a support agent for Acme.",
        "message": message, "model": model, "persist": False,
        "context": {"account_name": "Acme"},
    }
    payload.update(extra)
    return ChatRequest.model_validate(payload)


def tool_text(sent: list[dict[str, Any]]) -> str:
    return "\n".join(str(m["content"]) for m in sent if m.get("role") == "tool")


# -- Test 1 & 2: no tool for conversation or general knowledge --------------


async def test_greeting_costs_one_llm_call_and_no_rag(provider, retrieval):
    provider.turns = ["Hello! How can I help?"]

    response = await lc_agent.run_chat(make_request("Hello"))

    assert response.reply == "Hello! How can I help?"
    assert provider.calls == 1, "a greeting must cost exactly one LLM call"
    assert retrieval["calls"] == []
    assert response.action == "direct_answer"
    assert response.requires_rag is False


async def test_general_question_answers_directly(provider, retrieval):
    provider.turns = ["30."]

    response = await lc_agent.run_chat(make_request("What is 10 + 20?"))

    assert response.reply == "30."
    assert provider.calls == 1
    assert retrieval["calls"] == []
    assert response.action == "direct_answer"


# -- Test 3: knowledge question -> tool -> answer ---------------------------


async def test_knowledge_question_calls_rag_then_answers(provider, retrieval):
    provider.turns = [
        [("rag_search", {"query": "cancellation policy"})],
        "You can cancel free of charge within 24 hours.",
    ]
    retrieval["plan"]["results"] = [chunk()]

    response = await lc_agent.run_chat(make_request("What is the cancellation policy?"))

    assert retrieval["calls"] == ["cancellation policy"]
    assert provider.calls == 2, "one decision call, one answer call - no more"
    assert response.action == "rag_search"
    assert response.requires_rag is True
    assert response.reply.startswith("You can cancel")
    assert response.search_query == "cancellation policy"


async def test_retrieved_text_reaches_the_model_as_source_blocks(provider, retrieval):
    provider.turns = [[("rag_search", {"query": "cancellation policy"})], "Answer."]
    retrieval["plan"]["results"] = [chunk(), chunk("Late cancellations are charged.", index=1)]

    await lc_agent.run_chat(make_request("cancellation policy?"))

    body = tool_text(provider.messages_seen[1])
    assert "SOURCE 1" in body and "SOURCE 2" in body
    assert "Title: policy.pdf (page 4)" in body
    assert "Cancellations are free within 24 hours." in body


async def test_sources_are_returned_without_being_parsed_back_out(provider, retrieval):
    provider.turns = [[("rag_search", {"query": "q"})], "Answer."]
    retrieval["plan"]["results"] = [chunk()]

    response = await lc_agent.run_chat(make_request("cancellation policy?"))

    assert response.sources == [
        {"document_id": "doc-1", "chunk_id": "c-0", "source": "policy.pdf",
         "page": 4, "score": 0.88}
    ]


# -- Test 4: nothing found --------------------------------------------------


async def test_empty_retrieval_tells_the_model_not_to_guess(provider, retrieval):
    provider.turns = [
        [("rag_search", {"query": "XYZ policy"})],
        "I do not have information on that.",
    ]
    retrieval["plan"]["results"] = []

    response = await lc_agent.run_chat(make_request("What is the XYZ policy?"))

    body = tool_text(provider.messages_seen[1])
    assert "No documents in the knowledge base matched" in body
    assert "do not guess" in body.lower()
    assert response.status == "completed"
    assert response.sources == []


# -- Test 5: follow-up ------------------------------------------------------


async def test_follow_up_sees_prior_turns(provider, retrieval):
    provider.turns = [[("rag_search", {"query": "cancellation policy after 24 hours"})],
                      "After 24 hours a fee applies."]
    retrieval["plan"]["results"] = [chunk()]

    response = await lc_agent.run_chat(
        make_request(
            "What about after 24 hours?",
            context={
                "account_name": "Acme",
                "history": [
                    {"role": "user", "content": "What is the cancellation policy?"},
                    {"role": "assistant", "content": "Cancellations are free within 24h."},
                ],
            },
        )
    )

    sent = provider.messages_seen[0]
    assert [m["role"] for m in sent] == ["user", "assistant", "user"]
    assert sent[-1]["content"] == "What about after 24 hours?"
    assert response.reply.startswith("After 24 hours")


async def test_history_is_normalised_to_open_with_a_user_turn(provider, retrieval):
    provider.turns = ["ok"]

    await lc_agent.run_chat(
        make_request(
            "hi",
            context={"history": [
                {"role": "assistant", "content": "leading assistant turn"},
                {"role": "user", "content": "real first turn"},
            ]},
        )
    )

    assert provider.messages_seen[0][0]["role"] == "user"
    assert provider.messages_seen[0][0]["content"] == "real first turn"


# -- Test 6: tool failure ---------------------------------------------------


async def test_rag_failure_is_reported_to_the_model_not_raised(provider, retrieval):
    provider.turns = [[("rag_search", {"query": "q"})],
                      "I cannot look that up right now."]
    retrieval["plan"]["error"] = RuntimeError("pgvector is down")

    response = await lc_agent.run_chat(make_request("What is the policy?"))

    body = tool_text(provider.messages_seen[1])
    assert "temporarily unavailable" in body
    assert "pgvector" not in body, "the technical error must stay server-side"
    assert response.status == "completed"
    assert response.reply


async def test_llm_failure_returns_the_fallback_not_a_crash(provider, retrieval):
    provider.fail = True

    response = await lc_agent.run_chat(make_request("hello"))

    assert response.status == "failed"
    assert "provider exploded" in (response.error or "")
    assert response.reply


async def test_unknown_tool_name_is_told_to_the_model(provider, retrieval):
    provider.turns = [[("not_a_real_tool", {})], "Sorry, I cannot do that."]

    response = await lc_agent.run_chat(make_request("do something"))

    assert "Unknown tool" in tool_text(provider.messages_seen[1])
    assert response.status == "completed"


# -- the intent analyzer is gone -------------------------------------------


async def test_no_structured_intent_call_is_ever_made(provider, retrieval):
    """complete_json asserts if called - it was the intent analyzer's entry point."""
    provider.turns = [[("rag_search", {"query": "q"})], "Answer."]
    retrieval["plan"]["results"] = [chunk()]

    response = await lc_agent.run_chat(make_request("policy?"))

    assert response.status == "completed"
    assert response.intent is None, "nothing may classify intent any more"


async def test_no_intent_analyzer_code_remains():
    package = Path(__file__).resolve().parents[1] / "app" / "graph"
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "IntentResult" not in text, f"{path.name} still references IntentResult"
        assert "intent_analyzer" not in text, f"{path.name} references intent_analyzer"
    for gone in ("builder.py", "nodes.py", "state.py", "knowledge.py", "runner.py"):
        assert not (package / gone).exists(), f"{gone} still exists"


async def test_langgraph_is_not_imported_anywhere():
    root = Path(__file__).resolve().parents[1] / "app"
    for path in root.rglob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "langgraph" not in stripped, f"{path.name}: {stripped}"


# -- dynamic model ----------------------------------------------------------


async def test_the_model_comes_from_laravel_and_nothing_else(provider, retrieval):
    """MANDATORY: model-A then model-B, with no Python changed between them."""
    provider.turns = ["hi"]

    first = await lc_agent.run_chat(make_request("hello", model="groq_model_a"))
    assert first.model == "model-A"
    assert set(provider.models_used) == {"model-A"}

    provider.models_used.clear()
    provider._call = 0

    second = await lc_agent.run_chat(make_request("hello", model="groq_model_b"))
    assert second.model == "model-B"
    assert set(provider.models_used) == {"model-B"}


async def test_both_calls_of_a_rag_run_use_the_same_laravel_model(provider, retrieval):
    provider.turns = [[("rag_search", {"query": "q"})], "Answer."]
    retrieval["plan"]["results"] = [chunk()]

    await lc_agent.run_chat(make_request("policy?", model="groq_model_b"))

    assert provider.models_used == ["model-B", "model-B"]


async def test_no_model_id_is_hardcoded_in_the_package():
    package = Path(__file__).resolve().parents[1] / "app" / "graph"
    offenders = []
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for needle in ("gpt-oss", "llama-3", "gpt-4", "gpt-5", "claude-3", "gemini-1.5"):
            if needle in text:
                offenders.append(f"{path.name}: {needle}")
    assert not offenders, f"hardcoded model ids: {offenders}"


# -- tools and prompt -------------------------------------------------------


async def test_rag_tool_is_offered_with_its_decision_rules(provider, retrieval):
    provider.turns = ["hi"]

    await lc_agent.run_chat(make_request("hello"))

    assert "rag_search" in provider.tools_seen[0]
    system = provider.systems_seen[0]
    assert "Only call `rag_search` when" in system
    assert "Never search for greetings" in system
    # The client's own prompt is still the persona; this package adds no policy.
    assert "You are a support agent for Acme." in system


async def test_tools_allowed_empty_means_no_tools(provider, retrieval):
    provider.turns = ["hi"]

    await lc_agent.run_chat(make_request("hello", tools_allowed=[]))

    assert provider.tools_seen[0] == []


async def test_rag_tool_absent_when_retrieval_is_unconfigured(provider, monkeypatch):
    monkeypatch.setattr("app.rag.config.rag_settings.rag_database_url", "")
    monkeypatch.setattr("app.rag.config.rag_settings.database_url", "")
    provider.turns = ["hi"]

    await lc_agent.run_chat(make_request("hello"))

    assert "rag_search" not in provider.tools_seen[0]


async def test_tool_iterations_are_capped(provider, retrieval, monkeypatch):
    """A model that only ever calls tools must still produce a reply."""
    monkeypatch.setattr(app_settings, "max_tool_iterations", 2)
    provider.turns = [[("rag_search", {"query": "q"})]]  # repeats forever
    retrieval["plan"]["results"] = [chunk()]

    response = await lc_agent.run_chat(make_request("policy?"))

    assert provider.calls <= 3
    assert response.status == "completed"
    assert response.reply


# -- observability and response contract ------------------------------------


async def test_usage_sums_every_call_in_the_run(provider, retrieval):
    provider.turns = [[("rag_search", {"query": "q"})], "Answer."]
    retrieval["plan"]["results"] = [chunk()]

    response = await lc_agent.run_chat(make_request("policy?"))

    assert response.usage.input_tokens == 250   # 100 + 150
    assert response.usage.output_tokens == 50   # 20 + 30
    assert response.trace.provider_calls == 2
    assert [entry["node"] for entry in response.usage_by_node] == ["agent_llm", "agent_llm"]


async def test_trace_carries_the_prompt_that_was_sent(provider, retrieval):
    provider.turns = ["hi"]

    response = await lc_agent.run_chat(make_request("Hello there"))

    assert "You are a support agent for Acme." in response.trace.system_prompt
    assert response.trace.user_prompt == "Hello there"
    assert response.trace.all_prompt_text
    assert response.trace.model_resolved == "model-A"


async def test_response_contract_is_unchanged(provider, retrieval):
    provider.turns = ["hi"]

    body = (await lc_agent.run_chat(make_request("hello"))).model_dump()

    for field in ("run_id", "reply", "conversation_id", "thread_id", "intent", "action",
                  "requires_rag", "provider", "model", "sources", "tool_calls",
                  "handoff_requested", "usage", "usage_by_node", "trace",
                  "latency_ms", "status", "error"):
        assert field in body, f"{field} disappeared from the response"


async def test_existing_reply_endpoint_still_works(provider, retrieval, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(app_settings, "agent_api_token", "test-token")
    client = TestClient(app)
    provider.turns = ["Existing path reply."]

    response = client.post(
        "/agent/reply",
        headers={"Authorization": "Bearer test-token"},
        json={"client_id": "laravel-prod", "system_prompt": "You are a support agent.",
              "message": "Hello", "model": "groq_model_a", "persist": False},
    )

    assert response.status_code == 200
    assert response.json()["reply"] == "Existing path reply."


async def test_chat_endpoint_still_authenticated(provider, retrieval, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(app_settings, "agent_api_token", "test-token")
    client = TestClient(app)

    assert client.post("/api/v1/agent/chat", json={"message": "x"}).status_code == 401

    provider.turns = ["hi"]
    ok = client.post(
        "/api/v1/agent/chat",
        headers={"Authorization": "Bearer test-token"},
        json={"client_id": "c", "system_prompt": "p", "message": "Hello",
              "model": "groq_model_a", "persist": False},
    )
    assert ok.status_code == 200
    assert ok.json()["reply"] == "hi"


# -- token budget -----------------------------------------------------------


async def test_rag_result_is_capped_and_trimmed_by_whole_passages(provider, retrieval,
                                                                  monkeypatch):
    """The tool result is the biggest variable cost in a run."""
    from app.graph.config import graph_settings

    monkeypatch.setattr(graph_settings, "graph_rag_result_max_chars", 400)
    long = "x" * 300
    retrieval["plan"]["results"] = [chunk(long, index=index) for index in range(5)]
    provider.turns = [[("rag_search", {"query": "q"})], "Answer."]

    response = await lc_agent.run_chat(make_request("policy?"))

    body = tool_text(provider.messages_seen[1])
    assert len(body) <= 800, "the budget was not applied"
    # Whole passages only: never a truncated one.
    assert body.count("SOURCE") == body.count("Content:")
    for kept in body.split("SOURCE ")[1:]:
        assert long in kept, "a passage was cut mid-content"
    # Sources reflect what the model actually saw, not what was retrieved.
    assert len(response.sources) == body.count("SOURCE ")
    assert len(response.sources) < 5


async def test_the_best_passage_survives_any_budget(provider, retrieval, monkeypatch):
    """A budget must never turn a good match into an empty result."""
    from app.graph.config import graph_settings

    monkeypatch.setattr(graph_settings, "graph_rag_result_max_chars", 10)
    retrieval["plan"]["results"] = [chunk("y" * 2000)]
    provider.turns = [[("rag_search", {"query": "q"})], "Answer."]

    response = await lc_agent.run_chat(make_request("policy?"))

    assert "SOURCE 1" in tool_text(provider.messages_seen[1])
    assert len(response.sources) == 1


async def test_the_tool_rules_stay_small(provider, retrieval):
    """A regression guard on the prompt this package prepends to every call.

    It is sent on both calls of a RAG turn, so growth here is paid twice per
    answered question. If a rule genuinely needs adding, raise this number
    deliberately rather than letting it drift.
    """
    from app.graph.prompts import AGENT_TOOL_RULES

    assert len(AGENT_TOOL_RULES) < 800, (
        f"AGENT_TOOL_RULES is {len(AGENT_TOOL_RULES)} chars "
        f"(~{len(AGENT_TOOL_RULES) // 4} tokens) and sent on every provider call"
    )


# -- ENABLE_RAG_TOOL --------------------------------------------------------


async def test_rag_tool_is_withheld_when_disabled_by_env(provider, retrieval, monkeypatch):
    """ENABLE_RAG_TOOL=false: the model is never offered the search tool."""
    monkeypatch.setattr(app_settings, "enable_rag_tool", False)
    provider.turns = ["Hello!"]

    response = await lc_agent.run_chat(make_request("What is the cancellation policy?"))

    assert "rag_search" not in provider.tools_seen[0]
    assert retrieval["calls"] == []
    assert response.status == "completed"
    assert response.action == "direct_answer"


async def test_disabling_rag_leaves_the_other_tools_alone(provider, retrieval, monkeypatch):
    """One switch, scoped to one tool."""
    monkeypatch.setattr(app_settings, "enable_rag_tool", False)
    monkeypatch.setattr(app_settings, "gateway_base_url", "https://gateway.example")
    monkeypatch.setattr(app_settings, "enable_block_tools", True)
    provider.turns = ["hi"]

    await lc_agent.run_chat(make_request("hello"))

    offered = provider.tools_seen[0]
    assert "rag_search" not in offered
    assert "block_contact" in offered


async def test_the_same_switch_governs_the_reply_path(monkeypatch):
    """ENABLE_RAG_TOOL is one flag for both agents, not two."""
    from app.services import tools as tool_registry

    monkeypatch.setattr("app.rag.config.rag_settings.rag_database_url",
                        "postgresql://stub/stub")
    monkeypatch.setattr(app_settings, "enable_rag_tool", True)
    assert "rag_search" in tool_registry.available_tools()

    monkeypatch.setattr(app_settings, "enable_rag_tool", False)
    assert "rag_search" not in tool_registry.available_tools()


async def test_rag_routes_keep_working_when_the_tool_is_disabled(monkeypatch):
    """Disabling the TOOL must not disable ingestion or POST /api/rag/search."""
    from fastapi.testclient import TestClient

    from app.main import app

    monkeypatch.setattr(app_settings, "enable_rag_tool", False)
    monkeypatch.setattr(app_settings, "agent_api_token", "test-token")
    paths = {route.path for route in app.routes if getattr(route, "methods", None)}

    assert "/api/rag/search" in paths
    assert "/api/rag/documents/upload" in paths

    client = TestClient(app)
    # Still mounted and still authenticated - not removed.
    assert client.post("/api/rag/search", json={"query": "x"}).status_code == 401


# -- GRAPH_HISTORY_PAIRS ----------------------------------------------------


async def test_only_the_last_three_pairs_reach_the_model(provider, retrieval, monkeypatch):
    from app.graph.config import graph_settings

    monkeypatch.setattr(graph_settings, "graph_history_pairs", 3)
    provider.turns = ["ok"]
    history = []
    for index in range(10):
        history.append({"role": "user", "content": f"u{index}"})
        history.append({"role": "assistant", "content": f"a{index}"})

    await lc_agent.run_chat(make_request("now?", context={"history": history}))

    sent = provider.messages_seen[0]
    prior = sent[:-1]  # the new message is the last turn
    assert [m["content"] for m in prior] == ["u7", "a7", "u8", "a8", "u9", "a9"]
    assert sum(1 for m in prior if m["role"] == "user") == 3
    assert sum(1 for m in prior if m["role"] == "assistant") == 3


async def test_the_cap_counts_each_role_separately(provider, retrieval, monkeypatch):
    """A burst of user messages must not evict the assistant's replies.

    A flat 'last 6 turns' would hand the model 6 user turns here and no
    assistant reply at all - losing the answer the follow-up refers to.
    """
    from app.graph.config import graph_settings

    monkeypatch.setattr(graph_settings, "graph_history_pairs", 3)
    provider.turns = ["ok"]
    history = [
        {"role": "user", "content": "what is the price?"},
        {"role": "assistant", "content": "The Extended licence is listed."},
        {"role": "user", "content": "hi"},
        {"role": "user", "content": "are you there?"},
        {"role": "user", "content": "hello??"},
        {"role": "user", "content": "anyone?"},
    ]

    await lc_agent.run_chat(make_request("well?", context={"history": history}))

    prior = provider.messages_seen[0][:-1]
    # The assistant's answer survives - it is what the follow-up refers to.
    assert any(m["role"] == "assistant" for m in prior), "the reply was evicted"
    assert "The Extended licence is listed." in [m["content"] for m in prior]
    # It is preceded by the question it answered: the window is anchored to a
    # user turn, so at most one extra user turn is carried. Never more.
    assert prior[0]["role"] == "user"
    assert prior[0]["content"] == "what is the price?"
    assert sum(1 for m in prior if m["role"] == "user") <= 3 + 1


async def test_the_cap_applies_to_inline_history_too(provider, retrieval, monkeypatch):
    """THREAD_MAX_TURNS cannot bound turns that were never stored here."""
    from app.graph.config import graph_settings

    monkeypatch.setattr(graph_settings, "graph_history_pairs", 2)
    provider.turns = ["ok"]
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"t{index}"}
        for index in range(40)
    ]

    await lc_agent.run_chat(make_request("q", context={"history": history}))

    prior = provider.messages_seen[0][:-1]
    assert len(prior) == 4, "inline history was not capped"


async def test_zero_disables_the_cap(provider, retrieval, monkeypatch):
    from app.graph.config import graph_settings

    monkeypatch.setattr(graph_settings, "graph_history_pairs", 0)
    provider.turns = ["ok"]
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"t{index}"}
        for index in range(12)
    ]

    await lc_agent.run_chat(make_request("q", context={"history": history}))

    assert len(provider.messages_seen[0][:-1]) == 12


async def test_order_is_preserved_after_trimming(provider, retrieval, monkeypatch):
    from app.graph.config import graph_settings

    monkeypatch.setattr(graph_settings, "graph_history_pairs", 2)
    provider.turns = ["ok"]
    history = [
        {"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"}, {"role": "assistant", "content": "a3"},
    ]

    await lc_agent.run_chat(make_request("q", context={"history": history}))

    prior = [m["content"] for m in provider.messages_seen[0][:-1]]
    assert prior == ["u2", "a2", "u3", "a3"], "the conversation must read forwards"


async def test_the_anchor_costs_at_most_one_extra_turn(provider, retrieval, monkeypatch):
    """The anchor must not become an unbounded backwards walk."""
    from app.graph.config import graph_settings
    from app.graph.agent import _recent
    from app.schemas import HistoryTurn

    monkeypatch.setattr(graph_settings, "graph_history_pairs", 2)
    turns = [HistoryTurn(role="user", content=f"u{i}") for i in range(20)]
    turns.insert(10, HistoryTurn(role="assistant", content="the answer"))

    kept = _recent(turns, 2)

    # 2 users + 1 assistant available + at most 1 anchor.
    assert len(kept) <= 2 + 2 + 1
    assert kept[0].role == "user"
