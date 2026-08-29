"""Registry resolution, provider selection, Gemini translation and error handling.

Runs offline: the one test that exercises a real Gemini call is skipped unless
GEMINI_API_KEY is set, so CI stays deterministic and free.

    pip install pytest pytest-asyncio
    pytest tests/ -v
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.services import llm, models  # noqa: E402
from app.services.models import UnknownModel  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_registry(monkeypatch):
    """Isolate from a deployment's LLM_REGISTRY_PATH.

    The registry is parsed once and cached; an override in the developer's
    environment would otherwise be what these tests assert against. Tests that
    want a custom registry point LLM_REGISTRY_PATH at one explicitly and reload.
    """
    monkeypatch.delenv("LLM_REGISTRY_PATH", raising=False)
    models.reload()
    yield
    models.registry.cache_clear()


# -- the headline requirement -----------------------------------------------


@pytest.mark.parametrize(
    "text_llm,expected_provider,expected_model",
    [
        # Exactly the four mappings the specification names.
        ("groq_gpt_oss_120b", "groq", "openai/gpt-oss-120b"),
        ("groq_llama_3_3_70b", "groq", "llama-3.3-70b-versatile"),
        ("groq_compound", "groq", "compound"),
        ("openai_gpt_5", "openai", "gpt-5"),
        # Everything that already worked must keep working.
        ("gemini_3_flash", "gemini", "gemini-3.6-flash"),
        ("gemini_3_flash_lite", "gemini", "gemini-3.5-flash-lite"),
        ("gemini_3_pro", "gemini", "gemini-3.1-pro-preview"),
        ("gemini_2_5_pro", "gemini", "gemini-2.5-pro"),
        ("gemini_flash", "gemini", "gemini-flash-latest"),
        ("gemini_flash_lite", "gemini", "gemini-flash-lite-latest"),
        ("gemini_pro", "gemini", "gemini-pro-latest"),
        ("groq_llama_3_3", "groq", "llama-3.3-70b-versatile"),
        ("groq_llama_3_1_8b", "groq", "llama-3.1-8b-instant"),
        ("llama_3_3", "groq", "llama-3.3-70b-versatile"),
    ],
)
def test_text_llm_resolves_to_provider_and_model(text_llm, expected_provider, expected_model):
    resolved = models.resolve(text_llm)
    assert resolved.provider == expected_provider
    assert resolved.model == expected_model
    assert resolved.identifier == text_llm
    assert resolved.inferred is False


def test_changing_only_text_llm_changes_provider_and_model():
    """Deliverable 6, asserted directly: one input, no code path in between."""
    a = models.resolve("groq_gpt_oss_120b")
    b = models.resolve("openai_gpt_5")
    assert (a.provider, a.model) == ("groq", "openai/gpt-oss-120b")
    assert (b.provider, b.model) == ("openai", "gpt-5")


def test_lookup_is_case_insensitive():
    assert models.resolve("GROQ_GPT_OSS_120B").model == "openai/gpt-oss-120b"


def test_no_identifier_uses_the_registry_declared_default():
    resolved = models.resolve(None)
    assert resolved.provider and resolved.model
    assert resolved.identifier == models.registry().default


def test_every_registry_model_names_a_declared_provider():
    """A typo'd provider would resolve and then fail at construction."""
    for identifier in models.known_identifiers():
        assert models.resolve(identifier).provider in models.provider_names()


def test_every_registry_provider_has_a_backend():
    for name in models.provider_names():
        assert name in llm.SUPPORTED_PROVIDERS, f"registry names {name} but no backend exists"


# -- nothing is hardcoded ----------------------------------------------------

#: Shapes of real model ids. Any of these in a string LITERAL inside app/ means a
#: model name escaped the registry and went back into code. Comments and
#: docstrings are exempt: explaining the mapping is the opposite of hiding it.
MODEL_ID_SHAPES = re.compile(
    r"(llama-3\.\d|gpt-4[o.]|gpt-5|gemini-\d|claude-[a-z]+-\d|openai/gpt-oss)"
)


def _code_string_literals(path: Path):
    """Every string literal in a module that is not a docstring."""
    tree = ast.parse(path.read_text(encoding="utf-8"))

    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                yield node.lineno, node.value


def test_no_model_ids_are_hardcoded_in_service_code():
    """The point of the exercise: model strings live in the registry alone."""
    app_dir = Path(__file__).resolve().parents[1] / "app"
    offenders = []
    for path in app_dir.rglob("*.py"):
        for lineno, value in _code_string_literals(path):
            if MODEL_ID_SHAPES.search(value):
                offenders.append(f"{path.relative_to(app_dir)}:{lineno}: {value!r}")
    assert not offenders, "model ids found outside the registry:\n" + "\n".join(offenders)


def test_no_model_selection_is_configured_in_env_files():
    """.env keeps API keys. Nothing about which model runs."""
    root = Path(__file__).resolve().parents[1]
    banned = (
        "DEFAULT_MODEL", "GROQ_MODEL", "OPENAI_MODEL", "GEMINI_MODEL",
        "ANTHROPIC_MODEL", "MODEL_NAME", "MODEL_ALIASES", "LLM_PROVIDER",
    )
    offenders = []
    for name in (".env", ".env.example"):
        path = root / name
        if not path.exists():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # OPENAI_COMPAT_MODEL_NAME is the name THIS service advertises on its
            # OpenAI-compatible route. Not a provider model.
            if stripped.startswith("OPENAI_COMPAT_MODEL_NAME"):
                continue
            if any(word in stripped for word in banned):
                offenders.append(f"{name}:{number}: {stripped}")
    assert not offenders, "model selection found in env files:\n" + "\n".join(offenders)


# -- unregistered identifiers ------------------------------------------------


def test_unregistered_identifier_infers_the_vendor_and_keeps_answering():
    resolved = models.resolve("openai_gpt_6_turbo_max")
    assert resolved.provider == "openai"
    assert resolved.inferred is True
    # The model id is the provider's declared fallback, never a reconstruction.
    assert resolved.model == models.resolve("openai_gpt_4o_mini").model


def test_inference_warns_with_the_exact_registry_entry_to_add(caplog):
    with caplog.at_level("WARNING"):
        models.resolve("groq_something_brand_new")
    assert "NOT REGISTERED" in caplog.text
    assert "groq_something_brand_new" in caplog.text


@pytest.mark.parametrize(
    "identifier,provider",
    [
        ("gemini_9_ultra", "gemini"),
        ("claude_5_opus", "anthropic"),
        ("groq_whatever", "groq"),
        ("gpt_7", "openai"),
    ],
)
def test_vendor_prefixes_come_from_the_registry(identifier, provider):
    assert models.resolve(identifier).provider == provider


def test_an_unattributable_identifier_is_refused_not_guessed():
    with pytest.raises(UnknownModel) as excinfo:
        models.resolve("definitely-not-a-model")
    assert "llm_registry" in str(excinfo.value)


# -- registry loading --------------------------------------------------------


def test_registry_path_can_be_overridden(tmp_path, monkeypatch):
    """A deployment can swap the registry without touching the image."""
    custom = tmp_path / "registry.json"
    custom.write_text(
        '{"default": "house", "providers": {"groq": {"key_env": "GROQ_API_KEY"}}, '
        '"models": {"house": {"provider": "groq", "model": "some-model-id"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_REGISTRY_PATH", str(custom))
    models.reload()

    resolved = models.resolve("house")
    assert (resolved.provider, resolved.model) == ("groq", "some-model-id")
    assert models.resolve(None).identifier == "house"


def test_a_malformed_entry_is_skipped_not_fatal(tmp_path, monkeypatch, caplog):
    custom = tmp_path / "registry.json"
    custom.write_text(
        '{"providers": {"groq": {"key_env": "GROQ_API_KEY"}}, "models": {'
        '"broken": {"provider": "groq"},'
        '"orphan": {"provider": "nosuch", "model": "x"},'
        '"good": {"provider": "groq", "model": "real-id"}}}',
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_REGISTRY_PATH", str(custom))
    with caplog.at_level("ERROR"):
        models.reload()

    assert models.resolve("good").model == "real-id"
    assert models.known_identifiers() == ["good"]


def test_a_missing_registry_is_a_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_REGISTRY_PATH", str(tmp_path / "nope.json"))
    models.registry.cache_clear()
    with pytest.raises(models.RegistryError, match="No LLM registry"):
        models.registry()


def test_key_env_is_declared_by_the_registry_not_by_code():
    assert models.key_env("groq") == "GROQ_API_KEY"
    assert models.key_env("openai") == "OPENAI_API_KEY"
    assert models.key_env("anthropic") == "ANTHROPIC_API_KEY"


# -- providers hold no model of their own ------------------------------------


def test_a_provider_called_without_a_model_fails_loudly(monkeypatch):
    """The safety net behind "nothing but the resolver chooses a model"."""
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    from app.services.llm.gemini_provider import GeminiProvider

    with pytest.raises(llm.MissingModel, match="without a model id"):
        GeminiProvider().require_model(None)


def test_get_provider_requires_an_explicit_name():
    with pytest.raises(ValueError, match="needs a provider name"):
        llm.get_provider("")


# -- backward compatibility -------------------------------------------------


def test_existing_providers_still_registered():
    for name in ("groq", "anthropic"):
        assert name in llm.SUPPORTED_PROVIDERS


def test_gemini_and_google_both_registered():
    assert "gemini" in llm.SUPPORTED_PROVIDERS
    assert "google" in llm.SUPPORTED_PROVIDERS


def test_unknown_provider_names_the_supported_ones():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm._build("wishful-thinking")


# -- provider construction / error handling ---------------------------------


def test_missing_key_raises_a_message_naming_the_variable(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "google_api_key", "")
    from app.services.llm.gemini_provider import GeminiProvider

    with pytest.raises(RuntimeError) as excinfo:
        GeminiProvider()
    assert "GEMINI_API_KEY" in str(excinfo.value)


def test_google_api_key_is_accepted_as_a_synonym(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "")
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    assert settings.gemini_key == "test-key"


def test_gemini_key_prefers_gemini_over_google(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "primary")
    monkeypatch.setattr(settings, "google_api_key", "fallback")
    assert settings.gemini_key == "primary"


# -- Gemini message translation (no network) --------------------------------


@pytest.fixture()
def provider(monkeypatch):
    monkeypatch.setattr(settings, "gemini_api_key", "test-key")
    from app.services.llm.gemini_provider import GeminiProvider

    return GeminiProvider()


def test_roles_are_translated_to_gemini_vocabulary(provider):
    contents = provider._to_contents(
        [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
            {"role": "user", "content": "bye"},
        ]
    )
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts[0].text == "hi there"


def test_tools_are_translated_to_function_declarations(provider):
    tools = provider._translate_tools(
        [
            {
                "name": "block_contact",
                "description": "Block a contact.",
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                    "additionalProperties": False,
                },
            }
        ]
    )
    assert len(tools) == 1
    declaration = tools[0].function_declarations[0]
    assert declaration.name == "block_contact"
    # Raw JSON Schema is preserved rather than coerced.
    assert declaration.parameters_json_schema["required"] == ["reason"]


def test_no_tools_means_none(provider):
    assert provider._translate_tools(None) is None
    assert provider._translate_tools([]) is None


def test_tool_results_go_back_as_function_response_parts(provider):
    from app.services.llm.base import ToolCall, ToolOutcome

    outcomes = [
        ToolOutcome(call=ToolCall(id="1", name="rag_search", arguments={}), content='{"ok":1}'),
        ToolOutcome(
            call=ToolCall(id="2", name="block_contact", arguments={}),
            content="boom",
            is_error=True,
        ),
    ]
    messages = provider.tool_result_messages(outcomes)

    assert len(messages) == 1
    content = messages[0]
    assert content.role == "user"
    assert content.parts[0].function_response.name == "rag_search"
    assert content.parts[0].function_response.response == {"result": '{"ok":1}'}
    # Errors are labelled so the model can recover rather than treating the
    # failure text as a successful result.
    assert content.parts[1].function_response.response == {"error": "boom"}


def test_tool_result_content_round_trips_through_to_contents(provider):
    """The loop appends our own Content objects back into history."""
    from app.services.llm.base import ToolCall, ToolOutcome

    tool_messages = provider.tool_result_messages(
        [ToolOutcome(call=ToolCall(id="1", name="t", arguments={}), content="x")]
    )
    contents = provider._to_contents([{"role": "user", "content": "hi"}, *tool_messages])
    assert len(contents) == 2
    assert contents[1].parts[0].function_response is not None


def test_retired_flash_ids_are_not_offered_as_defaults():
    """gemini-2.5-flash now answers "no longer available to new users".

    Registering it would hand every new key a broken model, so those identifiers
    are absent from the registry. They still READ as Gemini, so they infer to
    that provider's declared fallback rather than being refused - what must never
    happen is resolving to the retired id itself.
    """
    for gone in ("gemini_2_5_flash", "gemini_2_5_flash_lite"):
        assert gone not in models.known_identifiers()
        resolved = models.resolve(gone)
        assert resolved.provider == "gemini"
        assert resolved.inferred is True
        assert resolved.model not in {"gemini-2.5-flash", "gemini-2.5-flash-lite"}


def test_thinking_budget_is_sent_so_short_replies_are_not_truncated(provider, monkeypatch):
    monkeypatch.setattr(settings, "gemini_thinking_budget", 128)
    config = provider._config(system="s", tools=None, max_tokens=400)
    assert config.thinking_config.thinking_budget == 128


def test_negative_thinking_budget_leaves_it_to_the_model(provider, monkeypatch):
    monkeypatch.setattr(settings, "gemini_thinking_budget", -1)
    config = provider._config(system="s", tools=None, max_tokens=400)
    assert config.thinking_config is None


def test_warns_when_max_tokens_cannot_fit_the_thinking_budget(provider, monkeypatch, caplog):
    monkeypatch.setattr(settings, "gemini_thinking_budget", 128)
    with caplog.at_level("WARNING"):
        provider._config(system="s", tools=None, max_tokens=60)
    assert "thinking" in caplog.text.lower()


def test_config_carries_system_prompt_and_limits(provider):
    config = provider._config(system="You are terse.", tools=None, max_tokens=123)
    assert config.system_instruction == "You are terse."
    assert config.max_output_tokens == 123
    assert config.temperature == settings.gemini_temperature


def test_automatic_function_calling_is_disabled_when_tools_are_present(provider):
    """The agent loop runs tools itself; the SDK must not also run them."""
    config = provider._config(
        system="s",
        tools=[{"name": "t", "description": "d", "parameters": {"type": "object"}}],
        max_tokens=None,
    )
    assert config.automatic_function_calling.disable is True


def test_parsing_extracts_text_calls_and_finish_reason(provider):
    from google.genai import types

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="Blocking them."),
                        types.Part(
                            function_call=types.FunctionCall(
                                name="block_contact", args={"reason": "abusive"}
                            )
                        ),
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    text, calls, content, finish = provider._parse(response)
    assert text == "Blocking them."
    assert [c.name for c in calls] == ["block_contact"]
    assert calls[0].arguments == {"reason": "abusive"}
    assert calls[0].id, "the loop needs a non-empty id even though Gemini omits one"
    assert finish == "STOP"
    assert content is not None


def test_safety_finish_reason_is_reported_as_a_refusal(provider):
    from google.genai import types

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(role="model", parts=[]),
                finish_reason=types.FinishReason.SAFETY,
            )
        ]
    )
    _, _, _, finish = provider._parse(response)
    assert finish in provider_refusal_reasons()


def provider_refusal_reasons():
    from app.services.llm.gemini_provider import _REFUSAL_REASONS

    return _REFUSAL_REASONS


def test_empty_response_does_not_crash(provider):
    from google.genai import types

    text, calls, content, finish = provider._parse(types.GenerateContentResponse(candidates=[]))
    assert (text, calls, content, finish) == ("", [], None, None)


def test_thought_parts_never_leak_into_the_reply(provider):
    from google.genai import types

    response = types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part(text="internal reasoning", thought=True),
                        types.Part(text="the actual answer"),
                    ],
                ),
                finish_reason=types.FinishReason.STOP,
            )
        ]
    )
    text, _, _, _ = provider._parse(response)
    assert text == "the actual answer"


# -- live call, opt-in ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.skipif(
    not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")),
    reason="set GEMINI_API_KEY to run the live Gemini test",
)
async def test_live_gemini_generates_a_reply():
    provider = llm.get_provider("gemini")
    result = await provider.complete(
        system="Reply with exactly the word: OK",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=32,
    )
    assert result.text
    assert result.usage.input_tokens > 0
    await provider.aclose()




# -- logging -----------------------------------------------------------------


def test_resolution_is_logged_in_the_required_format(caplog):
    with caplog.at_level("INFO"):
        models.resolve("groq_gpt_oss_120b")
    assert "LLM Identifier: groq_gpt_oss_120b" in caplog.text
    assert "Provider: groq" in caplog.text
    assert "Model: openai/gpt-oss-120b" in caplog.text


def test_no_api_key_is_ever_logged(caplog, monkeypatch):
    monkeypatch.setattr(settings, "groq_api_key", "gsk_supersecretvalue")
    with caplog.at_level("DEBUG"):
        models.resolve("groq_gpt_oss_120b")
        models.resolve("groq_unregistered_thing")
    assert "gsk_supersecretvalue" not in caplog.text


# -- explicit provider:model passthrough -------------------------------------


def test_explicit_provider_model_is_used_verbatim():
    """The escape hatch: any model, no registry edit, no code change."""
    resolved = models.resolve("groq:a-model-released-this-morning")
    assert resolved.provider == "groq"
    assert resolved.model == "a-model-released-this-morning"
    assert resolved.inferred is False


def test_explicit_form_still_requires_a_provider_we_can_build():
    with pytest.raises(UnknownModel):
        models.resolve("nosuchprovider:whatever")


def test_registry_entries_win_over_the_explicit_form():
    """A registered identifier is never reinterpreted as provider:model."""
    assert models.resolve("groq_gpt_oss_120b").model == "openai/gpt-oss-120b"
