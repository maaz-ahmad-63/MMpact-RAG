"""Pluggable LLM providers.

There is no default provider. Which one runs is decided per request by the
resolver in `services/models.py` from Laravel's `text_llm`, and this module only
knows how to construct the SDK client for a name it is handed.

The agent loop in `services/agent.py` is provider-agnostic: it deals only with
the neutral types in `base.py`.
"""

from __future__ import annotations

from importlib import import_module

from .base import LLMProvider, LLMResult, MissingModel, ToolCall, ToolOutcome, Usage

#: One client per provider, built on first use and reused. Different
#: conversations resolve to different providers, so several are live at a time.
_providers: dict[str, LLMProvider] = {}


#: Provider name -> (module, class). Imported lazily so a deployment that only
#: ever resolves to Groq never needs the openai, anthropic or google-genai
#: packages installed.
#:
#: This maps a name to an IMPLEMENTATION. It is not a model mapping and holds no
#: model ids - those live in services/llm_registry.json.
_BACKENDS = {
    "groq": (".groq_provider", "GroqProvider"),
    "openai": (".openai_provider", "OpenAIProvider"),
    "anthropic": (".anthropic_provider", "AnthropicProvider"),
    "gemini": (".gemini_provider", "GeminiProvider"),
    # Google's own naming for the same backend.
    "google": (".gemini_provider", "GeminiProvider"),
}

SUPPORTED_PROVIDERS = sorted(set(_BACKENDS))


def _build(name: str) -> LLMProvider:
    entry = _BACKENDS.get(name)
    if entry is None:
        raise ValueError(
            f"Unknown LLM provider: {name}. Supported: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    module_name, class_name = entry
    try:
        module = import_module(module_name, __name__)
    except ImportError as error:  # missing optional SDK
        raise RuntimeError(
            f"The '{name}' provider needs a package that is not installed: {error}. "
            f"Install it (see requirements.txt), or point this text_llm at another "
            f"provider in app/services/llm_registry.json."
        ) from error

    return getattr(module, class_name)()


def get_provider(name: str) -> LLMProvider:
    """The provider named by the resolver.

    Required, not optional: a caller that could omit it would be choosing a
    provider by default, which is the thing this design removes.
    """
    if not name or not str(name).strip():
        raise ValueError(
            "get_provider() needs a provider name. It comes from "
            "services/models.resolve(text_llm) - nothing else may choose one."
        )
    resolved = str(name).strip().lower()
    provider = _providers.get(resolved)
    if provider is None:
        provider = _providers[resolved] = _build(resolved)
    return provider


async def shutdown() -> None:
    for provider in list(_providers.values()):
        await provider.aclose()
    _providers.clear()


__all__ = [
    "SUPPORTED_PROVIDERS",
    "LLMProvider",
    "LLMResult",
    "MissingModel",
    "ToolCall",
    "ToolOutcome",
    "Usage",
    "get_provider",
    "shutdown",
]
