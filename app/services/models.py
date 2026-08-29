"""The LLM resolver: `text_llm` in, provider + model id out.

Laravel is the single source of truth for which model answers a conversation. It
sends one string — `text_llm` — and this module is the only place that turns it
into something a provider SDK can be called with.

    text_llm "groq_gpt_oss_120b"  ->  provider "groq", model "openai/gpt-oss-120b"
    text_llm "openai_gpt_5"       ->  provider "openai", model "gpt-5"

Where the mapping lives
-----------------------
In `llm_registry.json`, next to this file, and nowhere else. Not in `.env`, not
in a provider class, not in an `if provider == …` inside business logic. Model
ids have to be written down somewhere — the discipline is that they are written
down ONCE, as data, so that adding a model a dashboard offers is an edit to a
JSON file rather than a change to code. Point `LLM_REGISTRY_PATH` at a different
file to override it per deployment.

`.env` keeps API keys. That is all it keeps about models.

Unknown identifiers
-------------------
A name Laravel sends that the registry has never heard of is a real operational
event: someone added a model in the dashboard and nobody updated the registry.
Two things happen, in this order:

  1. The vendor is read off the front of the name (`openai_gpt_6_turbo` -> the
     `openai` provider) and that provider's declared fallback model is used, so
     WhatsApp keeps answering.
  2. It is logged as a WARNING naming the exact registry entry to add.

What is deliberately NOT done is guessing the model id. Turning
`groq_llama_3_3_70b` into `llama-3.3-70b-versatile` requires knowing the
`-versatile` suffix; inventing it produces a model name that fails at the API
with a far more confusing error than "not in the registry".
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Shipped with the service. `LLM_REGISTRY_PATH` overrides it.
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("llm_registry.json")


@dataclass(frozen=True)
class ResolvedModel:
    """One answer from the resolver: who to call, and with what."""

    provider: str
    model: str
    #: The `text_llm` value this came from, kept for logging and telemetry.
    identifier: str | None = None
    #: True when the identifier was not in the registry and the vendor prefix
    #: had to be read off the name. Callers may want to surface this.
    inferred: bool = False

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


class UnknownModel(Exception):
    """An identifier that is neither registered nor attributable to a vendor."""


class RegistryError(Exception):
    """The registry file is missing or malformed. Fatal - nothing can resolve."""


@dataclass(frozen=True)
class ProviderEntry:
    name: str
    key_env: str
    fallback: str | None
    prefixes: tuple[str, ...]


@dataclass(frozen=True)
class Registry:
    """The parsed registry. Immutable; reloaded only by `reload()`."""

    default: str | None
    providers: dict[str, ProviderEntry]
    models: dict[str, tuple[str, str]]
    source: str

    #: Vendor prefix -> provider name, flattened from `providers[*].prefixes`.
    prefixes: dict[str, str]


def registry_path() -> Path:
    """Where the registry is read from. A PATH in the environment, not a model."""
    override = os.environ.get("LLM_REGISTRY_PATH", "").strip()
    return Path(override) if override else DEFAULT_REGISTRY_PATH


def _parse(raw: dict[str, Any], source: str) -> Registry:
    providers: dict[str, ProviderEntry] = {}
    prefixes: dict[str, str] = {}

    for name, entry in (raw.get("providers") or {}).items():
        key = str(name).strip().lower()
        if not isinstance(entry, dict):
            logger.error("registry provider %r is not an object - skipping", name)
            continue
        provider = ProviderEntry(
            name=key,
            key_env=str(entry.get("key_env") or "").strip(),
            fallback=(str(entry["fallback"]).strip().lower() if entry.get("fallback") else None),
            prefixes=tuple(str(p).strip().lower() for p in (entry.get("prefixes") or []) if p),
        )
        providers[key] = provider
        for prefix in provider.prefixes:
            prefixes[prefix] = key

    models: dict[str, tuple[str, str]] = {}
    for identifier, entry in (raw.get("models") or {}).items():
        key = str(identifier).strip().lower()
        if not isinstance(entry, dict):
            logger.error("registry model %r is not an object - skipping", identifier)
            continue
        provider = str(entry.get("provider") or "").strip().lower()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            # Skip rather than crash: one malformed row must not take the whole
            # service down, and the resulting "not registered" warning names it.
            logger.error(
                "registry entry %r needs both 'provider' and 'model' - skipping", identifier
            )
            continue
        if provider not in providers:
            logger.error(
                "registry entry %r names provider %r, which is not declared under "
                "'providers' - skipping",
                identifier,
                provider,
            )
            continue
        models[key] = (provider, model)

    if not models:
        raise RegistryError(
            f"The LLM registry at {source} declares no usable models. Nothing can be "
            "resolved, so no reply can be generated."
        )

    default = raw.get("default")
    default = str(default).strip().lower() if default else None
    if default and default not in models:
        logger.error(
            "registry default %r is not one of its own models - falling back to no default",
            default,
        )
        default = None

    return Registry(
        default=default, providers=providers, models=models, prefixes=prefixes, source=source
    )


@lru_cache(maxsize=1)
def registry() -> Registry:
    """The parsed registry, read once per process."""
    path = registry_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RegistryError(
            f"No LLM registry at {path}. This service resolves every model through it, "
            "so it cannot answer anything without it."
        ) from error
    except json.JSONDecodeError as error:
        raise RegistryError(f"The LLM registry at {path} is not valid JSON: {error}") from error

    parsed = _parse(raw, str(path))
    logger.info(
        "LLM registry loaded: %d models across %d providers from %s (default=%s)",
        len(parsed.models),
        len(parsed.providers),
        parsed.source,
        parsed.default or "none",
    )
    return parsed


def reload() -> Registry:
    """Drop the cached registry and read it again. Used by tests and /health."""
    registry.cache_clear()
    return registry()


def known_identifiers() -> list[str]:
    return sorted(registry().models)


def provider_names() -> list[str]:
    return sorted(registry().providers)


def key_env(provider: str) -> str:
    """Which .env variable holds this provider's API key."""
    entry = registry().providers.get(provider.lower())
    return entry.key_env if entry else ""


def default_model() -> ResolvedModel:
    """For callers that send no identifier at all.

    Not a "default model" setting in the old sense: the registry declares it, so
    it is still data rather than a hardcoded name, and Laravel's WhatsApp path
    never reaches this — it always sends `text_llm`.
    """
    reg = registry()
    if not reg.default:
        raise UnknownModel(
            "No model identifier was supplied and the registry declares no 'default'. "
            f"Send `text_llm` (one of: {', '.join(known_identifiers())}), or add a "
            f"'default' to {reg.source}."
        )
    provider, model = reg.models[reg.default]
    return ResolvedModel(provider=provider, model=model, identifier=reg.default)


def _infer(identifier: str) -> ResolvedModel | None:
    """Read the vendor off the front of an unregistered identifier.

    `openai_gpt_6` -> the openai provider's declared fallback model. The vendor
    is the part that can be determined safely; the exact model id is not, so the
    provider's own fallback is used rather than a reconstruction.
    """
    reg = registry()
    head = identifier.strip().lower().replace("-", "_").split("_", 1)[0]
    provider_name = reg.prefixes.get(head)
    if provider_name is None:
        return None

    entry = reg.providers.get(provider_name)
    fallback = entry.fallback if entry else None
    if not fallback or fallback not in reg.models:
        logger.error(
            "identifier %r points at provider %r, but that provider declares no usable "
            "'fallback' in the registry",
            identifier,
            provider_name,
        )
        return None

    provider, model = reg.models[fallback]
    return ResolvedModel(provider=provider, model=model, identifier=identifier, inferred=True)


def resolve(identifier: str | None) -> ResolvedModel:
    """Turn Laravel's `text_llm` into a provider and a real model id.

    This is the ONLY place that decision is made. Every AI service path goes
    through it.
    """
    if not identifier or not str(identifier).strip():
        return default_model()

    name = str(identifier).strip()
    reg = registry()

    hit = reg.models.get(name.lower())
    if hit is not None:
        resolved = ResolvedModel(provider=hit[0], model=hit[1], identifier=name)
        logger.info(
            "LLM Identifier: %s | Provider: %s | Model: %s",
            resolved.identifier,
            resolved.provider,
            resolved.model,
        )
        return resolved

    # An explicit `provider:model`, used verbatim. This is the escape hatch that
    # makes ANY model reachable without touching the registry: a dashboard can
    # send `groq:some-model-released-this-morning` and it just works, because the
    # id is stated rather than looked up. The provider must be one we can build,
    # so a typo still fails loudly instead of reaching a backend that is not there.
    provider, sep, model = name.partition(":")
    if sep and model.strip() and provider.strip().lower() in reg.providers:
        resolved = ResolvedModel(
            provider=provider.strip().lower(), model=model.strip(), identifier=name
        )
        logger.info(
            "LLM Identifier: %s | Provider: %s | Model: %s (explicit provider:model)",
            resolved.identifier,
            resolved.provider,
            resolved.model,
        )
        return resolved

    inferred = _infer(name)
    if inferred is not None:
        logger.warning(
            "LLM Identifier: %s | Provider: %s | Model: %s  <- NOT REGISTERED. The vendor "
            "was read off the name and that provider's fallback model used. Add "
            '"%s": {"provider": "%s", "model": "<real-model-id>"} to %s to pin it.',
            inferred.identifier,
            inferred.provider,
            inferred.model,
            name.lower(),
            inferred.provider,
            reg.source,
        )
        return inferred

    raise UnknownModel(
        f"text_llm '{name}' is not in the LLM registry, is not an explicit "
        f"'<provider>:<model-id>', and no vendor could be read from it. Add it to "
        f"{reg.source} as "
        f'"{name.lower()}": {{"provider": "<{"|".join(provider_names())}>", '
        f'"model": "<real-model-id>"}}. Currently registered: '
        f"{', '.join(known_identifiers())}"
    )
