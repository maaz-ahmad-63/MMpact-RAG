"""Provider-neutral types shared by every LLM backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens
            + other.cache_creation_input_tokens,
        )


@dataclass
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolOutcome:
    """The result of running a ToolCall, on its way back to the model."""

    call: ToolCall
    content: str
    is_error: bool = False


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str | None = None
    model: str | None = None
    usage: Usage = field(default_factory=Usage)
    refused: bool = False
    # The assistant turn in the provider's own wire format, to append to history.
    assistant_message: Any = None


class MissingModel(Exception):
    """A provider was asked to run without a model id.

    Always a bug upstream, never a condition to paper over: the resolver is the
    only thing that chooses models, so a provider reaching this point means some
    path bypassed it.
    """


class LLMProvider(ABC):
    """Backends implement this; the agent loop knows nothing else about them.

    Providers hold NO model of their own. Which model runs is decided once, by
    `services/models.resolve()` from Laravel's `text_llm`, and passed in on every
    call. A provider that fell back to a configured default would silently answer
    on a model nobody chose — the exact failure this design exists to prevent.
    """

    name: str

    def require_model(self, model: str | None) -> str:
        """The model for this call, or a loud failure.

        Every provider calls this instead of reaching for a default.
        """
        if model and str(model).strip():
            return str(model).strip()
        raise MissingModel(
            f"The {self.name} provider was called without a model id. Models come from "
            "Laravel's `text_llm` via services/models.resolve() - nothing else may "
            "choose one."
        )

    @abstractmethod
    async def complete(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> LLMResult:
        """One model turn.

        `tools` uses the neutral shape: {"name", "description", "parameters"}.
        Providers translate it into their own schema.

        `model` overrides the provider's configured model for this call only -
        the calling application chose it, and it has already been resolved
        through the alias map, so it is a real model id by this point.
        """

    @abstractmethod
    def tool_result_messages(self, outcomes: list[ToolOutcome]) -> list[dict[str, Any]]:
        """Provider-native message(s) carrying tool results back to the model."""

    @abstractmethod
    async def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        schema_name: str = "response",
        max_tokens: int = 1024,
        model: str | None = None,
    ) -> tuple[dict[str, Any], LLMResult]:
        """A single turn constrained to JSON matching `schema`."""

    async def aclose(self) -> None:  # pragma: no cover - optional override
        return None
