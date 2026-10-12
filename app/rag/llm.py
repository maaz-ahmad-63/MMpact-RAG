"""LLM Provider abstraction for the RAG answer pipeline.

Decouples the RAG layer from any single LLM vendor. Supported providers:
- Groq (llama-3.3-70b-versatile, etc.)
- Google Gemini (gemini-2.5-flash, gemini-1.5-flash, etc.)
- OpenAI (gpt-4o-mini, gpt-4o, etc.)
- Anthropic (claude-3-5-sonnet, claude-3-haiku, etc.)
- Ollama (local self-hosted models)
- Mock (for deterministic offline unit tests)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .config import RagSettings, get_rag_settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM pipeline errors."""


class LLMConfigurationError(LLMError):
    """Raised when no LLM provider or API key is configured. -> HTTP 503."""


class LLMProviderError(LLMError):
    """Raised when upstream LLM provider fails (rate limit, outage, invalid request). -> HTTP 502."""


class BaseLLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    provider_name: str = "base"
    default_model: str = ""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a response given chat messages.

        Returns:
            tuple of (answer_text, usage_dict)
        """


class GroqProvider(BaseLLMProvider):
    """Groq Cloud provider (Ultra-fast GPT-OSS, Qwen, etc.)."""

    provider_name = "groq"
    default_model = "openai/gpt-oss-120b"

    def __init__(self, api_key: str):
        if not api_key:
            raise LLMConfigurationError("GROQ_API_KEY is not configured.")
        self.api_key = api_key

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        selected_model = model or self.default_model
        try:
            try:
                from groq import AsyncGroq

                client = AsyncGroq(api_key=self.api_key)
                response = await client.chat.completions.create(
                    messages=messages,
                    model=selected_model,
                    temperature=temperature if temperature is not None else 0.2,
                    max_tokens=max_tokens or 1024,
                )
                answer = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", 0) if response.usage else 0,
                    "completion_tokens": getattr(response.usage, "completion_tokens", 0) if response.usage else 0,
                    "total_tokens": getattr(response.usage, "total_tokens", 0) if response.usage else 0,
                }
                return answer, usage
            except ImportError:
                # Fallback to direct HTTP via httpx if groq library is not installed
                async with httpx.AsyncClient(timeout=60.0) as http_client:
                    resp = await http_client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": selected_model,
                            "messages": messages,
                            "temperature": temperature if temperature is not None else 0.2,
                            "max_tokens": max_tokens or 1024,
                        },
                    )
                    if resp.status_code != 200:
                        raise LLMProviderError(f"Groq API error ({resp.status_code}): {resp.text}")
                    data = resp.json()
                    answer = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return answer, usage
        except LLMProviderError:
            raise
        except Exception as error:
            logger.error("rag: Groq generation failed: %s", error)
            raise LLMProviderError(f"Groq generation failed: {error}") from error


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider via Google Generative AI / REST."""

    provider_name = "gemini"
    default_model = "gemini-2.5-flash"

    def __init__(self, api_key: str):
        if not api_key:
            raise LLMConfigurationError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not configured.")
        self.api_key = api_key

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        selected_model = model or self.default_model
        # Format messages for Gemini API
        contents = []
        system_instruction = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            elif role in ("user", "assistant"):
                gemini_role = "user" if role == "user" else "model"
                contents.append({"role": gemini_role, "parts": [{"text": content}]})

        if not contents and system_instruction:
            contents = [{"role": "user", "parts": [{"text": "Hello"}]}]

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else 0.2,
                "maxOutputTokens": max_tokens or 1024,
            },
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{selected_model}:generateContent?key={self.api_key}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    raise LLMProviderError(f"Gemini API error ({resp.status_code}): {resp.text}")
                data = resp.json()
                candidates = data.get("candidates", [])
                if not candidates:
                    return "I could not generate an answer from the retrieved documents.", {}
                parts = candidates[0].get("content", {}).get("parts", [])
                answer = "".join(p.get("text", "") for p in parts)
                usage = data.get("usageMetadata", {})
                return answer, usage
        except LLMProviderError:
            raise
        except Exception as error:
            logger.error("rag: Gemini generation failed: %s", error)
            raise LLMProviderError(f"Gemini generation failed: {error}") from error


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider (GPT-4o, GPT-4o-mini, etc.)."""

    provider_name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str):
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is not configured.")
        self.api_key = api_key

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        selected_model = model or self.default_model
        try:
            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=self.api_key)
                response = await client.chat.completions.create(
                    messages=messages,
                    model=selected_model,
                    temperature=temperature if temperature is not None else 0.2,
                    max_tokens=max_tokens or 1024,
                )
                answer = response.choices[0].message.content or ""
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                }
                return answer, usage
            except ImportError:
                async with httpx.AsyncClient(timeout=60.0) as http_client:
                    resp = await http_client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": selected_model,
                            "messages": messages,
                            "temperature": temperature if temperature is not None else 0.2,
                            "max_tokens": max_tokens or 1024,
                        },
                    )
                    if resp.status_code != 200:
                        raise LLMProviderError(f"OpenAI API error ({resp.status_code}): {resp.text}")
                    data = resp.json()
                    answer = data["choices"][0]["message"]["content"]
                    usage = data.get("usage", {})
                    return answer, usage
        except LLMProviderError:
            raise
        except Exception as error:
            logger.error("rag: OpenAI generation failed: %s", error)
            raise LLMProviderError(f"OpenAI generation failed: {error}") from error


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider."""

    provider_name = "anthropic"
    default_model = "claude-3-5-sonnet-20241022"

    def __init__(self, api_key: str):
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is not configured.")
        self.api_key = api_key

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        selected_model = model or self.default_model
        system_prompt = ""
        claude_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                claude_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

        try:
            try:
                from anthropic import AsyncAnthropic

                client = AsyncAnthropic(api_key=self.api_key)
                kwargs: dict[str, Any] = {
                    "model": selected_model,
                    "max_tokens": max_tokens or 1024,
                    "messages": claude_messages,
                    "temperature": temperature if temperature is not None else 0.2,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                response = await client.messages.create(**kwargs)
                answer = "".join(block.text for block in response.content if hasattr(block, "text"))
                usage = {
                    "input_tokens": getattr(response.usage, "input_tokens", 0) if response.usage else 0,
                    "output_tokens": getattr(response.usage, "output_tokens", 0) if response.usage else 0,
                }
                return answer, usage
            except ImportError:
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                }
                body: dict[str, Any] = {
                    "model": selected_model,
                    "max_tokens": max_tokens or 1024,
                    "messages": claude_messages,
                    "temperature": temperature if temperature is not None else 0.2,
                }
                if system_prompt:
                    body["system"] = system_prompt
                async with httpx.AsyncClient(timeout=60.0) as http_client:
                    resp = await http_client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers=headers,
                        json=body,
                    )
                    if resp.status_code != 200:
                        raise LLMProviderError(f"Anthropic API error ({resp.status_code}): {resp.text}")
                    data = resp.json()
                    answer = "".join(block.get("text", "") for block in data.get("content", []))
                    usage = data.get("usage", {})
                    return answer, usage
        except LLMProviderError:
            raise
        except Exception as error:
            logger.error("rag: Anthropic generation failed: %s", error)
            raise LLMProviderError(f"Anthropic generation failed: {error}") from error


class OllamaProvider(BaseLLMProvider):
    """Local self-hosted Ollama provider."""

    provider_name = "ollama"
    default_model = "llama3.2"

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        selected_model = model or self.default_model
        payload = {
            "model": selected_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else 0.2,
                "num_predict": max_tokens or 1024,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self.base_url}/api/chat", json=payload)
                if resp.status_code != 200:
                    raise LLMProviderError(f"Ollama error ({resp.status_code}): {resp.text}")
                data = resp.json()
                answer = data.get("message", {}).get("content", "")
                usage = {
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                }
                return answer, usage
        except LLMProviderError:
            raise
        except Exception as error:
            logger.error("rag: Ollama generation failed: %s", error)
            raise LLMProviderError(f"Ollama generation failed: {error}") from error


class MockLLMProvider(BaseLLMProvider):
    """Mock LLM provider for unit tests."""

    provider_name = "mock"
    default_model = "mock-model"

    def __init__(self, custom_answer: str | None = None):
        self.custom_answer = custom_answer

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        if self.custom_answer is not None:
            return self.custom_answer, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        # Check user query in messages
        user_query = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_query = m.get("content", "")
                break

        answer = f"Based on the provided documentation, here is the answer to '{user_query[:50]}': details verified. [SOURCE 1]"
        return answer, {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125}


def get_llm_provider(settings: RagSettings | None = None) -> BaseLLMProvider:
    """Resolve and instantiate the configured LLM provider."""
    config = settings or get_rag_settings()
    provider_name = (config.rag_llm_provider or "auto").strip().lower()

    if provider_name == "mock":
        return MockLLMProvider()

    if provider_name == "groq":
        return GroqProvider(config.groq_api_key)

    if provider_name in ("gemini", "google"):
        key = config.gemini_api_key or config.google_api_key
        return GeminiProvider(key)

    if provider_name == "openai":
        return OpenAIProvider(config.openai_api_key)

    if provider_name == "anthropic":
        return AnthropicProvider(config.anthropic_api_key)

    if provider_name == "ollama":
        return OllamaProvider(config.ollama_base_url)

    if provider_name == "auto":
        # Auto-detect first available provider with configured API key
        if config.groq_api_key:
            return GroqProvider(config.groq_api_key)
        if config.gemini_api_key or config.google_api_key:
            return GeminiProvider(config.gemini_api_key or config.google_api_key)
        if config.openai_api_key:
            return OpenAIProvider(config.openai_api_key)
        if config.anthropic_api_key:
            return AnthropicProvider(config.anthropic_api_key)

        raise LLMConfigurationError(
            "No LLM provider is configured. Please set RAG_LLM_PROVIDER and configure "
            "one of GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY in .env."
        )

    raise LLMConfigurationError(
        f"Unsupported RAG_LLM_PROVIDER '{provider_name}'. "
        f"Supported providers are: auto, groq, gemini, openai, anthropic, ollama, mock."
    )
