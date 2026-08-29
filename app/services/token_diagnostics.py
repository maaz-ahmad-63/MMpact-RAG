"""Per-request prompt-size breakdown, logged immediately before the LLM call.

Why an estimate rather than a real tokeniser: the providers here (Groq, Gemini,
Anthropic) do not share one, and pulling in `tiktoken` would add a large
dependency to report a number that is only ever used for diagnosis. Characters
divided by `CHARS_PER_TOKEN` tracks real usage closely enough to see which part
of a prompt is large, which is the entire purpose. The authoritative counts stay
what the provider reports back in `usage`, and those are what reach dm_ai_logs.

Off by default. Set `LOG_TOKEN_BREAKDOWN=true` to switch it on; it logs at INFO
and prints one block per provider call, so it is not something to leave running
on a busy number.

Nothing here is allowed to affect a reply: `log_prompt_breakdown` swallows its
own errors, exactly as the reporting paths in the gateway do.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Rough average across English prose and JSON for byte-pair encoders.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def _content_text(message: Any) -> str:
    """The billable text of one message, whatever shape it arrived in.

    Tool calls and tool results are part of the prompt on the next round-trip,
    so their serialised form counts too - omitting them would under-report
    exactly the case a tool loop makes expensive.
    """
    if not isinstance(message, dict):
        return str(getattr(message, "content", message) or "")

    parts: list[str] = []
    content = message.get("content")
    if isinstance(content, str):
        parts.append(content)
    elif content is not None:
        parts.append(json.dumps(content, ensure_ascii=False, default=str))

    for key in ("tool_calls", "tool_call_id", "name"):
        value = message.get(key)
        if value:
            parts.append(value if isinstance(value, str) else json.dumps(value, default=str))

    return "".join(parts)


def _tools_text(tools: list[dict[str, Any]] | None) -> str:
    if not tools:
        return ""
    return json.dumps(tools, ensure_ascii=False, default=str)


def log_prompt_breakdown(
    *,
    model: str,
    system: str,
    messages: list[Any],
    tools: list[dict[str, Any]] | None,
    max_tokens: int | None = None,
    provider_call: int = 1,
) -> None:
    """Log what is about to be sent, split by where the characters come from."""
    try:
        system_chars = len(system or "")
        tools_chars = len(_tools_text(tools))

        per_message = [(_content_text(m), m) for m in messages]
        # The last user turn is the message being answered; everything before it
        # is history. That split is the one that matters when a short "hi"
        # produces a large prompt.
        user_chars = len(per_message[-1][0]) if per_message else 0
        history_chars = sum(len(text) for text, _ in per_message[:-1])

        total_chars = system_chars + tools_chars + user_chars + history_chars

        logger.info(
            "PROMPT BREAKDOWN (provider call %s)\n"
            "  MODEL                  %s\n"
            "  MESSAGE COUNT          %s (+1 system)\n"
            "  SYSTEM PROMPT          chars=%s  est_tokens=%s\n"
            "  USER MESSAGE           chars=%s  est_tokens=%s\n"
            "  HISTORY                chars=%s  est_tokens=%s\n"
            "  TOOLS                  chars=%s  est_tokens=%s  count=%s\n"
            "  TOTAL ESTIMATED INPUT  chars=%s  est_tokens=%s\n"
            "  max_tokens (output cap, NOT input) = %s",
            provider_call,
            model,
            len(messages),
            system_chars, estimate_tokens(system or ""),
            user_chars, (user_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN,
            history_chars, (history_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN,
            tools_chars, estimate_tokens(_tools_text(tools)), len(tools or []),
            total_chars, (total_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN,
            max_tokens,
        )

        lines = [
            f"  [0] role=system\n      characters={system_chars}"
            f"\n      estimated_tokens={estimate_tokens(system or '')}"
        ]
        for index, (text, message) in enumerate(per_message, start=1):
            role = message.get("role", "unknown") if isinstance(message, dict) else "unknown"
            lines.append(
                f"  [{index}] role={role}\n      characters={len(text)}"
                f"\n      estimated_tokens={estimate_tokens(text)}"
            )
        logger.info("PROMPT MESSAGES\n%s", "\n".join(lines))
    except Exception:  # noqa: BLE001 - diagnostics must never break a reply
        logger.debug("token breakdown failed", exc_info=True)
