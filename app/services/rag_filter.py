"""Rule-based pre-filter to detect obvious casual/non-knowledge queries.

Evaluates queries in < 1 ms without an LLM call. Skips RAG retrieval for high-confidence
casual queries (greetings, thanks, small talk, acknowledgements, jokes) while letting any
query that might require document/business knowledge pass through to RAG search.
"""

from __future__ import annotations

import re
import string

# Exact phrases (lowercase, stripped of punctuation) that are high-confidence non-knowledge queries
CASUAL_EXACT_PHRASES: set[str] = {
    # Greetings
    "hello",
    "hi",
    "hey",
    "greetings",
    "hola",
    "hey there",
    "hi there",
    "hello there",
    "good morning",
    "good afternoon",
    "good evening",
    "howdy",
    "welcome",

    # Thanks & Gratitude
    "thanks",
    "thank you",
    "thx",
    "thank u",
    "thanks a lot",
    "thank you so much",
    "many thanks",
    "cheers",
    "tanks",

    # Casual conversation / Small talk
    "how are you",
    "how are u",
    "how r u",
    "how are you doing",
    "how is it going",
    "hows it going",
    "what's up",
    "whats up",
    "what is up",
    "who are you",
    "what is your name",
    "whats your name",
    "what's your name",
    "who made you",
    "who created you",
    "are you an ai",
    "are you human",
    "are you real",
    "are you a bot",
    "are you a robot",

    # Acknowledgements / Affirmations / Negations
    "ok",
    "okay",
    "got it",
    "sure",
    "yes",
    "no",
    "k",
    "fine",
    "understood",
    "alright",
    "all right",
    "yep",
    "nope",
    "yeah",
    "nah",
    "cool",
    "great",
    "awesome",
    "perfect",
    "sounds good",
    "bye",
    "goodbye",
    "see you",
    "take care",

    # Generic creative / amusement requests (clearly non-document)
    "tell me a joke",
    "tell a joke",
    "say a joke",
    "make me laugh",
    "sing a song",
    "write a poem",
    "tell me a story",
    "tell a story",
}

# Regex patterns for high-confidence casual matching
CASUAL_REGEX_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(hi|hello|hey|good morning|good afternoon|good evening)\b", re.IGNORECASE),
    re.compile(r"^(thanks?|thank you)\b", re.IGNORECASE),
    re.compile(r"^(ok|okay|got it|sure|alright|cool|great|yep|nope|yes|no)\b", re.IGNORECASE),
    re.compile(r"^tell (me )?a joke$", re.IGNORECASE),
    re.compile(r"^(who|what) are you\b", re.IGNORECASE),
]

_PUNCTUATION_TRANSLATOR = str.maketrans("", "", string.punctuation)


def normalize_query(text: str) -> str:
    """Normalize text by lowercasing, stripping punctuation, and compressing whitespace."""
    if not text:
        return ""
    cleaned = text.translate(_PUNCTUATION_TRANSLATOR).lower()
    return " ".join(cleaned.split())


def should_skip_rag(query: str) -> bool:
    """Determine whether to skip RAG retrieval for a query.

    Returns True ONLY for high-confidence, obvious casual/non-knowledge queries.
    If there is any ambiguity, returns False so RAG similarity search can evaluate it.
    """
    if not query or not query.strip():
        return True

    normalized = normalize_query(query)

    # 1. Exact match against curated casual phrases
    if normalized in CASUAL_EXACT_PHRASES:
        return True

    # 2. Check regex patterns for very short phrases (under 6 words)
    words = normalized.split()
    if len(words) <= 5:
        for pattern in CASUAL_REGEX_PATTERNS:
            if pattern.search(normalized):
                return True

    return False
