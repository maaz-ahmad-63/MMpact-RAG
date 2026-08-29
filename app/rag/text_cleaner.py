"""Normalisation of extracted text, before it is chunked.

Extraction is lossy in predictable ways: a PDF gives hyphenated line breaks and
a form feed per page, DOCX gives non-breaking spaces and smart quotes, and every
format gives runs of blank lines. Left alone those artefacts become part of the
embedded text, so two identical sentences from two file types embed differently.

The rule followed here: repair extraction damage, change nothing else. Case,
punctuation and stop words are all left exactly as written, because BGE-M3 is a
semantic model and stripping them removes meaning it uses.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width and bidi control characters. Invisible on screen, but they split
# words for a tokeniser and survive every other rule below.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
# A word broken across a line by a hyphen: "inter-\nnational" -> "international".
_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\n[ \t]*(\w)")
# A single newline inside a sentence is a wrap, not a break: join it with a space.
_SOFT_WRAP = re.compile(r"(?<![\n\r])\n(?![\n\r])")
_SPACES = re.compile(r"[ \t\u00a0\u2000-\u200a]+")
_BLANK_LINES = re.compile(r"\n{3,}")
# Control characters other than tab and newline, which extraction sometimes
# leaves behind and Postgres rejects inside a text value (\u0000 especially).
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Extraction artefacts out, meaning untouched."""
    if not text:
        return ""

    # NFKC folds ligatures ("ﬁ" -> "fi") and full-width forms that extraction
    # produces, so the same word from two sources embeds as the same word.
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = _INVISIBLE.sub("", cleaned)
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = _CONTROL.sub(" ", cleaned)
    cleaned = _HYPHEN_BREAK.sub(r"\1\2", cleaned)
    cleaned = _SOFT_WRAP.sub(" ", cleaned)
    cleaned = _SPACES.sub(" ", cleaned)
    cleaned = _BLANK_LINES.sub("\n\n", cleaned)

    return "\n".join(line.strip() for line in cleaned.split("\n")).strip()


def is_effectively_empty(text: str) -> bool:
    """True when there is nothing worth embedding.

    A scanned PDF typically extracts to a handful of stray punctuation marks
    rather than to nothing at all, so "empty" has to mean "no letters or digits",
    not "zero characters".
    """
    return not any(character.isalnum() for character in text or "")
