"""Splitting cleaned text into chunks that are worth embedding.

An embedding is one vector for the whole chunk, so a chunk is the unit of
retrieval AND the unit of meaning. Two failure modes bound the design:

  too large  - one vector averages several unrelated topics, matches everything
               weakly and nothing strongly, and drags irrelevant text into the
               prompt along with the part that was actually wanted.
  too small  - "See section 4.2." embeds to a vector about nothing, and may
               out-rank the paragraph that holds the answer.

So the splitter is boundary-aware: it breaks at paragraphs, then at sentences,
and only splits mid-sentence when a single sentence is longer than a whole
chunk. Size and overlap come from RAG_CHUNK_SIZE / RAG_CHUNK_OVERLAP - there is
no literal anywhere else in the codebase.

Overlap exists because the answer to a question does not respect the boundary
the splitter chose. Repeating the tail of one chunk at the head of the next
means a sentence cut in half is still whole in one of them.

Sizes are in CHARACTERS, not tokens: no tokeniser is loaded here, the numbers
stay meaningful when the embedding model changes, and BGE-M3's 8192-token window
is far away from any sane chunk size.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .config import RagSettings, rag_settings
from .document_loader import Segment

# A blank line is the author's own statement that a topic ended.
_PARAGRAPH = re.compile(r"\n\s*\n")
# Sentence end followed by whitespace. Deliberately simple: over-splitting is
# harmless here because the pieces are re-packed to the target size afterwards.
_SENTENCE = re.compile(r"(?<=[.!?。！？])\s+|\n")


@dataclass
class Chunk:
    text: str
    index: int
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_segments(
    segments: Iterable[Segment],
    *,
    document_id: str,
    source: str,
    settings: RagSettings | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Chunk every segment, numbering chunks continuously across the document.

    Chunking is per segment so a chunk never spans two pages, which keeps the
    page in its metadata a true citation rather than an approximation.
    """
    config = settings or rag_settings
    chunks: list[Chunk] = []

    for segment in segments:
        for text in split_text(
            segment.text,
            chunk_size=config.rag_chunk_size,
            overlap=config.rag_chunk_overlap,
            min_chunk_size=config.rag_min_chunk_size,
        ):
            chunks.append(
                Chunk(
                    text=text,
                    index=len(chunks),
                    page=segment.page,
                    metadata={
                        "document_id": document_id,
                        "chunk_index": len(chunks),
                        "source": source,
                        "page": segment.page,
                        **(extra_metadata or {}),
                    },
                )
            )

    return chunks


def split_text(
    text: str,
    *,
    chunk_size: int,
    overlap: int,
    min_chunk_size: int = 0,
) -> list[str]:
    """One string into overlapping, boundary-aligned pieces."""
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    if len(text) <= chunk_size:
        return [text]

    pieces: list[str] = []
    current = ""

    for atom in _atoms(text, chunk_size):
        if not current:
            current = atom
            continue

        joiner = "" if current.endswith("\n") else " "
        if len(current) + len(joiner) + len(atom) <= chunk_size:
            current = f"{current}{joiner}{atom}"
            continue

        pieces.append(current)
        # Start the next chunk with the tail of this one, so a thought split
        # across the boundary survives whole on one side of it.
        current = f"{_tail(current, overlap)}{atom}" if overlap else atom

    if current:
        pieces.append(current)

    return _absorb_runts(pieces, min_chunk_size, chunk_size)


def _atoms(text: str, chunk_size: int) -> list[str]:
    """The smallest units the packer is allowed to move around.

    Paragraphs where they fit, sentences where a paragraph is too long, and a
    hard character split only for a single sentence longer than a whole chunk
    (a table row, a minified block, a language without spaces).
    """
    atoms: list[str] = []
    for paragraph in _PARAGRAPH.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            atoms.append(paragraph)
            continue

        for sentence in _SENTENCE.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) <= chunk_size:
                atoms.append(sentence)
            else:
                atoms.extend(
                    sentence[start:start + chunk_size]
                    for start in range(0, len(sentence), chunk_size)
                )
    return atoms


def _tail(text: str, overlap: int) -> str:
    """The last `overlap` characters, cut at a word boundary where possible."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:]
    space = tail.find(" ")
    # Only realign when a boundary exists reasonably early; otherwise a single
    # long token would shrink the overlap to almost nothing.
    if 0 <= space < overlap // 2:
        tail = tail[space + 1:]
    return f"{tail.strip()} " if tail.strip() else ""


def _absorb_runts(pieces: list[str], min_chunk_size: int, chunk_size: int) -> list[str]:
    """Fold undersized pieces into their neighbour.

    A trailing fragment is the common case: the last paragraph of a document
    lands alone in a chunk of forty characters. Merged backwards it is context;
    stored alone it is a row that can out-rank real content.

    A document whose entire text is shorter than the minimum is kept as it is -
    it is short, not broken, and dropping it would silently lose the file.
    """
    if min_chunk_size <= 0 or len(pieces) < 2:
        return pieces

    merged: list[str] = []
    for piece in pieces:
        if (
            merged
            and len(piece) < min_chunk_size
            # Bounded overflow: better one 1.25x chunk than a runt, but never an
            # unbounded chunk built out of many.
            and len(merged[-1]) + len(piece) <= int(chunk_size * 1.25)
        ):
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged
