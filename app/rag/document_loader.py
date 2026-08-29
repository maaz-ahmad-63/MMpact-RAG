"""File bytes in, text segments out. One function per supported format.

A *segment* is a piece of text with the page it came from. PDFs have real pages;
DOCX and TXT do not, and they report `page=None` rather than inventing one - a
citation that says "page 1" of a text file is worse than no page at all.

The heavy parsers (pypdf, python-docx) are imported inside their own extractor,
not at module import. An installation missing one still serves the other two
formats, and reports a clear 503 for the one it cannot read instead of failing
to start.

Adding OCR later
----------------
`_pdf_segments` already isolates the "this page produced no text" case and hands
it to `_OCR_HOOK`, which is a no-op by default. A scanned-PDF backend (tesseract,
PaddleOCR, a hosted service) is installed by calling `set_ocr_hook(...)` once at
startup: pages that already contain text never reach it, so the normal PDF path
is unchanged by anything OCR does or fails to do.
"""

from __future__ import annotations

import codecs
import io
import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: tuple[str, ...] = ("pdf", "docx", "txt")


class UnsupportedFileType(Exception):
    """The extension is not one this loader handles. -> HTTP 400."""


class ExtractionError(Exception):
    """The file is of a supported type but could not be read. -> HTTP 422."""


class ParserUnavailable(Exception):
    """The parser for a supported type is not installed. -> HTTP 503."""


@dataclass(frozen=True)
class Segment:
    """A run of text and where it came from."""

    text: str
    page: int | None = None


#: Called with (page_number, raw_page_bytes_or_None) for pages that extracted no
#: text. Returns text, or "" when it cannot help. Replaced via set_ocr_hook.
_OCR_HOOK: Callable[[int, object], str] = lambda page_number, page: ""


def set_ocr_hook(hook: Callable[[int, object], str]) -> None:
    """Install an OCR backend for image-only PDF pages."""
    global _OCR_HOOK
    _OCR_HOOK = hook
    logger.info("rag: OCR hook installed (%s)", getattr(hook, "__name__", hook))


def file_type_of(filename: str) -> str:
    """The normalised extension, or raise UnsupportedFileType."""
    extension = (filename or "").rsplit(".", 1)[-1].strip().lower() if "." in (filename or "") else ""
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"Unsupported file type '{extension or filename}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return extension


def load(content: bytes, filename: str) -> list[Segment]:
    """Extract text from an uploaded file. Raises on an unreadable one."""
    file_type = file_type_of(filename)
    if not content:
        raise ExtractionError("The uploaded file is empty (0 bytes)")

    extractor = {"pdf": _pdf_segments, "docx": _docx_segments, "txt": _txt_segments}[file_type]
    segments = extractor(content)
    return [segment for segment in segments if segment.text.strip()]





def _pdf_segments(content: bytes) -> list[Segment]:
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ParserUnavailable(
            "PDF support needs `pypdf`. Install it, or upload DOCX/TXT."
        ) from error

    try:
        reader = PdfReader(io.BytesIO(content))
    except Exception as error:  # noqa: BLE001 - any parser failure is one 422
        raise ExtractionError(f"Could not read the PDF: {error}") from error

    if getattr(reader, "is_encrypted", False):
        # An empty user password is common and legitimately readable.
        try:
            reader.decrypt("")
        except Exception as error:  # noqa: BLE001
            raise ExtractionError("The PDF is password-protected") from error

    segments: list[Segment] = []
    blank_pages = 0
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:  # noqa: BLE001 - one bad page is not a bad file
            logger.warning("rag: page %s of the PDF could not be read (%s)", index, error)
            text = ""

        if not text.strip():
            # Image-only page: the extension point, inert until an OCR backend
            # is installed. Failure here must not lose the pages that did work.
            blank_pages += 1
            try:
                text = _OCR_HOOK(index, page) or ""
            except Exception as error:  # noqa: BLE001
                logger.warning("rag: OCR hook failed on page %s (%s)", index, error)
                text = ""

        segments.append(Segment(text=text, page=index))

    if blank_pages:
        logger.info(
            "rag: %s of %s PDF pages produced no text (scanned pages need OCR)",
            blank_pages, len(segments),
        )
    return segments


# -- DOCX -------------------------------------------------------------------


def _docx_segments(content: bytes) -> list[Segment]:
    try:
        import docx  # python-docx
    except ImportError as error:  # pragma: no cover
        raise ParserUnavailable(
            "DOCX support needs `python-docx`. Install it, or upload PDF/TXT."
        ) from error

    try:
        document = docx.Document(io.BytesIO(content))
    except Exception as error:  # noqa: BLE001
        raise ExtractionError(f"Could not read the DOCX: {error}") from error

    parts: list[str] = [
        paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()
    ]

    # Tables carry the facts a policy document is actually asked about (prices,
    # SLAs, limits). Rendered row by row with " | " between cells: readable to an
    # embedding model and to a human reading the retrieved chunk, without
    # pretending to reconstruct the layout.
    for table in document.tables:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            parts.append("\n".join(rows))

    # DOCX has no page concept until it is rendered - page is honestly None.
    return [Segment(text="\n\n".join(parts), page=None)]


# -- TXT --------------------------------------------------------------------

# Order matters. UTF-16 is tried ONLY when a BOM says so: without one it
# happily "decodes" ordinary 8-bit text into CJK nonsense, and that nonsense
# would be embedded and stored as if it were the document.
_BOMS = (
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_ENCODINGS = ("utf-8", "cp1252", "latin-1")


def _txt_segments(content: bytes) -> list[Segment]:
    for bom, encoding in _BOMS:
        if content.startswith(bom):
            try:
                return [Segment(text=content.decode(encoding), page=None)]
            except UnicodeDecodeError:
                break

    for encoding in _ENCODINGS:
        try:
            return [Segment(text=content.decode(encoding), page=None)]
        except (UnicodeDecodeError, LookupError):
            continue
    # latin-1 decodes any byte string, so reaching here means the tuple above was
    # edited; keep a definite answer rather than an exception nobody expects.
    return [Segment(text=content.decode("latin-1", errors="replace"), page=None)]
