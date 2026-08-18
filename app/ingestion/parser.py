"""
Document parser – supports PDF and plain-text files.

Produces a ``ParsedDocument`` made up of typed ``Element`` objects
(heading / paragraph / table) preserving page numbers and bounding boxes
for downstream metadata.  A backwards-compatible ``.text`` property gives
flat text when callers do not need structural information.

Public API
----------
load_document(path)  → ParsedDocument
load_pdf(path)       → str   (backwards-compat flat text)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

import pdfplumber

from app.ingestion.exceptions import DocumentParsingError

logger = logging.getLogger(__name__)


# Data models

@dataclass
class Element:
    """A single structural element extracted from a document page."""

    type: str                               # "heading" | "paragraph" | "table"
    text: str                               # rendered text (markdown for tables)
    page: int                               # 1-indexed page number
    bbox: Tuple[float, float, float, float] # (x0, top, x1, bottom)
    table_data: Optional[List[List[str]]] = None  # raw cells, tables only


@dataclass
class ParsedDocument:
    """Ordered list of structural elements extracted from a document."""

    elements: List[Element] = field(default_factory=list)
    source: str = ""
    
    # Backwards-compatible API

    @property
    def text(self) -> str:
        """Flat text in reading order (joins all element texts)."""
        return "\n\n".join(el.text for el in self.elements)

    def chunks(self, max_chars: int = 1500, overlap: int = 200) -> List[dict]:
        """
        Element-aware chunker used by the parent/child chunker.

        Tables are never split.  Paragraphs and headings are buffered and
        flushed when the buffer would exceed *max_chars*.  Each chunk carries
        a ``pages`` list for citation metadata.

        *overlap* controls how many characters from the end of the previous
        chunk are prepended to the next chunk so context at boundaries is not
        lost.

        Returns a list of dicts: ``{"text": str, "pages": list[int]}``.
        """
        chunks: List[dict] = []
        buf: List[str] = []
        buf_pages: set = set()
        buf_len: int = 0
        # Tail text carried over from the previous chunk for overlap
        _overlap_tail: str = ""

        def flush() -> None:
            nonlocal buf, buf_pages, buf_len, _overlap_tail
            if buf:
                text = "\n\n".join(buf)
                chunks.append({"text": text, "pages": sorted(buf_pages)})
                # Keep the last *overlap* chars of the flushed text as the
                # seed for the next chunk.
                _overlap_tail = text[-overlap:] if overlap > 0 else ""
            buf, buf_pages, buf_len = [], set(), 0

        for el in self.elements:
            piece_len = len(el.text)

            if el.type == "table":
                flush()
                chunks.append({"text": el.text, "pages": [el.page]})
                _overlap_tail = ""  # tables don't seed overlap
                continue

            if buf_len + piece_len > max_chars and buf:
                flush()
                # Prepend overlap tail into the new buffer so boundary context
                # is preserved across the split.
                if _overlap_tail:
                    buf.append(_overlap_tail)
                    buf_pages  # pages stay empty until real elements added
                    buf_len = len(_overlap_tail)

            buf.append(el.text)
            buf_pages.add(el.page)
            buf_len += piece_len

        flush()
        return chunks


# Internal helpers

def _table_to_text(table: List[List[Optional[str]]]) -> str:
    """Render a pdfplumber table as a pipe-delimited text block."""
    rows = [[cell if cell is not None else "" for cell in row] for row in table]
    return "\n".join(" | ".join(row) for row in rows)


def _is_heading(line_size: float, body_size: float) -> bool:
    """Return True if the line's font size is ≥115 % of the page body size."""
    return body_size > 0 and line_size >= body_size * 1.15


def _in_table(word: dict, table_bboxes: List[Tuple]) -> bool:
    """Return True when *word* falls inside any of the given table bounding boxes."""
    for (x0, top, x1, bottom) in table_bboxes:
        if x0 <= word["x0"] <= x1 and top <= word["top"] <= bottom:
            return True
    return False


# PDF loader

def _load_pdf_layout(path: str) -> ParsedDocument:
    """
    Extract structural elements from a PDF using pdfplumber.

    Processing order per page:
    1. Extract words with font-size metadata.
    2. Detect body font size as the median of all word sizes.
    3. Walk lines top-to-bottom; skip words inside table regions.
    4. Classify each line as heading or paragraph text.
    5. Extract tables separately and insert them in document order.
    """
    doc = ParsedDocument(source=path)

    try:
        pdf_file = pdfplumber.open(path)
    except Exception as exc:
        raise DocumentParsingError(f"Cannot open PDF '{path}': {exc}") from exc

    with pdf_file as pdf:
        logger.debug("Parsing PDF '%s' (%d pages)", path, len(pdf.pages))

        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                logger.debug("Page %d is empty, skipping", page_num)
                continue

            body_size: float = median(w["size"] for w in words) if words else 0.0

            # Group words by vertical position (rounded to 0.1 pt)
            lines: dict[float, list] = {}
            for w in words:
                key = round(w["top"], 1)
                lines.setdefault(key, []).append(w)

            table_bboxes = [t.bbox for t in page.find_tables()]

            sorted_tops = sorted(lines.keys())
            para_buf: List[str] = []
            para_start_top: Optional[float] = None
            prev_bottom: Optional[float] = None

            def flush_paragraph(end_top: float) -> None:
                nonlocal para_buf, para_start_top
                if para_buf:
                    text = " ".join(para_buf)
                    doc.elements.append(Element(
                        type="paragraph",
                        text=text,
                        page=page_num,
                        bbox=(0.0, para_start_top or 0.0, float(page.width), end_top),
                    ))
                para_buf.clear()
                para_start_top = None  # type: ignore[assignment]

            for top in sorted_tops:
                line_words = [w for w in lines[top] if not _in_table(w, table_bboxes)]
                if not line_words:
                    continue

                line_text = " ".join(w["text"] for w in line_words)
                line_size = median(w["size"] for w in line_words)
                line_bottom = max(w["bottom"] for w in line_words)
                gap_break = (
                    prev_bottom is not None
                    and (top - prev_bottom) > body_size * 0.8
                )

                if _is_heading(line_size, body_size):
                    flush_paragraph(top)
                    doc.elements.append(Element(
                        type="heading",
                        text=line_text,
                        page=page_num,
                        bbox=(0.0, top, float(page.width), line_bottom),
                    ))
                else:
                    if gap_break:
                        flush_paragraph(top)
                    if para_start_top is None:
                        para_start_top = top
                    para_buf.append(line_text)

                prev_bottom = line_bottom

            flush_paragraph(prev_bottom or 0.0)

            # Tables – extracted in their own pass then merged into document
            # order by comparing each table's top-coordinate against the bbox
            # of every text element already added for this page.  Without this
            # step tables would always appear *after* all text on the page,
            # breaking reading order for mid-page tables.
            page_element_start = len(doc.elements) - sum(
                1 for el in doc.elements if el.page == page_num
            )
            for table in page.find_tables():
                data = table.extract()
                if not data:
                    continue
                table_top = table.bbox[1]  # (x0, top, x1, bottom)
                table_el = Element(
                    type="table",
                    text=_table_to_text(data),
                    page=page_num,
                    bbox=table.bbox,
                    table_data=data,
                )
                # Find the insertion point: first page element whose bbox top
                # is greater than the table's top coordinate.
                insert_at = len(doc.elements)
                for idx in range(page_element_start, len(doc.elements)):
                    if doc.elements[idx].bbox[1] > table_top:
                        insert_at = idx
                        break
                doc.elements.insert(insert_at, table_el)

    logger.info(
        "PDF parsed | source=%s pages=%d elements=%d",
        path,
        len(pdf.pages),
        len(doc.elements),
    )
    return doc


# Text loader

def _load_txt(path: str) -> ParsedDocument:
    """Load a plain-text file as a single paragraph element."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to latin-1 for legacy files
        try:
            text = Path(path).read_text(encoding="latin-1")
        except Exception as exc:
            raise DocumentParsingError(f"Cannot decode text file '{path}': {exc}") from exc
    except OSError as exc:
        raise DocumentParsingError(f"Cannot read file '{path}': {exc}") from exc

    text = text.strip()
    if not text:
        raise DocumentParsingError(f"Text file '{path}' is empty")

    doc = ParsedDocument(source=path)
    doc.elements.append(Element(type="paragraph", text=text, page=1, bbox=(0.0, 0.0, 0.0, 0.0)))
    logger.info("TXT parsed | source=%s chars=%d", path, len(text))
    return doc


# Public API

_SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".doc", ".docx"}


def load_document(path: str) -> ParsedDocument:
    """
    Parse a document at *path* and return a ``ParsedDocument``.

    Supported formats: ``.pdf``, ``.txt``

    Raises
    ------
    DocumentParsingError
        If the file does not exist, is not a supported type, or cannot be parsed.
    """
    file_path = Path(path)

    if not file_path.exists():
        raise DocumentParsingError(f"File does not exist: '{path}'")
    if not file_path.is_file():
        raise DocumentParsingError(f"Path is not a file: '{path}'")

    extension = file_path.suffix.lower()
    if extension not in _SUPPORTED_EXTENSIONS:
        raise DocumentParsingError(
            f"Unsupported file type '{extension}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    logger.info("Loading document | path=%s type=%s", path, extension)

    try:
        if extension == ".pdf" or extension == ".doc" or extension == ".docx":
            return _load_pdf_layout(path)
        else:  # .txt
            return _load_txt(path)
    except DocumentParsingError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error parsing document '%s'", path)
        raise DocumentParsingError(f"Failed to parse document '{file_path.name}': {exc}") from exc


def load_pdf(path: str) -> str:
    """Backwards-compatible helper: return flat text from a PDF."""
    return _load_pdf_layout(path).text
