from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
import pdfplumber


@dataclass
class Element:
    type: str          # "heading" | "paragraph" | "table"
    text: str          # for tables: a markdown-ish rendering
    page: int           # 1-indexed
    bbox: tuple[float, float, float, float]  # (x0, top, x1, bottom)
    table_data: list[list[str]] | None = None  # raw cells, only for type=="table"


@dataclass
class ParsedDocument:
    elements: list[Element] = field(default_factory=list)
    source: str = ""

    @property
    def text(self) -> str:
        """Flat text, in reading order. Backwards-compatible with old API."""
        return "\n\n".join(el.text for el in self.elements)

    def chunks(self, max_chars: int = 1500, overlap: int = 200) -> list[dict]:
        """
        Simple element-aware chunker for RAG ingestion.
        Tables are never split; paragraphs are merged up to max_chars.
        Each chunk carries page range metadata for citation.
        """
        chunks = []
        buf, buf_pages, buf_len = [], set(), 0

        def flush():
            nonlocal buf, buf_pages, buf_len
            if buf:
                chunks.append({
                    "text": "\n\n".join(buf),
                    "pages": sorted(buf_pages),
                })
            buf, buf_pages, buf_len = [], set(), 0

        for el in self.elements:
            piece_len = len(el.text)

            if el.type == "table":
                flush()
                chunks.append({"text": el.text, "pages": [el.page]})
                continue

            if buf_len + piece_len > max_chars and buf:
                flush()

            buf.append(el.text)
            buf_pages.add(el.page)
            buf_len += piece_len

        flush()
        return chunks


def _table_to_text(table: list[list[str | None]]) -> str:
    rows = [[c if c is not None else "" for c in row] for row in table]
    return "\n".join(" | ".join(row) for row in rows)


def _is_heading(word_sizes: list[float], line_size: float, body_size: float) -> bool:
    # Heuristic: line font size noticeably larger than the page's typical body text
    return body_size > 0 and line_size >= body_size * 1.15


def load_pdf_layout(path: str) -> ParsedDocument:
    doc = ParsedDocument(source=path)

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue

            body_size = median(w["size"] for w in words) if words else 0

            # Group words into lines by their vertical position ("top")
            lines: dict[float, list[dict]] = {}
            for w in words:
                key = round(w["top"], 1)
                lines.setdefault(key, []).append(w)

            # Mark which regions belong to tables so we skip them in the
            # word-based pass and avoid duplicating table cells as prose
            table_bboxes = [t.bbox for t in page.find_tables()]

            def in_table(word) -> bool:
                for (x0, top, x1, bottom) in table_bboxes:
                    if x0 <= word["x0"] <= x1 and top <= word["top"] <= bottom:
                        return True
                return False

            sorted_tops = sorted(lines.keys())
            paragraph_buf, para_start_top = [], None

            def flush_paragraph(end_top):
                nonlocal paragraph_buf, para_start_top
                if paragraph_buf:
                    text = " ".join(paragraph_buf)
                    doc.elements.append(Element(
                        type="paragraph",
                        text=text,
                        page=page_num,
                        bbox=(0, para_start_top or 0, page.width, end_top),
                    ))
                paragraph_buf, para_start_top = [], None

            prev_bottom = None
            for top in sorted_tops:
                line_words = [w for w in lines[top] if not in_table(w)]
                if not line_words:
                    continue

                line_text = " ".join(w["text"] for w in line_words)
                line_size = median(w["size"] for w in line_words)
                line_bottom = max(w["bottom"] for w in line_words)

                # Large vertical gap since the previous line -> new paragraph
                gap_break = prev_bottom is not None and (top - prev_bottom) > body_size * 0.8

                if _is_heading([w["size"] for w in line_words], line_size, body_size):
                    flush_paragraph(top)
                    doc.elements.append(Element(
                        type="heading",
                        text=line_text,
                        page=page_num,
                        bbox=(0, top, page.width, line_bottom),
                    ))
                else:
                    if gap_break:
                        flush_paragraph(top)
                    if para_start_top is None:
                        para_start_top = top
                    paragraph_buf.append(line_text)

                prev_bottom = line_bottom

            flush_paragraph(prev_bottom or 0)

            # Tables, extracted in their own pass, inserted in document order
            for table in page.find_tables():
                data = table.extract()
                doc.elements.append(Element(
                    type="table",
                    text=_table_to_text(data),
                    page=page_num,
                    bbox=table.bbox,
                    table_data=data,
                ))

    return doc


def load_pdf(path: str) -> str:
    """Backwards-compatible: flat text only."""
    return load_pdf_layout(path).text


def load_document(path: str) -> ParsedDocument:
    extension = Path(path).suffix.lower()
    if extension == ".pdf":
        return load_pdf_layout(path)
    elif extension == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        doc = ParsedDocument(source=path)
        doc.elements.append(Element(type="paragraph", text=text, page=1, bbox=(0, 0, 0, 0)))
        return doc
    else:
        raise ValueError(f"Unsupported file type: {extension}")