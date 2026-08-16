"""
Parent / child chunker.

Splits a ``ParsedDocument`` into a two-level hierarchy:

  Parent chunks  – larger context windows (PARENT_CHUNK_SIZE words)
  Child chunks   – smaller retrieval units (CHILD_CHUNK_SIZE words)
                   generated from each parent

All IDs are deterministic SHA-256 → UUID5 digests so re-ingesting the
same document produces the same IDs and Chroma can safely upsert.

Sizes default to the values in ``runtime_settings`` but can be overridden
per-call for testing or API-level configuration.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import List, Tuple

from app.ingestion.exceptions import ChunkingError
from app.models.schemas import ChildChunk, ParentChunk

logger = logging.getLogger(__name__)


# ID generation

def _stable_id(namespace: str, text: str) -> str:
    """
    Generate a stable UUID from *namespace* and *text*.

    Uses SHA-256 as the seed for UUID5 so the same inputs always produce
    the same ID regardless of Python hash randomisation.
    """
    digest = hashlib.sha256(f"{namespace}:{text}".encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, digest))


# Word-level text splitter
def split_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[str]:
    """
    Split *text* into overlapping word-level windows.

    Parameters
    ----------
    text:       Input text.
    chunk_size: Maximum number of words per chunk.
    overlap:    Number of words shared between consecutive chunks.

    Returns
    -------
    List of non-empty strings.
    """
    if not text or chunk_size <= 0:
        return []

    words = text.split()
    if not words:
        return []

    if overlap >= chunk_size:
        logger.warning(
            "overlap (%d) >= chunk_size (%d); clamping overlap to chunk_size - 1",
            overlap,
            chunk_size,
        )
        overlap = chunk_size - 1

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        piece = " ".join(words[start:end]).strip()
        if piece:
            chunks.append(piece)
        if end == len(words):
            break
        start += step

    return chunks


# Parent / child hierarchy builder
def create_parent_child_chunks(
    document_id: str,
    parsed_doc: object,
    parent_size: int = 1000,
    child_size: int = 300,
    overlap: int = 50,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """
    Build a two-level chunk hierarchy from *parsed_doc*.

    Parameters
    ----------
    document_id:  Stable identifier for the source document.
    parsed_doc:   Any object with a ``chunks(max_chars)`` method that returns
                  ``[{"text": str, "pages": list[int]}, ...]``.
    parent_size:  Maximum character length passed to ``parsed_doc.chunks()``.
    child_size:   Word-level window size for child splits.
    overlap:      Word-level overlap between sibling children.

    Returns
    -------
    (parents, children) tuple.

    Raises
    ------
    ChunkingError  – on validation failures or unexpected errors.
    """
    if parent_size <= 0:
        raise ChunkingError(f"parent_size must be > 0, got {parent_size}")
    if child_size <= 0:
        raise ChunkingError(f"child_size must be > 0, got {child_size}")
    if overlap < 0:
        raise ChunkingError(f"overlap must be >= 0, got {overlap}")
    if not document_id:
        raise ChunkingError("document_id must not be empty")

    parents: List[ParentChunk] = []
    children: List[ChildChunk] = []

    try:
        raw_chunks = parsed_doc.chunks(max_chars=parent_size)
    except Exception as exc:
        raise ChunkingError(f"Failed to obtain parent chunks from document: {exc}") from exc

    for parent_index, raw in enumerate(raw_chunks):
        parent_text = (raw.get("text") or "").strip()
        if not parent_text:
            logger.debug(
                "Skipping empty parent chunk at index %d for document %s",
                parent_index,
                document_id,
            )
            continue

        parent_id = _stable_id(f"{document_id}:parent:{parent_index}", parent_text)
        pages = raw.get("pages", [])

        parents.append(ParentChunk(
            id=parent_id,
            document_id=document_id,
            text=parent_text,
            metadata={"pages": pages, "index": parent_index},
        ))

        child_texts = split_text(parent_text, chunk_size=child_size, overlap=overlap)
        if not child_texts:
            logger.warning(
                "Parent chunk %s produced no children (text length=%d)",
                parent_id,
                len(parent_text),
            )
            continue

        for child_index, child_text in enumerate(child_texts):
            child_id = _stable_id(
                f"{document_id}:child:{parent_id}:{child_index}",
                child_text,
            )
            children.append(ChildChunk(
                id=child_id,
                parent_id=parent_id,
                document_id=document_id,
                text=child_text,
                metadata={
                    "pages": pages,
                    "parent_index": parent_index,
                    "child_index": child_index,
                },
            ))

    logger.info(
        "Chunking complete | document_id=%s parents=%d children=%d",
        document_id,
        len(parents),
        len(children),
    )
    return parents, children
