from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from rank_bm25 import BM25Okapi

from app.ingestion.exceptions import BM25IndexError
from app.models.schemas import ChildChunk
from app.runtime_settings import settings

logger = logging.getLogger(__name__)


# Result schema

@dataclass
class BM25Result:
    chunk_id: str
    parent_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)



# Index service

class BM25Index:
    """
    Maintains a persistent BM25 index over child chunks.

    The internal state is:
    * ``_chunks``   – list of ChildChunk, ordered as they were inserted
    * ``_doc_ids``  – set of document_ids currently in the index
    * ``_bm25``     – BM25Okapi instance (rebuilt lazily after mutations)

    The index is stored at ``settings.BM25_INDEX_PATH`` and loaded
    automatically on first access after a restart.
    """

    def __init__(self, index_path: Optional[str] = None) -> None:
        self._index_path = Path(index_path or settings.BM25_INDEX_PATH)
        self._chunks: List[ChildChunk] = []
        self._doc_chunk_ids: Dict[str, List[str]] = {}  # document_id → [chunk_id]
        self._bm25: Optional[BM25Okapi] = None
        self._dirty = False  # True when _bm25 must be rebuilt before next search

    # Persistence

    def save(self) -> None:
        """Persist the current index state to disk."""
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "chunks": self._chunks,
            "doc_chunk_ids": self._doc_chunk_ids,
        }
        tmp = self._index_path.with_suffix(".tmp")
        try:
            with open(tmp, "wb") as fh:
                pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(self._index_path)
        except Exception as exc:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise BM25IndexError(f"Failed to save BM25 index to {self._index_path}: {exc}") from exc

        logger.info(
            "BM25 index saved | path=%s chunks=%d documents=%d",
            self._index_path,
            len(self._chunks),
            len(self._doc_chunk_ids),
        )

    def load(self) -> None:
        """Load the persisted index state from disk (if it exists)."""
        if not self._index_path.exists():
            logger.info("No persisted BM25 index found at %s; starting empty", self._index_path)
            return

        try:
            with open(self._index_path, "rb") as fh:
                state = pickle.load(fh)
        except Exception as exc:
            raise BM25IndexError(
                f"Failed to load BM25 index from {self._index_path}: {exc}"
            ) from exc

        self._chunks = state.get("chunks", [])
        self._doc_chunk_ids = state.get("doc_chunk_ids", {})
        self._dirty = True  # rebuild BM25 on next search

        logger.info(
            "BM25 index loaded | path=%s chunks=%d documents=%d",
            self._index_path,
            len(self._chunks),
            len(self._doc_chunk_ids),
        )

    # Internal: BM25 rebuild
    def _rebuild(self) -> None:
        """Rebuild the BM25Okapi object from the current chunk list."""
        if not self._chunks:
            self._bm25 = None
            self._dirty = False
            return

        tokenized = [c.text.lower().split() for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        self._dirty = False

        logger.debug("BM25 rebuilt | corpus_size=%d", len(self._chunks))

    def _ensure_built(self) -> None:
        if self._dirty or self._bm25 is None:
            self._rebuild()

    # Mutations

    def index_documents(self, chunks: List[ChildChunk]) -> None:
        """
        Add or replace child chunks in the index.

        If chunks for a given document_id already exist they are removed
        first (upsert semantics).  This avoids duplicates when a document
        is re-ingested.

        Parameters
        ----------
        chunks: List of ChildChunk dataclass instances.
        """
        if not chunks:
            return

        # Determine which document_ids are being updated
        incoming_doc_ids = {c.document_id for c in chunks}
        for doc_id in incoming_doc_ids:
            if doc_id in self._doc_chunk_ids:
                self._remove_document_chunks(doc_id)

        # Append new chunks
        for chunk in chunks:
            chunk_id = chunk.id
            self._chunks.append(chunk)
            self._doc_chunk_ids.setdefault(chunk.document_id, []).append(chunk_id)

        self._dirty = True

        logger.info(
            "BM25 indexed | new_chunks=%d total_chunks=%d documents=%d",
            len(chunks),
            len(self._chunks),
            len(self._doc_chunk_ids),
        )

    def delete_document(self, document_id: str) -> None:
        """
        Remove all chunks belonging to *document_id* from the index.

        Parameters
        ----------
        document_id: The document whose chunks should be purged.
        """
        if document_id not in self._doc_chunk_ids:
            logger.warning("BM25 delete: document_id '%s' not found in index", document_id)
            return

        removed_count = self._remove_document_chunks(document_id)
        self._dirty = True

        logger.info(
            "BM25 document deleted | document_id=%s removed_chunks=%d",
            document_id,
            removed_count,
        )

    def _remove_document_chunks(self, document_id: str) -> int:
        chunk_ids = set(self._doc_chunk_ids.pop(document_id, []))
        before = len(self._chunks)
        self._chunks = [c for c in self._chunks if c.id not in chunk_ids]
        return before - len(self._chunks)

    # Query

    def search(self, query: str, top_k: int = 20) -> List[BM25Result]:
        """
        Retrieve the top-*k* chunks for *query* using BM25Okapi scoring.

        Parameters
        ----------
        query: Natural-language query string.
        top_k: Maximum number of results to return.

        Returns
        -------
        List of BM25Result sorted by score descending.
        """
        if not query or not query.strip():
            raise BM25IndexError("Query text must not be empty")

        self._ensure_built()

        if self._bm25 is None or len(self._chunks) == 0:
            logger.warning("BM25 search called on an empty index")
            return []

        tokens = query.lower().split()
        try:
            scores = self._bm25.get_scores(tokens)
        except Exception as exc:
            raise BM25IndexError(f"BM25 scoring failed: {exc}") from exc

        # Rank indices by score descending
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        results: List[BM25Result] = []
        for idx in ranked_indices[:top_k]:
            chunk = self._chunks[idx]
            results.append(BM25Result(
                chunk_id=chunk.id,
                parent_id=chunk.parent_id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=float(scores[idx]),
                metadata=chunk.metadata,
            ))

        logger.debug(
            "BM25 search complete | query_tokens=%d results=%d top_score=%.4f",
            len(tokens),
            len(results),
            results[0].score if results else 0.0,
        )
        return results

    # Introspection

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def document_count(self) -> int:
        return len(self._doc_chunk_ids)

    def health_check(self) -> dict:
        return {
            "status": "ok",
            "chunk_count": self.chunk_count,
            "document_count": self.document_count,
            "index_path": str(self._index_path),
            "index_exists": self._index_path.exists(),
        }
