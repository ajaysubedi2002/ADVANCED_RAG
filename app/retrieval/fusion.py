from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.runtime_settings import settings

logger = logging.getLogger(__name__)


# Result schema
@dataclass
class FusedResult:
    chunk_id: str
    parent_id: str
    document_id: str
    text: str
    rrf_score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    dense_rank: Optional[int] = None
    bm25_rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Intermediate normalised record (used internally)

@dataclass
class _RankedItem:
    chunk_id: str
    parent_id: str
    document_id: str
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# Fusion logic

class RRFFusion:
    """
    Combines two ranked lists (dense + BM25) using Reciprocal Rank Fusion.

    Parameters
    ----------
    k:      RRF smoothing constant (default ``settings.RRF_K``).
    top_k:  Maximum results to return after fusion.
    """

    def __init__(
        self,
        k: Optional[int] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.k = k if k is not None else settings.RRF_K
        self.top_k = top_k if top_k is not None else settings.HYBRID_TOP_K

    def fuse(
        self,
        dense_results: List[Any],
        bm25_results: List[Any],
    ) -> List[FusedResult]:
        """
        Fuse *dense_results* and *bm25_results* into a single ranked list.

        Both argument lists must contain objects with at least the attributes:
          chunk_id, parent_id, document_id, text, score, metadata

        (The ``DenseResult`` and ``BM25Result`` dataclasses satisfy this.)

        Returns
        -------
        List of ``FusedResult`` sorted by ``rrf_score`` descending,
        limited to ``self.top_k``.
        """
        # Accumulator: chunk_id → accumulated state
        acc: Dict[str, dict] = {}

        def _ensure(chunk_id: str, item: Any) -> None:
            if chunk_id not in acc:
                acc[chunk_id] = {
                    "chunk_id": chunk_id,
                    "parent_id": getattr(item, "parent_id", ""),
                    "document_id": getattr(item, "document_id", ""),
                    "text": getattr(item, "text", ""),
                    "rrf_score": 0.0,
                    "dense_score": 0.0,
                    "bm25_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None,
                    "metadata": getattr(item, "metadata", {}),
                }

        # Dense list contribution
        for rank, item in enumerate(dense_results, start=1):
            cid = item.chunk_id
            _ensure(cid, item)
            acc[cid]["rrf_score"] += 1.0 / (self.k + rank)
            acc[cid]["dense_score"] = float(getattr(item, "score", 0.0))
            acc[cid]["dense_rank"] = rank

        # BM25 list contribution
        for rank, item in enumerate(bm25_results, start=1):
            cid = item.chunk_id
            _ensure(cid, item)
            acc[cid]["rrf_score"] += 1.0 / (self.k + rank)
            acc[cid]["bm25_score"] = float(getattr(item, "score", 0.0))
            acc[cid]["bm25_rank"] = rank

        # Sort by RRF score descending
        sorted_items = sorted(
            acc.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        results = [
            FusedResult(
                chunk_id=item["chunk_id"],
                parent_id=item["parent_id"],
                document_id=item["document_id"],
                text=item["text"],
                rrf_score=item["rrf_score"],
                dense_score=item["dense_score"],
                bm25_score=item["bm25_score"],
                dense_rank=item["dense_rank"],
                bm25_rank=item["bm25_rank"],
                metadata=item["metadata"],
            )
            for item in sorted_items[: self.top_k]
        ]

        logger.debug(
            "RRF fusion complete | dense_in=%d bm25_in=%d fused_out=%d k=%d",
            len(dense_results),
            len(bm25_results),
            len(results),
            self.k,
        )
        return results
