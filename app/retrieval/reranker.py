from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.ingestion.exceptions import RerankerError
from app.runtime_settings import settings
from sentence_transformers import CrossEncoder 

logger = logging.getLogger(__name__)


# Result schema

@dataclass
class RerankedResult:
    chunk_id: str
    parent_id: str
    document_id: str
    text: str
    score: float          # rerank_score (primary sort key after reranking)
    rrf_score: float = 0.0
    dense_score: float = 0.0
    bm25_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# Reranker

class Reranker:
    """
    Reranks candidate chunks using a BGE cross-encoder.

    Parameters
    ----------
    model_name:  HuggingFace model ID (defaults to ``settings.RERANKER_MODEL``).
    device:      Torch device string, e.g. ``"cpu"`` or ``"cuda"``
                 (defaults to ``settings.RERANKER_DEVICE``).
    top_k:       Maximum candidates to return after reranking.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.RERANKER_MODEL
        self.device = device or settings.RERANKER_DEVICE
        self.top_k = top_k if top_k is not None else settings.RERANK_TOP_K
        self._model: Optional[Any] = None
        self._lock = threading.Lock()

        logger.info(
            "Reranker configured | model=%s device=%s top_k=%d",
            self.model_name,
            self.device,
            self.top_k,
        )

    # Lazy model loading

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        with self._lock:
            if self._model is not None:  # double-checked locking
                return self._model

            logger.info("Loading reranker model '%s' on device '%s'", self.model_name, self.device)
            t0 = time.perf_counter()
            try:
                self._model = CrossEncoder(self.model_name, device=self.device)
            except Exception as exc:
                logger.exception("Failed to load reranker model '%s'", self.model_name)
                raise RerankerError(
                    f"Cannot load reranker model '{self.model_name}': {exc}"
                ) from exc

            logger.info(
                "Reranker model loaded | model=%s device=%s load_ms=%.1f",
                self.model_name,
                self.device,
                (time.perf_counter() - t0) * 1000,
            )
        return self._model

    # Public API

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_k: Optional[int] = None,
    ) -> List[RerankedResult]:
        """
        Rerank *candidates* against *query* using the cross-encoder.

        Parameters
        ----------
        query:      Natural-language query.
        candidates: List of objects with ``chunk_id``, ``parent_id``,
                    ``document_id``, ``text``, and score attributes
                    (compatible with ``FusedResult``).
        top_k:      Override for the number of results (defaults to
                    ``self.top_k``).

        Returns
        -------
        List of ``RerankedResult`` sorted by ``score`` descending.
        """
        if not candidates:
            logger.debug("Reranker received empty candidate list; returning []")
            return []

        effective_top_k = top_k if top_k is not None else self.top_k

        model = self._get_model()
        pairs = [(query, getattr(c, "text", "")) for c in candidates]

        t0 = time.perf_counter()
        try:
            raw_scores = model.predict(pairs)
        except Exception as exc:
            logger.exception("Reranker predict() failed")
            raise RerankerError(f"Reranking failed: {exc}") from exc

        latency_ms = (time.perf_counter() - t0) * 1000

        results: List[RerankedResult] = []
        for candidate, raw_score in zip(candidates, raw_scores):
            results.append(RerankedResult(
                chunk_id=getattr(candidate, "chunk_id", ""),
                parent_id=getattr(candidate, "parent_id", ""),
                document_id=getattr(candidate, "document_id", ""),
                text=getattr(candidate, "text", ""),
                score=float(raw_score),
                rrf_score=float(getattr(candidate, "rrf_score", 0.0)),
                dense_score=float(getattr(candidate, "dense_score", 0.0)),
                bm25_score=float(getattr(candidate, "bm25_score", 0.0)),
                metadata=getattr(candidate, "metadata", {}),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        results = results[:effective_top_k]

        logger.info(
            "Reranking complete | candidates=%d returned=%d top_score=%.4f latency_ms=%.1f",
            len(candidates),
            len(results),
            results[0].score if results else 0.0,
            latency_ms,
        )
        return results
