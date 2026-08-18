from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from app.ingestion.bm25_index import BM25Index
from app.ingestion.embedding_service import EmbeddingService
from app.retrieval.dense_search import VectorStore
from app.retrieval.fusion import FusedResult, RRFFusion
from app.runtime_settings import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Executes dense and sparse retrieval in parallel and fuses the results.

    Parameters
    ----------
    embedding_service:  Pre-constructed ``EmbeddingService``.
    vector_store:       Pre-constructed ``VectorStore`` (ChromaDB).
    bm25_index:         Pre-constructed ``BM25Index``.
    dense_top_k:        Number of candidates from dense retrieval.
    bm25_top_k:         Number of candidates from BM25 retrieval.
    rrf_k:              RRF smoothing constant.
    hybrid_top_k:       Final number of fused results returned.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        dense_top_k: Optional[int] = None,
        bm25_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        hybrid_top_k: Optional[int] = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.dense_top_k = dense_top_k if dense_top_k is not None else settings.DENSE_TOP_K
        self.bm25_top_k = bm25_top_k if bm25_top_k is not None else settings.BM25_TOP_K
        self.rrf_k = rrf_k if rrf_k is not None else settings.RRF_K
        self.hybrid_top_k = hybrid_top_k if hybrid_top_k is not None else settings.HYBRID_TOP_K
        self._fusion = RRFFusion(k=self.rrf_k, top_k=self.hybrid_top_k)

    def search(self, query: str) -> List[FusedResult]:
        """
        Run hybrid retrieval for *query* and return fused results.

        Parameters
        ----------
        query: Natural-language query string.

        Returns
        -------
        List of ``FusedResult`` sorted by RRF score descending.
        """
        if not query or not query.strip():
            logger.warning("HybridRetriever received an empty query")
            return []

        t0 = time.perf_counter()

        # Dense and BM25 retrieval run concurrently — neither depends on the
        # other's output, so parallelising them reduces end-to-end latency.
        dense_results = []
        bm25_results = []
        dense_latency = 0.0
        bm25_latency = 0.0

        def _run_dense() -> tuple:
            t = time.perf_counter()
            qv = self.embedding_service.embed_query(query)
            results = self.vector_store.search(qv, limit=self.dense_top_k)
            return results, time.perf_counter() - t

        def _run_bm25() -> tuple:
            t = time.perf_counter()
            results = self.bm25_index.search(query, top_k=self.bm25_top_k)
            return results, time.perf_counter() - t

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(_run_dense)
            future_bm25 = executor.submit(_run_bm25)
            dense_results, dense_latency = future_dense.result()
            bm25_results, bm25_latency = future_bm25.result()

        # RRF fusion
        t_fuse = time.perf_counter()
        fused = self._fusion.fuse(dense_results, bm25_results)
        fuse_latency = time.perf_counter() - t_fuse

        total_latency = time.perf_counter() - t0
        logger.info(
            "Hybrid retrieval | dense=%d bm25=%d fused=%d "
            "dense_ms=%.1f bm25_ms=%.1f fuse_ms=%.1f total_ms=%.1f",
            len(dense_results),
            len(bm25_results),
            len(fused),
            dense_latency * 1000,
            bm25_latency * 1000,
            fuse_latency * 1000,
            total_latency * 1000,
        )
        return fused
