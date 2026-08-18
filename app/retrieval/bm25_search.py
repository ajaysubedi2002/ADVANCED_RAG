"""
Backwards-compatibility shim.

The BM25 implementation has moved to ``app.ingestion.bm25_index``.
This file re-exports ``BM25Index`` under the old name ``BM25Retriever``
so any existing imports continue to work while we transition.
"""
from app.ingestion.bm25_index import BM25Index as BM25Retriever  # noqa: F401

__all__ = ["BM25Retriever"]
