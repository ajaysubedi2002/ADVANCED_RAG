"""
Backwards-compatibility shim.

The vector store implementation has moved to ``app.retrieval.dense_search``.
``VectorStore`` is re-exported here so old imports continue to work.
"""
from app.retrieval.dense_search import VectorStore  # noqa: F401

__all__ = ["VectorStore"]
