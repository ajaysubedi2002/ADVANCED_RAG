"""
Backwards-compatibility shim.

The embedding implementation has moved to ``app.ingestion.embedding_service``.
``EmbeddingModel`` is re-exported here so old imports continue to work.
"""
from app.ingestion.embedding_service import EmbeddingService as EmbeddingModel  # noqa: F401

__all__ = ["EmbeddingModel"]
