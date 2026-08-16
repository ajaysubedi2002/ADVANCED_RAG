"""
Embedding service – thin, validated wrapper around Ollama embeddings.

Design decisions
----------------
* No module-level singleton that calls Ollama on import.  Callers must
  instantiate ``EmbeddingService`` explicitly (or use the DI helpers in
  ``app/services/document_service.py``).
* All array truth-value checks use ``len(x) == 0`` or ``x is None`` – never
  ``if embeddings:`` which raises an ambiguous truth-value error for NumPy.
* Embedding dimensions are validated on every call so mismatches are caught
  early rather than silently inserted into ChromaDB.
* Batch support: ``embed_documents`` accepts any number of texts and handles
  list normalisation from Ollama's response schema.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

from ollama import Client

from app.ingestion.exceptions import EmbeddingServiceError
from app.runtime_settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Produce dense embedding vectors via Ollama.

    Parameters
    ----------
    model_name:    Ollama model tag (defaults to ``settings.EMBED_MODEL``).
    host:          Ollama base URL (defaults to ``settings.OLLAMA_HOST``).
    client:        Optional pre-constructed ``ollama.Client`` (for testing).
    expected_dim:  If provided, every call validates returned dimensions match.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        client: Optional[Any] = None,
        expected_dim: Optional[int] = None,
    ) -> None:
        self.model_name = model_name or settings.EMBED_MODEL
        self.host = host or settings.OLLAMA_HOST
        self.client: Any = client or Client(host=self.host)
        self.expected_dim = expected_dim

        logger.info(
            "EmbeddingService initialised | model=%s host=%s expected_dim=%s",
            self.model_name,
            self.host,
            self.expected_dim,
        )

    # Internal helpers

    def _call_ollama(self, inputs: List[str]) -> List[List[float]]:
        """
        Call the Ollama ``/api/embed`` endpoint and return a list of vectors.

        Handles both ``{"embeddings": [[…]]}`` and the legacy single-vector
        ``{"embedding": […]}`` response shapes.
        """
        try:
            response = self.client.embed(model=self.model_name, input=inputs)
        except Exception as exc:
            logger.exception(
                "Ollama embed call failed | model=%s host=%s", self.model_name, self.host
            )
            raise EmbeddingServiceError(
                f"Cannot reach Ollama embedding service at {self.host}: {exc}"
            ) from exc

        # Normalise response
        if isinstance(response, dict):
            raw = response.get("embeddings")
            if raw is None:
                # Legacy single-vector format
                single = response.get("embedding")
                if single is not None:
                    raw = [single]

        else:
            # Some SDK versions return the object directly
            raw = getattr(response, "embeddings", None)

        if raw is None or len(raw) == 0:
            raise EmbeddingServiceError(
                "Ollama embedding service returned no embeddings"
            )

        # Normalise: if the first element is a float, it's a flat vector
        if isinstance(raw[0], float):
            raw = [raw]

        # Convert to plain Python lists and strip None values
        vectors: List[List[float]] = []
        for idx, vec in enumerate(raw):
            if vec is None:
                raise EmbeddingServiceError(f"Embedding at index {idx} is None")
            as_list = list(vec)
            if len(as_list) == 0:
                raise EmbeddingServiceError(f"Embedding at index {idx} is empty")
            vectors.append(as_list)

        return vectors

    def _validate_dimensions(self, vectors: List[List[float]]) -> None:
        """Ensure all vectors share the same dimension and match expected_dim."""
        if len(vectors) == 0:
            raise EmbeddingServiceError("No vectors to validate")

        dims = {len(v) for v in vectors}
        if len(dims) != 1:
            raise EmbeddingServiceError(
                f"Inconsistent embedding dimensions in batch: {dims}"
            )

        actual_dim = next(iter(dims))
        if self.expected_dim is not None and actual_dim != self.expected_dim:
            raise EmbeddingServiceError(
                f"Embedding dimension mismatch: expected {self.expected_dim}, "
                f"received {actual_dim} (model={self.model_name})"
            )

        logger.debug(
            "Dimension validation passed | model=%s dim=%d count=%d",
            self.model_name,
            actual_dim,
            len(vectors),
        )

    # Public API

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Embed a batch of document texts.

        Parameters
        ----------
        texts: Non-empty sequence of strings to embed.

        Returns
        -------
        List of embedding vectors (one per input text).

        Raises
        ------
        EmbeddingServiceError
        """
        if texts is None or len(texts) == 0:
            return []

        texts_list = list(texts)
        logger.info(
            "Embedding documents | model=%s host=%s count=%d",
            self.model_name,
            self.host,
            len(texts_list),
        )

        vectors = self._call_ollama(texts_list)

        if len(vectors) != len(texts_list):
            raise EmbeddingServiceError(
                f"Expected {len(texts_list)} embeddings, received {len(vectors)}"
            )

        self._validate_dimensions(vectors)

        logger.info(
            "Document embeddings created | model=%s count=%d dim=%d",
            self.model_name,
            len(vectors),
            len(vectors[0]),
        )
        return vectors

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Parameters
        ----------
        query: Non-empty query text.

        Returns
        -------
        Single embedding vector.

        Raises
        ------
        EmbeddingServiceError
        """
        if not query or not query.strip():
            raise EmbeddingServiceError("Query text must not be empty")

        logger.info(
            "Embedding query | model=%s host=%s", self.model_name, self.host
        )

        vectors = self._call_ollama([query])
        self._validate_dimensions(vectors)

        vector = vectors[0]
        logger.info(
            "Query embedding created | model=%s dim=%d", self.model_name, len(vector)
        )
        return vector

    def get_dimension(self) -> int:
        """
        Return the embedding dimension by encoding a single test token.
        Useful for validating ChromaDB collection compatibility at startup.
        """
        probe = self.embed_query("dimension probe")
        return len(probe)
