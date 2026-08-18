from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.ingestion.exceptions import VectorStoreError
from app.runtime_settings import settings as app_settings

logger = logging.getLogger(__name__)


# Result schema
@dataclass
class DenseResult:
    chunk_id: str
    parent_id: str
    document_id: str
    text: str
    score: float                        # cosine similarity in [0, 1] (higher = more similar)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Vector store

class VectorStore:
    """
    ChromaDB-backed dense vector store.

    Parameters
    ----------
    collection_name: Chroma collection name (defaults to ``settings.COLLECTIONS_NAME``).
    path:            Path for PersistentClient storage (defaults to ``settings.CHROMA_DB_PATH``).
    """

    def __init__(
        self,
        collection_name: Optional[str] = None,
        path: Optional[str] = None,
    ) -> None:
        self.collection_name = collection_name or app_settings.COLLECTIONS_NAME
        self.db_path = path or app_settings.CHROMA_DB_PATH
        self._collection: Any = None

        self.client = chromadb.PersistentClient(
            path=self.db_path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        logger.info(
            "VectorStore initialised | collection=%s path=%s",
            self.collection_name,
            self.db_path,
        )

    # Collection management

    def _get_collection(self) -> Any:
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def create_collection(self, vector_size: Optional[int] = None) -> Any:
        """
        Ensure the collection exists.

        If *vector_size* is provided, checks that any existing embeddings
        share that dimension.  Raises ``VectorStoreError`` on mismatch so
        callers receive a clear error rather than silently inserting
        incompatible vectors.
        """
        collection = self._get_collection()

        if vector_size is not None:
            existing = collection.get(limit=1, include=["embeddings"])
            _raw = existing.get("embeddings")
            # ChromaDB may return a NumPy array; avoid bare truthiness check
            # which raises "ambiguous truth value" on empty arrays.
            existing_embeddings = _raw if _raw is not None else []
            if len(existing_embeddings) > 0:
                existing_dim = len(existing_embeddings[0])
                if existing_dim != vector_size:
                    raise VectorStoreError(
                        f"Embedding dimension mismatch in collection '{self.collection_name}'. "
                        f"Existing dimension: {existing_dim}, configured model produces: {vector_size}. "
                        "Delete the collection and re-index all documents to resolve."
                    )

        return collection

    # Mutations

    def add_chunks(self, chunks: list, embeddings: List[List[float]]) -> None:
        """
        Upsert child chunks and their embeddings into the collection.

        Parameters
        ----------
        chunks:     List of ``ChildChunk`` schema objects.
        embeddings: Parallel list of embedding vectors.
        """
        if chunks is None or len(chunks) == 0:
            return
        if embeddings is None or len(embeddings) == 0:
            return
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk count ({len(chunks)}) != embedding count ({len(embeddings)})"
            )

        vector_size = len(embeddings[0])
        collection = self.create_collection(vector_size=vector_size)

        ids: List[str] = []
        vectors: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[dict] = []

        for chunk, embedding in zip(chunks, embeddings):
            if embedding is None or len(embedding) != vector_size:
                got = len(embedding) if embedding is not None else "None"
                raise VectorStoreError(
                    f"Embedding dimension mismatch for chunk '{chunk.id}': "
                    f"expected {vector_size}, got {got}"
                )
            ids.append(str(chunk.id))
            vectors.append(list(embedding))
            documents.append(chunk.text)
            # Flat metadata — Chroma WHERE filters require top-level scalar keys
            metadatas.append({
                "chunk_id": str(chunk.id),
                "parent_id": str(chunk.parent_id),
                "document_id": str(chunk.document_id),
                # Serialise the inner metadata dict as a JSON string so it
                # survives the round-trip without breaking Chroma's type check.
                # json.dumps (not str()) guarantees valid, re-parseable JSON.
                "chunk_metadata": json.dumps(chunk.metadata),
            })

        logger.info(
            "Upserting %d vectors | collection=%s dim=%d",
            len(vectors),
            self.collection_name,
            vector_size,
        )
        collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info("Upsert complete | collection=%s total=%d", self.collection_name, collection.count())

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to *document_id*."""
        collection = self._get_collection()
        # Delete directly by metadata filter — no need to fetch IDs first.
        collection.delete(where={"document_id": str(document_id)})
        logger.info("Deleted chunks for document_id=%s", document_id)

    # Query

    def search(self, query_vector: List[float], limit: int = 20) -> List[DenseResult]:
        """
        Return the *limit* most similar chunks to *query_vector*.

        Parameters
        ----------
        query_vector: Query embedding produced by ``EmbeddingService.embed_query()``.
        limit:        Maximum number of results.

        Returns
        -------
        List of ``DenseResult`` sorted by cosine similarity descending.
        """
        collection = self._get_collection()

        try:
            raw = collection.query(
                query_embeddings=[list(query_vector)],
                n_results=min(limit, collection.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chroma query failed: {exc}") from exc

        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: List[DenseResult] = []
        for i, doc_text in enumerate(documents):
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else 2.0
            # ChromaDB cosine space returns cosine *distance* in [0, 2]
            # (0 = identical vectors, 2 = opposite vectors).
            # Convert to cosine *similarity* in [0, 1]:  sim = 1 - dist / 2
            score = max(0.0, 1.0 - float(dist) / 2.0)
            results.append(DenseResult(
                chunk_id=meta.get("chunk_id", ""),
                parent_id=meta.get("parent_id", ""),
                document_id=meta.get("document_id", ""),
                text=doc_text or "",
                score=score,
                metadata=meta,
            ))

        logger.debug(
            "Dense search complete | limit=%d returned=%d", limit, len(results)
        )
        return results

    # Introspection

    def get_document_chunks(self, document_id: str) -> dict:
        """Return all Chroma records for a given document_id."""
        return self._get_collection().get(
            where={"document_id": str(document_id)}
        )

    def count(self) -> int:
        return self._get_collection().count()

    def health_check(self) -> dict:
        try:
            collection = self._get_collection()
            return {
                "status": "ok",
                "collection": self.collection_name,
                "count": collection.count(),
                "path": self.db_path,
            }
        except Exception as exc:
            logger.exception("Chroma health check failed")
            raise VectorStoreError(f"Chroma health check failed: {exc}") from exc
