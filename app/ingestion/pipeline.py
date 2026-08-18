"""
Full document ingestion pipeline.

Sequence
--------
1.  Validate file existence and type.
2.  Mark document PROCESSING in PostgreSQL (create or update).
3.  Parse the document → ``ParsedDocument``.
4.  Create parent / child chunks (hierarchical).
5.  Generate embeddings for child chunks only.
6.  Delete stale vectors from ChromaDB (re-index safety).
7.  Upsert child embeddings into ChromaDB.
8.  Delete stale entries from BM25 index (re-index safety).
9.  Upsert child chunks into BM25 index; persist to disk.
10. Replace Document, ParentChunk, ChildChunk in PostgreSQL atomically
    (delete-old → bulk-insert-new).
11. Mark document COMPLETED in PostgreSQL.
12. Return ``IngestionSummary`` with document_id, chunk counts, and timing.

Status transitions
------------------
  PENDING / (existing status)
       ↓  [step 2]
  PROCESSING
       ↓  [all stores succeed]
  COMPLETED
       ↓  [any store fails]
  FAILED

Cross-store consistency
-----------------------
``transaction.atomic()`` covers only PostgreSQL.  ChromaDB and BM25 are
external services that cannot be rolled back by Django.

The chosen strategy is **delete-before-write** for all three stores:

  1. Delete old Chroma vectors for this document_id  (idempotent on the index)
  2. Write new Chroma vectors
  3. Delete old BM25 entries for this document_id   (idempotent on the index)
  4. Write new BM25 entries
  5. Inside a single transaction.atomic():
       a. Delete old ParentChunk rows (CASCADE removes ChildChunk rows)
       b. bulk_create new ParentChunk rows
       c. bulk_create new ChildChunk rows
       d. Set Document.status = COMPLETED

If step 2 or 3 raises, we set status = FAILED and re-raise.  Chroma/BM25
may have been partially written, but the document stays FAILED in Postgres,
so retrieval skips it and an operator can retry.

If step 5 raises, Chroma/BM25 already contain the new vectors.  Postgres
rolls back to the previous (stale) state.  Status is set to FAILED.  The
Chroma/BM25 state is newer than Postgres, but because the document is FAILED
no query should surface it.  A retry will clean everything up.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from app.ingestion.bm25_index import BM25Index
from app.ingestion.chunker import create_parent_child_chunks
from app.ingestion.embedding_service import EmbeddingService
from app.ingestion.exceptions import (
    BM25IndexError,
    ChunkingError,
    DocumentParsingError,
    EmbeddingServiceError,
    VectorStoreError,
)
from app.ingestion.parser import load_document
from app.models.schemas import ChildChunk, ParentChunk
from app.retrieval.dense_search import VectorStore
from app.runtime_settings import settings
from django.db import transaction
from app.models.documents import (
    ChildChunk as ChildChunkORM,
    Document as DocumentORM,
    ParentChunk as ParentChunkORM,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary schema
# ---------------------------------------------------------------------------

@dataclass
class IngestionSummary:
    document_id: str
    filename: str
    file_type: str
    parent_count: int = 0
    child_count: int = 0
    embedding_dim: int = 0
    parse_ms: float = 0.0
    chunk_ms: float = 0.0
    embed_ms: float = 0.0
    index_ms: float = 0.0
    total_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class IngestionPipeline:
    """
    Coordinates the full document ingestion workflow.

    Parameters
    ----------
    document_id:        Stable ID for this document; generated if omitted.
    embedding_service:  Injected ``EmbeddingService`` instance.
    vector_store:       Injected ``VectorStore`` (ChromaDB).
    bm25_index:         Injected ``BM25Index``.
    """

    def __init__(
        self,
        document_id: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
    ) -> None:
        self.document_id = document_id or str(uuid.uuid4())
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._bm25_index = bm25_index

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def ingest_file(
        self,
        file_path: str,
        parent_size: int = settings.PARENT_CHUNK_SIZE,
        child_size: int = settings.CHILD_CHUNK_SIZE,
        overlap: int = settings.CHUNK_OVERLAP,
    ) -> IngestionSummary:
        """
        Run the full ingestion pipeline for *file_path*.

        Parameters
        ----------
        file_path:   Absolute path to a ``.pdf`` or ``.txt`` file.
        parent_size: Max characters per parent chunk.
        child_size:  Max words per child chunk.
        overlap:     Word-level overlap between sibling children.

        Returns
        -------
        ``IngestionSummary``
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise DocumentParsingError(f"File not found: '{file_path}'")

        t_total = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 1 – Mark PROCESSING
        # ------------------------------------------------------------------
        logger.info(
            "Ingestion stage 1/6: mark PROCESSING | document_id=%s file=%s",
            self.document_id,
            path.name,
        )
        self._mark_processing(path)

        # ------------------------------------------------------------------
        # Stage 2 – Parse
        # ------------------------------------------------------------------
        logger.info(
            "Ingestion stage 2/6: parse | document_id=%s", self.document_id
        )
        t0 = time.perf_counter()
        try:
            parsed_doc = load_document(str(path))
        except DocumentParsingError:
            self._mark_failed()
            raise
        except Exception as exc:
            self._mark_failed()
            logger.exception("Unexpected parse error for %s", file_path)
            raise DocumentParsingError(
                f"Failed to parse '{path.name}': {exc}"
            ) from exc
        parse_ms = (time.perf_counter() - t0) * 1000

        if not parsed_doc.elements:
            self._mark_failed()
            raise DocumentParsingError(
                f"Document '{path.name}' produced no parseable content"
            )

        # ------------------------------------------------------------------
        # Stage 3 – Chunk
        # ------------------------------------------------------------------
        logger.info(
            "Ingestion stage 3/6: chunk | parent_size=%d child_size=%d overlap=%d",
            parent_size,
            child_size,
            overlap,
        )
        t0 = time.perf_counter()
        try:
            parents, children = create_parent_child_chunks(
                document_id=self.document_id,
                parsed_doc=parsed_doc,
                parent_size=parent_size,
                child_size=child_size,
                overlap=overlap,
            )
        except ChunkingError:
            self._mark_failed()
            raise
        except Exception as exc:
            self._mark_failed()
            raise ChunkingError(
                f"Chunking failed for '{path.name}': {exc}"
            ) from exc
        chunk_ms = (time.perf_counter() - t0) * 1000

        if not children:
            logger.warning(
                "No child chunks produced for document_id=%s", self.document_id
            )
            # Nothing to index — persist empty state and mark COMPLETED.
            self._replace_db_chunks(path, parents=[], children=[])
            return IngestionSummary(
                document_id=self.document_id,
                filename=path.name,
                file_type=path.suffix.lower().lstrip("."),
                parent_count=len(parents),
                child_count=0,
                parse_ms=round(parse_ms, 1),
                chunk_ms=round(chunk_ms, 1),
                total_ms=round((time.perf_counter() - t_total) * 1000, 1),
                metadata={"source_path": str(path)},
            )

        # ------------------------------------------------------------------
        # Stage 4 – Embed
        # ------------------------------------------------------------------
        embedding_dim = 0
        embed_ms = 0.0
        embeddings: List[List[float]] = []

        if self._embedding_service is not None:
            logger.info(
                "Ingestion stage 4/6: embed | child_count=%d", len(children)
            )
            t0 = time.perf_counter()
            try:
                embeddings = self._embedding_service.embed_documents(
                    [c.text for c in children]
                )
            except EmbeddingServiceError:
                self._mark_failed()
                raise
            except Exception as exc:
                self._mark_failed()
                raise EmbeddingServiceError(
                    f"Embedding failed: {exc}"
                ) from exc
            embed_ms = (time.perf_counter() - t0) * 1000
            embedding_dim = len(embeddings[0]) if embeddings else 0
            logger.info(
                "Embeddings created | count=%d dim=%d embed_ms=%.1f",
                len(embeddings),
                embedding_dim,
                embed_ms,
            )
        else:
            logger.warning(
                "No EmbeddingService provided; skipping embedding stage"
            )

        index_ms_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Stage 5a – ChromaDB: delete stale vectors, then insert new ones
        # ------------------------------------------------------------------
        if self._vector_store is not None and embeddings:
            logger.info(
                "Ingestion stage 5a/6: ChromaDB upsert | collection=%s",
                self._vector_store.collection_name,
            )
            try:
                # Remove old vectors for this document so re-indexing never
                # leaves stale embeddings behind.
                self._vector_store.delete_document(self.document_id)
                self._vector_store.create_collection(vector_size=embedding_dim)
                self._vector_store.add_chunks(children, embeddings)
            except VectorStoreError:
                self._mark_failed()
                raise
            except Exception as exc:
                self._mark_failed()
                raise VectorStoreError(
                    f"Vector store insertion failed: {exc}"
                ) from exc
        else:
            logger.warning(
                "Skipping vector store stage (no store or no embeddings)"
            )

        # ------------------------------------------------------------------
        # Stage 5b – BM25: delete stale entries, then insert new ones
        # ------------------------------------------------------------------
        if self._bm25_index is not None:
            logger.info(
                "Ingestion stage 5b/6: BM25 index | chunks=%d", len(children)
            )
            try:
                # BM25Index.index_documents() already removes old entries for
                # the same document_id before inserting (upsert semantics),
                # so an explicit delete_document() call is not required here.
                # We call it anyway to make the intent explicit and robust
                # against any future changes to index_documents().
                self._bm25_index.delete_document(self.document_id)
                self._bm25_index.index_documents(children)
                self._bm25_index.save()
            except BM25IndexError:
                self._mark_failed()
                raise
            except Exception as exc:
                self._mark_failed()
                raise BM25IndexError(
                    f"BM25 indexing failed: {exc}"
                ) from exc
        else:
            logger.warning("Skipping BM25 stage (no BM25Index provided)")

        index_ms = (time.perf_counter() - index_ms_start) * 1000

        # ------------------------------------------------------------------
        # Stage 6 – PostgreSQL: atomic replace + mark COMPLETED
        # ------------------------------------------------------------------
        logger.info("Ingestion stage 6/6: persist to PostgreSQL")
        self._replace_db_chunks(path, parents, children)

        total_ms = (time.perf_counter() - t_total) * 1000

        logger.info(
            "Ingestion complete | document_id=%s parents=%d children=%d "
            "parse_ms=%.1f chunk_ms=%.1f embed_ms=%.1f index_ms=%.1f total_ms=%.1f",
            self.document_id,
            len(parents),
            len(children),
            parse_ms,
            chunk_ms,
            embed_ms,
            index_ms,
            total_ms,
        )

        return IngestionSummary(
            document_id=self.document_id,
            filename=path.name,
            file_type=path.suffix.lower().lstrip("."),
            parent_count=len(parents),
            child_count=len(children),
            embedding_dim=embedding_dim,
            parse_ms=round(parse_ms, 1),
            chunk_ms=round(chunk_ms, 1),
            embed_ms=round(embed_ms, 1),
            index_ms=round(index_ms, 1),
            total_ms=round(total_ms, 1),
            metadata={"source_path": str(path)},
        )

    # -----------------------------------------------------------------------
    # Status helpers
    # -----------------------------------------------------------------------

    def _mark_processing(self, path: Path) -> None:
        """
        Create or update the Document row with status = PROCESSING.

        This is the first database write.  It records the intent to index
        and prevents stale COMPLETED status from being seen by retrieval
        while indexing is in progress.
        """
        try:
            DocumentORM.objects.update_or_create(
                id=self.document_id,
                defaults={
                    "filename": path.name,
                    "file_type": path.suffix.lower().lstrip("."),
                    "file_path": str(path),
                    "status": DocumentORM.Status.PROCESSING,
                },
            )
        except Exception as exc:
            # Not fatal on its own — log and continue.  A subsequent failure
            # will still call _mark_failed().
            logger.exception(
                "Could not set PROCESSING status for document_id=%s: %s",
                self.document_id,
                exc,
            )

    def _mark_failed(self) -> None:
        """
        Set Document.status = FAILED without touching chunk data.

        Called in every except branch so the document is never left in
        PROCESSING state after an error.
        """
        try:
            DocumentORM.objects.filter(id=self.document_id).update(
                status=DocumentORM.Status.FAILED
            )
            logger.warning(
                "Document marked FAILED | document_id=%s", self.document_id
            )
        except Exception as exc:
            # Best-effort; log but do not shadow the original exception.
            logger.exception(
                "Could not mark document FAILED for document_id=%s: %s",
                self.document_id,
                exc,
            )

    # -----------------------------------------------------------------------
    # DB persistence (stage 6)
    # -----------------------------------------------------------------------

    def _replace_db_chunks(
        self,
        path: Path,
        parents: List[ParentChunk],
        children: List[ChildChunk],
    ) -> None:
        """
        Atomically replace all chunk records for this document and mark it
        COMPLETED.

        Strategy
        --------
        Inside a single ``transaction.atomic()``:

        1. Delete all existing ``ParentChunkORM`` rows that belong to this
           document.  Because ``ChildChunkORM.parent`` uses
           ``on_delete=CASCADE``, this also deletes all child rows.
        2. ``bulk_create`` the new parent rows.
        3. ``bulk_create`` the new child rows.
        4. Set ``Document.status = COMPLETED``.

        This guarantees:
        * No stale parent or child rows remain after re-indexing.
        * The PostgreSQL chunk representation exactly matches the current
          chunking output.
        * Foreign-key ordering is respected: parents are inserted before
          children.
        * The entire replacement is atomic; a failure at any step rolls back
          PostgreSQL to the state it was in before this method was called
          (PROCESSING status and whatever old chunks existed).

        Raises ``ChunkingError`` on failure so callers know that vector stores
        have been updated but the DB record is inconsistent — this triggers
        ``_mark_failed()`` in the caller.
        """
        try:
            with transaction.atomic():
                # Ensure the Document row exists (created by _mark_processing,
                # but this is idempotent).
                doc_obj, _ = DocumentORM.objects.update_or_create(
                    id=self.document_id,
                    defaults={
                        "filename": path.name,
                        "file_type": path.suffix.lower().lstrip("."),
                        "file_path": str(path),
                        # Keep PROCESSING inside the transaction; we set
                        # COMPLETED at the end once everything succeeds.
                        "status": DocumentORM.Status.PROCESSING,
                    },
                )

                # Step 1 – delete stale parents (cascades to children).
                deleted_parents, _ = ParentChunkORM.objects.filter(
                    document=doc_obj
                ).delete()
                if deleted_parents:
                    logger.debug(
                        "Deleted %d stale parent chunks for document_id=%s",
                        deleted_parents,
                        self.document_id,
                    )

                # Step 2 – bulk-create parent chunks (must precede children).
                if parents:
                    ParentChunkORM.objects.bulk_create(
                        [
                            ParentChunkORM(
                                id=p.id,
                                document=doc_obj,
                                text=p.text,
                                metadata=p.metadata,
                            )
                            for p in parents
                        ],
                        # ignore_conflicts=False (default): raise on duplicate
                        # primary keys — this should never happen after the
                        # delete above, but we want to know if it does.
                    )

                # Step 3 – bulk-create child chunks.
                if children:
                    ChildChunkORM.objects.bulk_create(
                        [
                            ChildChunkORM(
                                id=c.id,
                                parent_id=c.parent_id,
                                document=doc_obj,
                                text=c.text,
                                metadata=c.metadata,
                            )
                            for c in children
                        ],
                    )

                # Step 4 – mark COMPLETED now that everything is consistent.
                doc_obj.status = DocumentORM.Status.COMPLETED
                doc_obj.save(update_fields=["status", "updated_at"])

            logger.info(
                "PostgreSQL persist complete | document_id=%s parents=%d children=%d",
                self.document_id,
                len(parents),
                len(children),
            )

        except Exception as exc:
            logger.exception(
                "PostgreSQL persist failed for document_id=%s",
                self.document_id,
            )
            # Mark FAILED outside the (rolled-back) transaction.
            self._mark_failed()
            raise ChunkingError(
                f"Database persistence failed for document '{path.name}': {exc}"
            ) from exc
