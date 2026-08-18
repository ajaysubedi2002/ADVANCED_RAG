from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.ingestion.bm25_index import BM25Index
from app.ingestion.embedding_service import EmbeddingService
from app.ingestion.exceptions import DocumentNotFoundError, LLMServiceError
from app.ingestion.pipeline import IngestionPipeline, IngestionSummary
from app.generation.llm import LLMService
from app.generation.prompt import PromptBuilder
from app.retrieval.dense_search import VectorStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.parent_expansion import ExpandedParent, ParentExpander
from app.retrieval.reranker import Reranker
from app.services.cache import CacheService
from app.runtime_settings import settings

logger = logging.getLogger(__name__)


# Response schemas

@dataclass
class SourceReference:
    document_id: str
    parent_id: str
    chunk_id: str
    score: float
    text: str


@dataclass
class QueryResponse:
    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    request_id: str = ""
    latency_ms: float = 0.0
    cached: bool = False


# Singleton component registry

_lock = threading.Lock()
_components: Dict[str, Any] = {}


def _get_components() -> dict:
    """
    Initialise all service components once per process (lazy, thread-safe).

    Returns a dict with keys:
      embedding_service, vector_store, bm25_index,
      hybrid_retriever, reranker, llm_service,
      prompt_builder, cache_service
    """
    global _components  # noqa: PLW0603
    if _components:
        return _components

    with _lock:
        if _components:
            return _components

        logger.info("Initialising RAG service components")

        embedding_service = EmbeddingService()
        vector_store = VectorStore()
        bm25_index = BM25Index()

        # Load persisted BM25 index
        bm25_index.load()

        hybrid_retriever = HybridRetriever(
            embedding_service=embedding_service,
            vector_store=vector_store,
            bm25_index=bm25_index,
        )
        reranker = Reranker()
        llm_service = LLMService()
        prompt_builder = PromptBuilder()
        cache_service = CacheService()

        _components = {
            "embedding_service": embedding_service,
            "vector_store": vector_store,
            "bm25_index": bm25_index,
            "hybrid_retriever": hybrid_retriever,
            "reranker": reranker,
            "llm_service": llm_service,
            "prompt_builder": prompt_builder,
            "cache_service": cache_service,
        }

        logger.info("RAG service components ready")
        return _components


# DocumentService

class DocumentService:
    """
    Facade over the full RAG pipeline.

    All public methods are stateless with respect to a single request;
    shared state is managed by the component singletons in ``_components``.
    """

    # Ingestion

    def ingest_document(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        parent_size: Optional[int] = None,
        child_size: Optional[int] = None,
        overlap: Optional[int] = None,
    ) -> IngestionSummary:
        """
        Parse, chunk, embed, and index a document file.

        Parameters
        ----------
        file_path:   Absolute or relative path to a .pdf or .txt file.
        document_id: Optional caller-supplied stable ID; generated if omitted.
        parent_size: Override for parent chunk size (words).
        child_size:  Override for child chunk size (words).
        overlap:     Override for chunk overlap (words).

        Returns
        -------
        ``IngestionSummary`` with document_id, chunk counts, and timing.
        """
        c = _get_components()
        doc_id = document_id or str(uuid.uuid4())

        pipeline = IngestionPipeline(
            document_id=doc_id,
            embedding_service=c["embedding_service"],
            vector_store=c["vector_store"],
            bm25_index=c["bm25_index"],
        )

        summary = pipeline.ingest_file(
            file_path=file_path,
            parent_size=parent_size if parent_size is not None else settings.PARENT_CHUNK_SIZE,
            child_size=child_size if child_size is not None else settings.CHILD_CHUNK_SIZE,
            # Use explicit None check — overlap=0 is valid and must not fall
            # back to settings.CHUNK_OVERLAP via a falsy `or` expression.
            overlap=overlap if overlap is not None else settings.CHUNK_OVERLAP,
        )
        return summary

    # Query

    def query(self, question: str, use_cache: bool = True) -> QueryResponse:
        """
        Run the full RAG pipeline for *question*.

        Pipeline
        --------
        1. Cache lookup
        2. Hybrid retrieval (dense + BM25 → RRF)
        3. BGE reranking
        4. Parent expansion
        5. Prompt assembly
        6. LLM generation
        7. Cache write

        Parameters
        ----------
        question:  Natural-language question.
        use_cache: Whether to read/write Redis cache.

        Returns
        -------
        ``QueryResponse`` with answer, sources, and latency.
        """
        if not question or not question.strip():
            return QueryResponse(
                answer="Please provide a non-empty question.",
                request_id=str(uuid.uuid4()),
            )

        request_id = str(uuid.uuid4())
        c = _get_components()
        cache: CacheService = c["cache_service"]
        t_total = time.perf_counter()

        # --- Cache lookup 
        if use_cache:
            cache_key = cache.query_key(question)
            cached_value = cache.get(cache_key)
            if cached_value is not None:
                logger.info("Cache hit | request_id=%s", request_id)
                return QueryResponse(
                    answer=cached_value["answer"],
                    sources=[SourceReference(**s) for s in cached_value["sources"]],
                    request_id=request_id,
                    latency_ms=0.0,
                    cached=True,
                )

        #  Hybrid retrieval 
        hybrid: HybridRetriever = c["hybrid_retriever"]
        fused_results = hybrid.search(question)

        if not fused_results:
            return QueryResponse(
                answer="No relevant documents found for your question.",
                request_id=request_id,
                latency_ms=(time.perf_counter() - t_total) * 1000,
            )

        # Reranking 
        reranker: Reranker = c["reranker"]
        reranked = reranker.rerank(question, fused_results)

        # Parent expansion 
        expander = ParentExpander(top_k=settings.PARENT_CONTEXT_TOP_K)
        parents: List[ExpandedParent] = expander.expand(reranked)

        # --- Prompt + LLM generation 
        prompt_builder: PromptBuilder = c["prompt_builder"]
        system_prompt, user_prompt = prompt_builder.build(question, parents)

        logger.debug(
            "Prompt built | parents=%d context_chars=%d",
            len(parents),
            len(user_prompt),
        )
        if not parents:
            logger.warning("No parent contexts available — LLM will receive empty context")

        llm: LLMService = c["llm_service"]
        try:
            answer = llm.generate(system_prompt, user_prompt)
        except LLMServiceError as exc:
            logger.exception("LLM generation failed | request_id=%s", request_id)
            answer = f"Generation failed: {exc}"

        # Build source references 
        sources: List[SourceReference] = []
        for r in reranked:
            sources.append(SourceReference(
                document_id=r.document_id,
                parent_id=r.parent_id,
                chunk_id=r.chunk_id,
                score=round(r.score, 6),
                text=r.text[:300],  # truncate for API response size
            ))

        total_ms = (time.perf_counter() - t_total) * 1000
        logger.info(
            "RAG query complete | request_id=%s latency_ms=%.1f sources=%d",
            request_id,
            total_ms,
            len(sources),
        )

        response = QueryResponse(
            answer=answer,
            sources=sources,
            request_id=request_id,
            latency_ms=round(total_ms, 1),
        )

        # Cache write 
        if use_cache:
            cache.set(
                cache_key,
                {
                    "answer": response.answer,
                    "sources": [
                        {
                            "document_id": s.document_id,
                            "parent_id": s.parent_id,
                            "chunk_id": s.chunk_id,
                            "score": s.score,
                            "text": s.text,
                        }
                        for s in response.sources
                    ],
                },
            )

        return response

    # Deletion

    def delete_document(self, document_id: str) -> dict:
        """
        Remove a document and all its chunks from every index.

        Deletes from:
          * ChromaDB (dense vectors)
          * BM25 in-memory index + persisted file
          * PostgreSQL ORM (Document, ParentChunk, ChildChunk)
          * Redis cache (invalidates all query cache entries)

        Returns
        -------
        Summary dict with deleted counts.
        """
        c = _get_components()
        vector_store: VectorStore = c["vector_store"]
        bm25_index: BM25Index = c["bm25_index"]
        cache: CacheService = c["cache_service"]

        # Validate existence before deletion
        try:
            from app.models.documents import Document as DocumentORM  # noqa: PLC0415
            if not DocumentORM.objects.filter(id=document_id).exists():
                raise DocumentNotFoundError(f"Document '{document_id}' not found")
        except ImportError:
            pass  # ORM not available (standalone / test mode)

        # Vector store
        vector_store.delete_document(document_id)

        # BM25
        bm25_index.delete_document(document_id)
        bm25_index.save()

        # PostgreSQL ORM – cascade deletes ParentChunk and ChildChunk
        try:
            from app.models.documents import Document as DocumentORM  # noqa: PLC0415
            deleted_count, _ = DocumentORM.objects.filter(id=document_id).delete()
        except Exception as exc:
            logger.warning("ORM delete failed for document_id=%s: %s", document_id, exc)
            deleted_count = 0

        # Invalidate entire RAG query cache (simple strategy)
        cache.invalidate_pattern(f"{cache.prefix}:query:*")

        logger.info("Document deleted | document_id=%s", document_id)
        return {
            "document_id": document_id,
            "status": "deleted",
            "orm_records_deleted": deleted_count,
        }

    # Health

    def health(self) -> dict:
        c = _get_components()
        return {
            "vector_store": c["vector_store"].health_check(),
            "bm25": c["bm25_index"].health_check(),
            "ollama": c["llm_service"].health_check(),
            "cache": c["cache_service"].health_check(),
        }
