"""
Django REST Framework views for the RAG system.

Endpoints
---------
POST   /api/ingestion/documents/          – ingest a document (file upload or path)
GET    /api/ingestion/documents/          – list all ingested documents
DELETE /api/ingestion/documents/{id}/    – delete a document from all indexes

POST   /api/rag/query/                    – run the full RAG pipeline

GET    /api/health/                       – overall health
GET    /api/health/vector/               – ChromaDB health
GET    /api/health/bm25/                 – BM25 index health
GET    /api/health/ollama/               – Ollama reachability

All views are thin wrappers that delegate to ``DocumentService``.
They validate inputs, map domain exceptions to HTTP status codes, and
never expose raw stack traces to clients.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path

from drf_spectacular.utils import (
    OpenApiExample,
    OpenApiParameter,
    OpenApiResponse,
    extend_schema,
    inline_serializer,
)
from rest_framework import serializers, status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import FormParser, JSONParser, MultiPartParser
from rest_framework.request import Request
from rest_framework.response import Response

logger = logging.getLogger(__name__)

_document_service = None


def _get_service():
    global _document_service  
    if _document_service is None:
        from app.services.document_service import DocumentService  # noqa: PLC0415
        _document_service = DocumentService()
    return _document_service


# Exceptions are in a lightweight module with no heavy deps – safe to import now.
from app.ingestion.exceptions import (  
    ChunkingError,
    DocumentNotFoundError,
    DocumentParsingError,
    EmbeddingServiceError,
    LLMServiceError,
    VectorStoreError,
)
from app.ingestion.parser import _SUPPORTED_EXTENSIONS  


# Health endpoints

@extend_schema(
    tags=["Health"],
    description="Overall system health check.",
    responses={200: OpenApiResponse(description="All components healthy.")},
)
@api_view(["GET"])
def health_check(request: Request) -> Response:
    try:
        health = _get_service().health()
        all_ok = all(v.get("status") == "ok" for v in health.values())
        http_status = status.HTTP_200_OK if all_ok else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response({"status": "ok" if all_ok else "degraded", "components": health}, status=http_status)
    except Exception as exc:
        logger.exception("Health check failed")
        return Response({"status": "error", "detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@extend_schema(tags=["Health"], description="ChromaDB vector store health.")
@api_view(["GET"])
def health_vector(request: Request) -> Response:
    try:
        from app.services.document_service import _get_components  # noqa: PLC0415
        result = _get_components()["vector_store"].health_check()
        return Response(result)
    except Exception as exc:
        return Response({"status": "error", "detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@extend_schema(tags=["Health"], description="BM25 index health.")
@api_view(["GET"])
def health_bm25(request: Request) -> Response:
    try:
        from app.services.document_service import _get_components  # noqa: PLC0415
        result = _get_components()["bm25_index"].health_check()
        return Response(result)
    except Exception as exc:
        return Response({"status": "error", "detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


@extend_schema(tags=["Health"], description="Ollama LLM service health.")
@api_view(["GET"])
def health_ollama(request: Request) -> Response:
    try:
        from app.services.document_service import _get_components  # noqa: PLC0415
        result = _get_components()["llm_service"].health_check()
        http_status = status.HTTP_200_OK if result.get("status") == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
        return Response(result, status=http_status)
    except Exception as exc:
        return Response({"status": "error", "detail": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)


# Ingestion endpoints
@extend_schema(
    tags=["Ingestion"],
    description=(
        "Ingest a PDF or TXT document. "
        "Accepts a multipart file upload OR a JSON body with ``file_path``."
    ),
    request={
        "multipart/form-data": {
            "type": "object",
            "properties": {
                "file": {"type": "string", "format": "binary"},
                "document_id": {"type": "string", "example": "doc-001"},
                "parent_size": {"type": "integer", "default": 1000},
                "child_size": {"type": "integer", "default": 300},
                "overlap": {"type": "integer", "default": 50},
            },
            "required": ["file"],
        },
        "application/json": inline_serializer(
            name="IngestPathRequest",
            fields={
                "file_path": serializers.CharField(required=True),
                "document_id": serializers.CharField(required=False, default="", help_text="Optional stable ID. Leave blank to auto-generate a UUID."),
                "parent_size": serializers.IntegerField(required=False, default=1000),
                "child_size": serializers.IntegerField(required=False, default=300),
                "overlap": serializers.IntegerField(required=False, default=50),
            },
        ),
    },
    responses={
        201: OpenApiResponse(description="Document ingested successfully."),
        400: OpenApiResponse(description="Invalid input or unsupported file type."),
        404: OpenApiResponse(description="file_path not found on server."),
        503: OpenApiResponse(description="Embedding or vector-store service unavailable."),
    },
    examples=[
        OpenApiExample(
            "Ingest by server path",
            value={"file_path": "/data/docs/report.pdf", "document_id": "doc-001"},
            request_only=True,
        ),
    ],
)
@api_view(["POST"])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def ingest_document(request: Request) -> Response:
    uploaded_file = request.FILES.get("file")
    temp_path: str | None = None

    try:
        payload = request.data or {}
    except Exception:
        return Response({"error": "Invalid request body"}, status=status.HTTP_400_BAD_REQUEST)

    file_path = payload.get("file_path")

    if uploaded_file is None and not file_path:
        return Response(
            {"error": "Provide either a file upload (multipart) or file_path (JSON)"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    # Validate optional params
    # Strip whitespace and treat blank strings as "not provided" so the
    # Swagger UI placeholder value (or an accidental empty field) never
    # gets persisted as the document id.
    _raw_doc_id = str(payload.get("document_id") or "").strip()
    document_id = _raw_doc_id if _raw_doc_id else str(uuid.uuid4())
    try:
        parent_size = int(payload.get("parent_size", 1000))
        child_size = int(payload.get("child_size", 300))
        overlap = int(payload.get("overlap", 50))
    except (TypeError, ValueError):
        return Response(
            {"error": "parent_size, child_size, and overlap must be integers"},
            status=status.HTTP_400_BAD_REQUEST,
        )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            return Response(
                {"error": f"Unsupported file type '{suffix}'. Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            temp_path = tmp.name
        resolved_path = temp_path
    else:
        resolved = Path(str(file_path)).expanduser().resolve()
        if not resolved.exists() or not resolved.is_file():
            return Response({"error": f"File not found: {resolved}"}, status=status.HTTP_404_NOT_FOUND)
        resolved_path = str(resolved)

    try:
        summary = _get_service().ingest_document(
            file_path=resolved_path,
            document_id=document_id,
            parent_size=parent_size,
            child_size=child_size,
            overlap=overlap,
        )
        return Response(
            {
                "status": "ok",
                "document_id": summary.document_id,
                "filename": summary.filename,
                "file_type": summary.file_type,
                "parent_chunks": summary.parent_count,
                "child_chunks": summary.child_count,
                "embedding_dim": summary.embedding_dim,
                "timing": {
                    "parse_ms": summary.parse_ms,
                    "chunk_ms": summary.chunk_ms,
                    "embed_ms": summary.embed_ms,
                    "index_ms": summary.index_ms,
                    "total_ms": summary.total_ms,
                },
            },
            status=status.HTTP_201_CREATED,
        )
    except DocumentParsingError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except ChunkingError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    except EmbeddingServiceError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except VectorStoreError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except BM25IndexError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except Exception as exc:
        logger.exception("Unexpected ingestion error for document_id=%s", document_id)
        return Response(
            {"error": "Internal ingestion error. Check server logs."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@extend_schema(
    tags=["Ingestion"],
    description="List all ingested documents.",
    responses={200: OpenApiResponse(description="List of documents.")},
)
@api_view(["GET"])
def list_documents(request: Request) -> Response:
    try:
        from app.models.documents import Document  # noqa: PLC0415
        docs = Document.objects.all().values(
            "id", "filename", "file_type", "status", "created_at", "updated_at"
        )
        return Response({"documents": list(docs)})
    except Exception as exc:
        logger.exception("list_documents failed")
        return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@extend_schema(
    tags=["Ingestion"],
    description="Delete a document and all associated chunks from every index.",
    parameters=[OpenApiParameter("document_id", str, OpenApiParameter.PATH)],
    responses={
        200: OpenApiResponse(description="Document deleted."),
        404: OpenApiResponse(description="Document not found."),
    },
)
@api_view(["DELETE"])
def delete_document(request: Request, document_id: str) -> Response:
    try:
        result = _get_service().delete_document(document_id)
        return Response(result)
    except DocumentNotFoundError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_404_NOT_FOUND)
    except Exception as exc:
        logger.exception("delete_document failed for document_id=%s", document_id)
        return Response({"error": str(exc)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# RAG query endpoint
@extend_schema(
    tags=["RAG"],
    description=(
        "Run the full RAG pipeline: hybrid retrieval → reranking → "
        "parent expansion → LLM generation."
    ),
    request=inline_serializer(
        name="QueryRequest",
        fields={
            "question": serializers.CharField(required=True, help_text="Natural-language question."),
            "use_cache": serializers.BooleanField(required=False, default=True),
        },
    ),
    responses={
        200: OpenApiResponse(description="RAG answer with source citations."),
        400: OpenApiResponse(description="Invalid request."),
        503: OpenApiResponse(description="RAG service unavailable."),
    },
    examples=[
        OpenApiExample(
            "Example query",
            value={"question": "What are the main findings of the report?"},
            request_only=True,
        ),
        OpenApiExample(
            "Example response",
            value={
                "answer": "The main findings are ...",
                "sources": [
                    {
                        "document_id": "doc-001",
                        "parent_id": "parent-uuid",
                        "chunk_id": "child-uuid",
                        "score": 0.91,
                        "text": "The report states that ...",
                    }
                ],
                "request_id": "req-uuid",
                "latency_ms": 1240.5,
                "cached": False,
            },
            response_only=True,
        ),
    ],
)
@api_view(["POST"])
@parser_classes([JSONParser])
def rag_query(request: Request) -> Response:
    question = (request.data or {}).get("question", "")
    use_cache = bool((request.data or {}).get("use_cache", True))

    if not question or not str(question).strip():
        return Response(
            {"error": "Field 'question' is required and must not be empty."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    try:
        response = _get_service().query(question=str(question), use_cache=use_cache)
        return Response({
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
            "request_id": response.request_id,
            "latency_ms": response.latency_ms,
            "cached": response.cached,
        })
    except EmbeddingServiceError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except LLMServiceError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except VectorStoreError as exc:
        return Response({"error": str(exc)}, status=status.HTTP_503_SERVICE_UNAVAILABLE)
    except Exception as exc:
        logger.exception("RAG query failed")
        return Response(
            {"error": "Query processing failed. Check server logs."},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
