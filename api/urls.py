from django.urls import path

from .views import (
    delete_document,
    health_bm25,
    health_check,
    health_ollama,
    health_vector,
    ingest_document,
    list_documents,
    rag_query,
)

urlpatterns = [
    # Health
    path("api/health/", health_check, name="health"),
    path("api/health/vector/", health_vector, name="health-vector"),
    path("api/health/bm25/", health_bm25, name="health-bm25"),
    path("api/health/ollama/", health_ollama, name="health-ollama"),

    # Ingestion
    path("api/ingestion/documents/", ingest_document, name="ingest-document"),
    path("api/ingestion/documents/list/", list_documents, name="list-documents"),
    path("api/ingestion/documents/<str:document_id>/", delete_document, name="delete-document"),

    # RAG
    path("api/rag/query/", rag_query, name="rag-query"),
]
