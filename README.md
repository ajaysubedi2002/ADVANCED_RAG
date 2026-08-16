# Advanced RAG System

A Retrieval-Augmented Generation (RAG) API built with Django, featuring hybrid retrieval, cross-encoder reranking, parent-document expansion, and LangSmith observability.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipelines](#pipelines)
  - [Ingestion Pipeline](#ingestion-pipeline)
  - [Query Pipeline](#query-pipeline)
- [API Reference](#api-reference)
  - [Health Endpoints](#health-endpoints)
  - [Ingestion Endpoints](#ingestion-endpoints)
  - [RAG Query Endpoint](#rag-query-endpoint)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Docker Compose](#docker-compose)
- [Data Models](#data-models)
- [Observability](#observability)
- [Testing](#testing)

---

## Overview

This system ingests PDF and TXT documents, indexes them using both dense vector embeddings and BM25 sparse retrieval, then answers natural-language questions by:

1. Running hybrid retrieval (dense + BM25) fused via Reciprocal Rank Fusion (RRF)
2. Reranking candidates using a BGE cross-encoder
3. Expanding top child chunks back to their parent context windows
4. Assembling a grounded prompt and generating an answer via Ollama

Results are cached in Redis and all LLM calls are automatically traced in LangSmith.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Django REST API                              │
│   POST /ingest   GET /documents   DELETE /document   POST /query      │
└──────────────────┬─────────────────────────────┬────────────────────┘
                   │                             │
          ┌────────▼─────────┐         ┌────────▼──────────────┐
          │ Ingestion Pipeline│         │    Query Pipeline      │
          │                  │         │                        │
          │ 1. Parse (PDF/TXT)│         │ 1. Cache lookup        │
          │ 2. Chunk (parent/ │         │ 2. Hybrid Retrieval    │
          │    child hierachy)│         │    ├─ Dense (ChromaDB) │
          │ 3. Embed children │         │    └─ BM25 (sparse)    │
          │ 4a. Index ChromaDB│         │ 3. RRF Fusion          │
          │ 4b. Index BM25    │         │ 4. BGE Reranking       │
          │ 5. Persist to DB  │         │ 5. Parent Expansion    │
          └──────────────────┘         │ 6. LLM Generation      │
                                       │ 7. Cache write         │
                                       └────────────────────────┘
                                                │
                   ┌───────────────────────────┼──────────────────┐
                   ▼                           ▼                  ▼
           ┌──────────────┐         ┌──────────────────┐  ┌────────────┐
           │  PostgreSQL  │         │     ChromaDB     │  │   Redis    │
           │  (metadata,  │         │ (dense vectors)  │  │  (cache)   │
           │   chunks)    │         └──────────────────┘  └────────────┘
           └──────────────┘
                                              ▼
                                    ┌──────────────────┐
                                    │  Ollama (local)  │
                                    │  - Chat model    │
                                    │  - Embed model   │
                                    └──────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | Django 5.1 + Django REST Framework 3.15 |
| API documentation | drf-spectacular (Swagger / ReDoc) |
| Database | PostgreSQL 16 (metadata & chunk storage) |
| Vector store | ChromaDB 0.5 (dense embeddings) |
| Sparse retrieval | rank-bm25 (BM25Okapi, persisted as pickle) |
| Embeddings | Ollama (`nomic-embed-text:v1.5`) |
| LLM | Ollama via LangChain `ChatOllama` (`tinyllama` / any model) |
| Reranker | `BAAI/bge-reranker-base` via sentence-transformers |
| Caching | Redis 7 (SHA-256 keyed query results) |
| Settings | pydantic-settings (env var validation) |
| Document parsing | pdfplumber (PDF), built-in (TXT) |
| Observability | LangSmith (auto-tracing), structlog |
| Production server | Gunicorn |
| Containerisation | Docker Compose |

---

## Project Structure

```
Advanced_Rag/
├── api/                        # Django app – HTTP views and URL routing
│   ├── views.py                # All DRF view functions
│   ├── urls.py                 # URL patterns
│   ├── models.py               # ORM model re-exports
│   └── docker/
│       └── dockerfile
├── app/
│   ├── ingestion/              # Document ingestion pipeline
│   │   ├── pipeline.py         # 5-stage orchestrator
│   │   ├── parser.py           # PDF / TXT parser
│   │   ├── chunker.py          # Parent/child hierarchical chunking
│   │   ├── embedding_service.py# Ollama embedding wrapper
│   │   ├── bm25_index.py       # Persistent BM25Okapi index
│   │   ├── indexer.py          # Index coordination helpers
│   │   ├── metadata.py         # DocumentMetadata dataclass
│   │   └── exceptions.py       # Domain-specific exceptions
│   ├── retrieval/              # Multi-stage retrieval
│   │   ├── hybrid.py           # Dense + BM25 hybrid orchestrator
│   │   ├── dense_search.py     # ChromaDB VectorStore
│   │   ├── bm25_search.py      # BM25 search wrapper
│   │   ├── fusion.py           # Reciprocal Rank Fusion (RRF)
│   │   ├── reranker.py         # BGE cross-encoder reranker
│   │   └── parent_expansion.py # Child → parent context expansion
│   ├── generation/             # LLM generation
│   │   ├── llm.py              # LangChain ChatOllama service
│   │   └── prompt.py           # System + user prompt builder
│   ├── models/
│   │   ├── documents.py        # Django ORM models (Document, ParentChunk, ChildChunk)
│   │   └── schemas.py          # In-memory dataclass schemas (no DB dependency)
│   ├── services/
│   │   ├── document_service.py # High-level facade / singleton component registry
│   │   └── cache.py            # Redis cache service
│   ├── rag.py                  # Top-level RAG facade
│   └── runtime_settings.py     # Centralised pydantic-settings singleton
├── config/                     # Django project configuration
│   ├── settings.py
│   ├── urls.py                 # Root URL config (admin, OpenAPI, RAG routes)
│   ├── api_urls.py
│   ├── wsgi.py
│   └── asgi.py
├── scripts/
│   ├── ollama-entrypoint.sh    # Auto-pulls models on container start
│   └── reranker-entrypoint.sh
├── docker-compose.yml
├── requirements.txt
├── manage.py
└── .env                        # Environment configuration (copy from template below)
```

---

## Pipelines

### Ingestion Pipeline

Triggered via `POST /api/ingestion/documents/`. Runs 5 sequential stages:

```
Stage 1  Parse        pdfplumber (PDF) or built-in reader (TXT) → ParsedDocument
Stage 2  Chunk        Hierarchical split → ParentChunk (large) + ChildChunk (small)
Stage 3  Embed        Ollama embed_documents() for each ChildChunk
Stage 4a Vector index  Upsert child embeddings into ChromaDB collection
Stage 4b BM25 index   Add child texts to BM25Okapi; pickle to disk
Stage 5  Persist      Write Document, ParentChunk, ChildChunk to PostgreSQL
```

**Chunking strategy:**
- Parent chunks hold broad context (default 1000 chars) — stored in PostgreSQL only.
- Child chunks are smaller retrieval units (default 300 chars, 50-char overlap) — embedded and indexed in both ChromaDB and BM25.
- Each child carries a `parent_id` reference for later expansion.

### Query Pipeline

Triggered via `POST /api/rag/query/`. Runs 7 stages:

```
Stage 1  Cache lookup     SHA-256 hash of question → Redis GET
Stage 2  Dense retrieval  Embed query → ChromaDB cosine similarity (top DENSE_TOP_K)
Stage 3  BM25 retrieval   Tokenise query → BM25Okapi scores (top BM25_TOP_K)
Stage 4  RRF fusion       Reciprocal Rank Fusion of both lists (top HYBRID_TOP_K)
Stage 5  Reranking        BGE cross-encoder scores all candidates (top RERANK_TOP_K)
Stage 6  Parent expansion Map top reranked children to parent chunks (top PARENT_CONTEXT_TOP_K)
Stage 7  LLM generation   PromptBuilder → ChatOllama → answer string
         Cache write      Store result in Redis (TTL = RAG_CACHE_TTL seconds)
```

---

## API Reference

Interactive documentation is available at:
- Swagger UI: `http://localhost:8000/api/docs/`
- ReDoc: `http://localhost:8000/api/redoc/`
- OpenAPI schema: `http://localhost:8000/api/schema/`

### Health Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health/` | Overall system health (all components) |
| `GET` | `/api/health/vector/` | ChromaDB vector store health |
| `GET` | `/api/health/bm25/` | BM25 index health |
| `GET` | `/api/health/ollama/` | Ollama LLM service reachability |

**Example response:**
```json
{
  "status": "ok",
  "components": {
    "vector_store": { "status": "ok" },
    "bm25": { "status": "ok" },
    "ollama": { "status": "ok", "model": "tinyllama:latest" }
  }
}
```

---

### Ingestion Endpoints

#### Ingest a document

```
POST /api/ingestion/documents/
```

Accepts multipart file upload **or** a JSON body with a server-side file path.

**Multipart upload:**
```bash
curl -X POST http://localhost:8000/api/ingestion/documents/ \
  -F "file=@/path/to/report.pdf" \
  -F "document_id=doc-001" \
  -F "parent_size=1000" \
  -F "child_size=300" \
  -F "overlap=50"
```

**JSON body (server-side path):**
```bash
curl -X POST http://localhost:8000/api/ingestion/documents/ \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/data/docs/report.pdf", "document_id": "doc-001"}'
```

**Request parameters:**

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `file` | file | Yes* | — | PDF or TXT file upload |
| `file_path` | string | Yes* | — | Absolute server-side path (JSON only) |
| `document_id` | string | No | auto UUID | Stable document identifier |
| `parent_size` | integer | No | 1000 | Max characters per parent chunk |
| `child_size` | integer | No | 300 | Max characters per child chunk |
| `overlap` | integer | No | 50 | Character overlap between child chunks |

*Either `file` or `file_path` is required.

**Success response (201):**
```json
{
  "status": "ok",
  "document_id": "doc-001",
  "filename": "report.pdf",
  "file_type": "pdf",
  "parent_chunks": 12,
  "child_chunks": 48,
  "embedding_dim": 768,
  "timing": {
    "parse_ms": 340.2,
    "chunk_ms": 12.1,
    "embed_ms": 1820.5,
    "index_ms": 95.3,
    "total_ms": 2268.1
  }
}
```

---

#### List documents

```
GET /api/ingestion/documents/list/
```

```bash
curl http://localhost:8000/api/ingestion/documents/list/
```

**Response:**
```json
{
  "documents": [
    {
      "id": "doc-001",
      "filename": "report.pdf",
      "file_type": "pdf",
      "status": "completed",
      "created_at": "2026-08-15T10:00:00Z",
      "updated_at": "2026-08-15T10:00:05Z"
    }
  ]
}
```

---

#### Delete a document

```
DELETE /api/ingestion/documents/{document_id}/
```

Removes the document from ChromaDB, BM25 index, PostgreSQL, and invalidates the Redis cache.

```bash
curl -X DELETE http://localhost:8000/api/ingestion/documents/doc-001/
```

**Response (200):**
```json
{
  "status": "ok",
  "document_id": "doc-001",
  "deleted_chunks": 48
}
```

---

### RAG Query Endpoint

```
POST /api/rag/query/
```

Runs the full RAG pipeline and returns an answer with source citations.

```bash
curl -X POST http://localhost:8000/api/rag/query/ \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings of the report?", "use_cache": true}'
```

**Request body:**

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `question` | string | Yes | — | Natural-language question |
| `use_cache` | boolean | No | `true` | Read/write Redis cache |

**Success response (200):**
```json
{
  "answer": "The main findings indicate that...",
  "sources": [
    {
      "document_id": "doc-001",
      "parent_id": "parent-uuid",
      "chunk_id": "child-uuid",
      "score": 0.9134,
      "text": "The report states that..."
    }
  ],
  "request_id": "req-uuid",
  "latency_ms": 1240.5,
  "cached": false
}
```

---

## Configuration

All settings are loaded from environment variables (or `.env`). Copy the template below and adjust for your environment.

```bash
cp .env.example .env   # or edit .env directly
```

### Environment Variables

#### Application

| Variable | Default | Description |
|---|---|---|
| `APP_ENV` | `development` | Environment name |
| `DEBUG` | `true` | Django debug mode |
| `SECRET_KEY` | — | Django secret key (min 50 chars) |
| `ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated allowed hosts |

#### PostgreSQL

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_DB` | `rag_db` | Database name |
| `POSTGRES_USER` | `rag_user` | Database user |
| `POSTGRES_PASSWORD` | — | Database password |
| `POSTGRES_HOST` | `postgres` | Host (use `localhost` outside Docker) |
| `POSTGRES_PORT` | `5432` | Port |

#### Redis

| Variable | Default | Description |
|---|---|---|
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection URL |

#### Ollama (LLM & Embeddings)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_HOST` | `http://ollama:11434` | Ollama base URL |
| `CHAT_MODEL` | `tinyllama:latest` | Generation model tag |
| `EMBED_MODEL` | `nomic-embed-text:v1.5` | Embedding model tag |

#### Vector Store (ChromaDB)

| Variable | Default | Description |
|---|---|---|
| `CHROMA_DB_PATH` | `/models/chroma_db` | Persistent storage path |
| `COLLECTIONS_NAME` | `documents` | ChromaDB collection name |

#### BM25 Index

| Variable | Default | Description |
|---|---|---|
| `BM25_INDEX_PATH` | `/models/bm25/bm25_index.pkl` | Pickle file path |

#### Reranker

| Variable | Default | Description |
|---|---|---|
| `RERANKER_MODEL` | `BAAI/bge-reranker-base` | HuggingFace model ID |
| `RERANKER_DEVICE` | `cpu` | Torch device (`cpu` or `cuda`) |

#### Chunking

| Variable | Default | Description |
|---|---|---|
| `PARENT_CHUNK_SIZE` | `1000` | Max chars per parent chunk |
| `CHILD_CHUNK_SIZE` | `300` | Max chars per child chunk |
| `CHUNK_OVERLAP` | `50` | Char overlap between children |

#### Retrieval

| Variable | Default | Description |
|---|---|---|
| `DENSE_TOP_K` | `20` | Dense retrieval candidates |
| `BM25_TOP_K` | `20` | BM25 retrieval candidates |
| `HYBRID_TOP_K` | `20` | RRF fusion output count |
| `RRF_K` | `60` | RRF smoothing constant |
| `RERANK_TOP_K` | `10` | Reranker output count |
| `PARENT_CONTEXT_TOP_K` | `5` | Parent contexts sent to LLM |

#### Caching & Generation

| Variable | Default | Description |
|---|---|---|
| `RAG_CACHE_TTL` | `300` | Cache TTL in seconds |
| `MAX_RAG_CONTEXT_CHARS` | `6000` | Max chars in LLM context |

#### LangSmith Observability

| Variable | Default | Description |
|---|---|---|
| `LANGSMITH_TRACING` | `false` | Enable LangSmith tracing |
| `LANGSMITH_API_KEY` | — | API key from smith.langchain.com |
| `LANGSMITH_PROJECT` | `advanced-rag` | Project name in LangSmith |
| `LANGSMITH_ENDPOINT` | `https://api.smith.langchain.com` | LangSmith endpoint |

---

## Getting Started

### Prerequisites

- Docker and Docker Compose (recommended)
- Or: Python 3.11+, PostgreSQL 16, Redis 7, Ollama

### Local Development

**1. Clone and set up the virtual environment:**

```bash
git clone <repo-url>
cd Advanced_Rag
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Configure environment:**

```bash
cp .env .env.local
# Edit .env.local:
#   POSTGRES_HOST=localhost
#   REDIS_URL=redis://localhost:6379/0
#   OLLAMA_HOST=http://localhost:11434
```

**3. Start external services (Postgres, Redis, Ollama):**

```bash
# Start only infrastructure services
docker compose up postgres redis ollama -d
```

**4. Run migrations and start the API:**

```bash
python manage.py migrate
python manage.py runserver
```

**5. Pull required Ollama models:**

```bash
ollama pull nomic-embed-text:v1.5
ollama pull tinyllama:latest
```

---

### Docker Compose

The full stack is defined in `docker-compose.yml` with 4 core services and 1 optional service.

**Start all services:**

```bash
docker compose up -d
```

**Start with the BGE reranker as a separate inference service:**

```bash
docker compose --profile reranker up -d
```

**Service ports:**

| Service | Internal port | Host port |
|---|---|---|
| Django API | 8000 | 8000 |
| Ollama | 11434 | 11435 |
| PostgreSQL | 5432 | 5444 |
| Redis | 6379 | 6380 |

**Check all services are healthy:**

```bash
docker compose ps
curl http://localhost:8000/api/health/
```

**View logs:**

```bash
docker compose logs -f api
docker compose logs -f ollama
```

**Run migrations manually (if needed):**

```bash
docker compose exec api python manage.py migrate
```

**Shut down and clean up:**

```bash
docker compose down
# Remove persistent volumes too:
docker compose down -v
```

---

## Data Models

### PostgreSQL Schema

```
Document (id, filename, file_type, file_path, status, created_at, updated_at)
    │
    └── ParentChunk (id, document_id FK, text, metadata, created_at)
            │
            └── ChildChunk (id, parent_id FK, document_id FK, text, metadata, created_at)
```

- **Document** — top-level record with ingestion status (`pending` → `processing` → `completed` / `failed`).
- **ParentChunk** — broad context windows stored for parent expansion. Not embedded.
- **ChildChunk** — small retrieval units. Indexed in ChromaDB (dense) and BM25 (sparse).

### In-memory Schemas (no DB dependency)

All pipeline stages operate on typed Python dataclasses defined in `app/models/schemas.py`:

| Schema | Used in |
|---|---|
| `ParentChunk`, `ChildChunk` | Ingestion pipeline |
| `DenseResult` | ChromaDB search results |
| `BM25Result` | BM25 search results |
| `FusedResult` | After RRF fusion |
| `RerankedResult` | After BGE reranking |
| `ExpandedContext` | After parent expansion |
| `SourceReference`, `QueryResponse` | API responses |

---

## Observability

### LangSmith (LLM Tracing)

Every `generate()` call is automatically traced in LangSmith when `LANGSMITH_TRACING=true`. No manual instrumentation is required — LangChain instruments the `ChatOllama.invoke()` call transparently.

Traces include: full message list, model name, token counts, and latency.

Set up at [smith.langchain.com](https://smith.langchain.com).

### Structured Logging

All modules use Python's standard `logging` with structured key=value fields. Component-level latency is logged for every pipeline stage, e.g.:

```
Hybrid retrieval | dense=18 bm25=15 fused=20 dense_ms=45.2 bm25_ms=3.1 fuse_ms=0.4 total_ms=48.7
Reranking complete | candidates=20 returned=10 top_score=4.2831 latency_ms=312.5
Parent expansion complete | children_in=10 parents_out=5 top_k=5
LLM generation complete | model=tinyllama:latest latency_ms=2140.3 response_chars=412
```

---

## Testing

Tests use pytest with the Django test runner.

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run a specific test file
pytest app/tests/test_phase1_core.py -v
```

The test configuration is in `config/settings_test.py` and `conftest.py`. Tests are designed to run without live Ollama, ChromaDB, or PostgreSQL connections by injecting mock components.

---

## Exception Hierarchy

All domain errors are defined in `app/ingestion/exceptions.py` and map to HTTP status codes in the views:

| Exception | HTTP Status | Cause |
|---|---|---|
| `DocumentParsingError` | 400 | Unsupported file type or empty document |
| `ChunkingError` | 400 | Chunking configuration error |
| `EmbeddingServiceError` | 503 | Ollama embedding unreachable |
| `VectorStoreError` | 503 | ChromaDB insertion failure |
| `BM25IndexError` | 503 | BM25 index failure |
| `LLMServiceError` | 503 | Ollama chat model unreachable |
| `DocumentNotFoundError` | 404 | Document ID not in database |
| `RerankerError` | 503 | Cross-encoder load or predict failure |
