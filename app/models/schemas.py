from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


# Ingestion schemas

@dataclass
class ParentChunk:
    """Larger context window; stored in PostgreSQL; NOT embedded."""
    id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChildChunk:
    """Smaller retrieval unit; embedded into ChromaDB and indexed in BM25."""
    id: str
    parent_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# API response schemas

@dataclass
class SourceReference:
    """Single source citation included in the API response."""
    document_id: str
    parent_id: str
    chunk_id: str
    score: float
    text: str


@dataclass
class QueryResponse:
    """Full RAG query response returned to the API client."""
    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    request_id: str = ""
    latency_ms: float = 0.0
    cached: bool = False
