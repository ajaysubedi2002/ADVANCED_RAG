from __future__ import annotations

import logging
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class RuntimeSettings(BaseSettings):
    """
    Centralised, validated settings loaded from environment variables / .env.
    All components must import `settings` from this module – never read
    os.environ directly inside business logic.
    """

    app_name: str = "Advanced RAG API"
    APP_ENV: str = "development"

    OLLAMA_HOST: str = "http://ollama:11434"
    CHAT_MODEL: str = "llama3:latest"
    EMBED_MODEL: str = "nomic-embed-text"

    # Reranker
    # Accept both RERANKER_MODEL (env) and RERANK_MODEL (legacy) via alias
    RERANKER_MODEL: str = Field("BAAI/bge-reranker-base", alias="RERANKER_MODEL")
    RERANKER_DEVICE: str = "cpu"

    # Vector store (ChromaDB)
    CHROMA_DB_PATH: str = "/data/chroma"
    COLLECTIONS_NAME: str = "documents"

    # BM25 index persistence
    BM25_INDEX_PATH: str = "/data/bm25/bm25_index.pkl"

    # PostgreSQL
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "rag_db"
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "change_this_to_a_strong_password"

    # Redis
    REDIS_URL: str = "redis://redis:6379/0"

    # Chunking
    PARENT_CHUNK_SIZE: int = 1000
    CHILD_CHUNK_SIZE: int = 300
    CHUNK_OVERLAP: int = 50

    # Retrieval
    DENSE_TOP_K: int = 20
    BM25_TOP_K: int = 20
    HYBRID_TOP_K: int = 20
    RRF_K: int = 60
    RERANK_TOP_K: int = 10
    PARENT_CONTEXT_TOP_K: int = 5

    # Caching
    RAG_CACHE_TTL: int = 300          # seconds

    # Misc
    MAX_RAG_CONTEXT_CHARS: int = 6000
    RAG_CONTEXT_PLACEHOLDER: str = "No relevant context was found."
    VECTOR_JSON_PATH: str = "/tmp/vectors.json"

    # LangSmith – Observability
    LANGSMITH_TRACING: bool = False
    LANGSMITH_API_KEY: str = ""
    LANGSMITH_PROJECT: str = "advanced-rag"
    LANGSMITH_ENDPOINT: str = "https://api.smith.langchain.com"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
        # Allow RERANKER_MODEL from .env to populate the field
        populate_by_name=True,
    )

    # Field validators

    @field_validator(
        "PARENT_CHUNK_SIZE",
        "CHILD_CHUNK_SIZE",
        "DENSE_TOP_K",
        "BM25_TOP_K",
        "HYBRID_TOP_K",
        "RRF_K",
        "RERANK_TOP_K",
        "PARENT_CONTEXT_TOP_K",
        "RAG_CACHE_TTL",
        mode="before",
    )
    @classmethod
    def validate_positive_int(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("Value must be greater than 0")
        return int(value)

    @field_validator("CHUNK_OVERLAP", mode="before")
    @classmethod
    def validate_overlap(cls, value: int) -> int:
        v = int(value)
        if v < 0:
            raise ValueError("CHUNK_OVERLAP must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "RuntimeSettings":
        if self.CHUNK_OVERLAP >= self.CHILD_CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than "
                f"CHILD_CHUNK_SIZE ({self.CHILD_CHUNK_SIZE})"
            )
        if self.CHILD_CHUNK_SIZE > self.PARENT_CHUNK_SIZE:
            raise ValueError(
                f"CHILD_CHUNK_SIZE ({self.CHILD_CHUNK_SIZE}) must be <= "
                f"PARENT_CHUNK_SIZE ({self.PARENT_CHUNK_SIZE})"
            )
        return self

    # Derived helpers (not env vars)

    @property
    def bm25_index_dir(self) -> Path:
        return Path(self.BM25_INDEX_PATH).parent

    @property
    def chroma_db_dir(self) -> Path:
        return Path(self.CHROMA_DB_PATH)


# Module-level singleton – import this everywhere
settings = RuntimeSettings()


def validate_runtime_settings() -> RuntimeSettings:
    """
    Called at Django application startup (AppConfig.ready).
    Logs the active configuration and validates required connectivity settings.
    """
    required = {
        "OLLAMA_HOST": settings.OLLAMA_HOST,
        "EMBED_MODEL": settings.EMBED_MODEL,
        "CHAT_MODEL": settings.CHAT_MODEL,
        "RERANKER_MODEL": settings.RERANKER_MODEL,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required runtime settings: {', '.join(missing)}"
        )

    logger.info(
        "Runtime settings validated | env=%s ollama=%s embed=%s chat=%s reranker=%s langsmith_tracing=%s",
        settings.APP_ENV,
        settings.OLLAMA_HOST,
        settings.EMBED_MODEL,
        settings.CHAT_MODEL,
        settings.RERANKER_MODEL,
        settings.LANGSMITH_TRACING,
    )
    return settings
