# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "API"

    # Models
    CHAT_MODEL: str = "llama3:latest"
    EMBED_MODEL: str = "nomic-embed-text"
    MAX_RAG_CONTEXT_CHARS: int = 6000
    RAG_CONTEXT_PLACEHOLDER: str = ""
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

    # Vector Store
    CHROMA_DB_PATH: str = "/data/chroma"

    # Vector store collections
    COLLECTIONS_NAME: str = "documents"

    # Ollama configuration
    OLLAMA_HOST: str = "http://ollama:11434"

    VECTOR_JSON_PATH: str = "/tmp/vectors.json"

    # Pydantic V2 configuration management
    model_config = SettingsConfigDict(
        extra="ignore"
    )

settings = Settings()
