class DocumentParsingError(ValueError):
    """Raised when a document cannot be read, decoded, or structurally parsed."""


class ChunkingError(RuntimeError):
    """Raised when hierarchical parent/child chunking fails."""


class EmbeddingServiceError(RuntimeError):
    """Raised when embeddings cannot be produced by the configured model."""


class VectorStoreError(RuntimeError):
    """Raised when ChromaDB fails to create, search, or mutate a collection."""


class BM25IndexError(RuntimeError):
    """Raised when the BM25 index cannot be built, queried, saved, or loaded."""


class RerankerError(RuntimeError):
    """Raised when the BGE cross-encoder reranker fails to score candidates."""


class LLMServiceError(RuntimeError):
    """Raised when generation with the configured Ollama model fails."""


class CacheError(RuntimeError):
    """Raised when the Redis cache layer encounters an unrecoverable error."""


class DocumentNotFoundError(LookupError):
    """Raised when a requested document_id does not exist in the store."""


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing or invalid."""
