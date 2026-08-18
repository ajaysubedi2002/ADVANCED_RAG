from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from app.runtime_settings import settings

logger = logging.getLogger(__name__)

_REDIS_CLIENT: Optional[Any] = None


def _get_redis() -> Optional[Any]:
    """
    Lazily initialise the Redis client.

    Returns ``None`` if Redis is unavailable so callers can gracefully
    degrade without a hard failure.
    """
    global _REDIS_CLIENT 
    if _REDIS_CLIENT is not None:
        return _REDIS_CLIENT
    try:
        import redis  # noqa: PLC0415
        client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=2)
        client.ping()
        _REDIS_CLIENT = client
        logger.info("Redis connected | url=%s", settings.REDIS_URL)
    except Exception as exc:
        logger.warning("Redis unavailable (%s); caching disabled", exc)
        _REDIS_CLIENT = None
    return _REDIS_CLIENT


class CacheService:
    """
    Thin wrapper around Redis for query-result caching.

    Parameters
    ----------
    prefix:  Key prefix (default ``"rag"``).
    ttl:     Cache TTL in seconds (default ``settings.RAG_CACHE_TTL``).
    """

    def __init__(
        self,
        prefix: str = "rag",
        ttl: Optional[int] = None,
    ) -> None:
        self.prefix = prefix
        self.ttl = ttl if ttl is not None else settings.RAG_CACHE_TTL

    # Key helpers

    def make_key(self, *parts: str) -> str:
        """
        Build a deterministic cache key from *parts*.

        Example
        -------
        ``make_key("query", "What is RAG?")``  →  ``"rag:query:<sha256>"``
        """
        payload = ":".join(str(p) for p in parts)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"{self.prefix}:query:{digest}"

    def query_key(self, query: str) -> str:
        """Cache key for a RAG query (includes embed model and chat model)."""
        return self.make_key(query, settings.EMBED_MODEL, settings.CHAT_MODEL)

    # Cache operations

    def get(self, key: str) -> Optional[Any]:
        """Return the cached value for *key* or ``None`` on miss/error."""
        client = _get_redis()
        if client is None:
            return None
        try:
            raw = client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Cache get failed for key '%s': %s", key, exc)
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Serialise *value* and store it under *key* with TTL."""
        client = _get_redis()
        if client is None:
            return
        effective_ttl = ttl if ttl is not None else self.ttl
        try:
            client.setex(key, effective_ttl, json.dumps(value))
        except Exception as exc:
            logger.warning("Cache set failed for key '%s': %s", key, exc)

    def delete(self, key: str) -> None:
        """Remove a single cache entry."""
        client = _get_redis()
        if client is None:
            return
        try:
            client.delete(key)
        except Exception as exc:
            logger.warning("Cache delete failed for key '%s': %s", key, exc)

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching *pattern*.

        Uses SCAN instead of KEYS so the operation is non-blocking on large
        Redis instances.  Returns the number of deleted keys (0 on error).
        """
        client = _get_redis()
        if client is None:
            return 0
        try:
            deleted = 0
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += client.delete(*keys)
                if cursor == 0:
                    break
            return deleted
        except Exception as exc:
            logger.warning("Cache pattern invalidation failed for '%s': %s", pattern, exc)
            return 0

    def health_check(self) -> dict:
        """Return Redis health status."""
        client = _get_redis()
        if client is None:
            return {"status": "unavailable", "url": settings.REDIS_URL}
        try:
            client.ping()
            return {"status": "ok", "url": settings.REDIS_URL}
        except Exception as exc:
            return {"status": "error", "detail": str(exc), "url": settings.REDIS_URL}
