"""Simple in-memory query result cache with TTL."""

import hashlib
import time
from typing import Optional, Dict, Any


class QueryCache:
    """Cache query results to avoid re-computation."""

    def __init__(self, ttl_seconds: int = 3600):
        """Initialize cache with TTL (default 1 hour)."""
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl_seconds

    def _hash_query(self, query: str) -> str:
        """Hash query text for use as cache key."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached result if exists and not expired."""
        key = self._hash_query(query)
        if key not in self.cache:
            return None

        entry = self.cache[key]
        if time.time() - entry["timestamp"] > self.ttl:
            del self.cache[key]
            return None

        return entry["response"]

    def set(self, query: str, response: Dict[str, Any]) -> None:
        """Cache query response."""
        key = self._hash_query(query)
        self.cache[key] = {
            "timestamp": time.time(),
            "response": response,
        }

    def clear(self) -> None:
        """Clear all cached entries."""
        self.cache.clear()

    def cleanup(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [
            key for key, entry in self.cache.items()
            if now - entry["timestamp"] > self.ttl
        ]
        for key in expired:
            del self.cache[key]
        return len(expired)

    def stats(self) -> Dict[str, int]:
        """Cache statistics."""
        self.cleanup()
        return {
            "size": len(self.cache),
            "ttl_seconds": self.ttl,
        }
