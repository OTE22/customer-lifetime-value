"""
Caching Layer for CLV Prediction System
Provides in-memory and Redis caching for predictions and data.
"""

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from functools import wraps
from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from .config import CacheConfig, get_config
from .logging_config import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


@dataclass
class CacheEntry(Generic[T]):
    """Represents a cached item with metadata."""

    value: T
    created_at: float
    ttl_seconds: int
    hits: int = 0

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        return time.time() - self.created_at > self.ttl_seconds

    def touch(self) -> None:
        """Update hit count."""
        self.hits += 1


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all values from the cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class MemoryCache(CacheBackend):
    """Thread-safe in-memory LRU cache implementation."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(key)
            entry.touch()
            self._hits += 1

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in the cache."""
        with self._lock:
            # Evict oldest if at capacity
            while len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)

            self._cache[key] = CacheEntry(
                value=value, created_at=time.time(), ttl_seconds=ttl or self._default_ttl
            )
            return True

    def delete(self, key: str) -> bool:
        """Delete a value from the cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> bool:
        """Clear all values from the cache."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            return True

    def exists(self, key: str) -> bool:
        """Check if a key exists and is not expired."""
        with self._lock:
            if key not in self._cache:
                return False
            if self._cache[key].is_expired:
                del self._cache[key]
                return False
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "type": "memory",
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "default_ttl": self._default_ttl,
            }

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count."""
        with self._lock:
            expired_keys = [key for key, entry in self._cache.items() if entry.is_expired]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)


class RedisCache(CacheBackend):
    """Redis-based cache implementation."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        default_ttl: int = 3600,
        prefix: str = "clv:",
    ):
        self._host = host
        self._port = port
        self._db = db
        self._default_ttl = default_ttl
        self._prefix = prefix
        self._client = None
        self._connected = False

    def _get_client(self):
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis

                self._client = redis.Redis(
                    host=self._host, port=self._port, db=self._db, decode_responses=True
                )
                self._client.ping()
                self._connected = True
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._connected = False
                raise
        return self._client

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis."""
        try:
            client = self._get_client()
            data = client.get(self._make_key(key))
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis."""
        try:
            client = self._get_client()
            data = json.dumps(value)
            client.setex(self._make_key(key), ttl or self._default_ttl, data)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a value from Redis."""
        try:
            client = self._get_client()
            return client.delete(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def clear(self) -> bool:
        """Clear all values with prefix from Redis."""
        try:
            client = self._get_client()
            keys = client.keys(f"{self._prefix}*")
            if keys:
                client.delete(*keys)
            return True
        except Exception as e:
            logger.error(f"Redis clear error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if a key exists in Redis."""
        try:
            client = self._get_client()
            return client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        try:
            client = self._get_client()
            info = client.info()
            keys = client.keys(f"{self._prefix}*")

            return {
                "type": "redis",
                "connected": self._connected,
                "host": self._host,
                "port": self._port,
                "size": len(keys),
                "used_memory": info.get("used_memory_human", "unknown"),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "default_ttl": self._default_ttl,
            }
        except Exception as e:
            return {"type": "redis", "connected": False, "error": str(e)}


class CacheManager:
    """Manages caching for the CLV system."""

    _instance: Optional["CacheManager"] = None

    def __init__(self, config: Optional[CacheConfig] = None):
        if config is None:
            config = get_config().cache

        self._config = config
        self._enabled = config.enabled

        if not self._enabled:
            self._backend = None
        elif config.type == "redis":
            self._backend = RedisCache(
                host=config.host, port=config.port, default_ttl=config.ttl_seconds
            )
        else:
            self._backend = MemoryCache(max_size=config.max_size, default_ttl=config.ttl_seconds)

    @classmethod
    def get_instance(cls) -> "CacheManager":
        """Get the singleton cache manager instance."""
        if cls._instance is None:
            cls._instance = CacheManager()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance."""
        if cls._instance is not None:
            cls._instance.clear()
        cls._instance = None

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from arguments."""
        key_data = {"args": args, "kwargs": kwargs}
        key_hash = hashlib.md5(
            json.dumps(key_data, sort_keys=True, default=str).encode()
        ).hexdigest()
        return f"{prefix}:{key_hash}"

    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if not self._enabled or self._backend is None:
            return None
        return self._backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        if not self._enabled or self._backend is None:
            return False
        return self._backend.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        if not self._enabled or self._backend is None:
            return False
        return self._backend.delete(key)

    def clear(self) -> bool:
        """Clear the cache."""
        if not self._enabled or self._backend is None:
            return False
        return self._backend.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self._enabled:
            return {"enabled": False}
        if self._backend is None:
            return {"enabled": True, "backend": None}

        stats = self._backend.get_stats()
        stats["enabled"] = True
        return stats

    def get_prediction(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction for a customer."""
        return self.get(f"prediction:{customer_id}")

    def set_prediction(
        self, customer_id: str, prediction: Dict[str, Any], ttl: Optional[int] = None
    ) -> bool:
        """Cache a prediction for a customer."""
        return self.set(f"prediction:{customer_id}", prediction, ttl)

    def invalidate_customer(self, customer_id: str) -> bool:
        """Invalidate all cached data for a customer."""
        return self.delete(f"prediction:{customer_id}")


def cached(
    prefix: str = "func", ttl: Optional[int] = None, key_func: Optional[Callable[..., str]] = None
):
    """Decorator for caching function results."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = CacheManager.get_instance()

            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = cache._generate_key(prefix, *args, **kwargs)

            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            logger.debug(f"Cached result for {key}")

            return result

        return wrapper

    return decorator


def get_cache() -> CacheManager:
    """Get the global cache manager instance."""
    return CacheManager.get_instance()
