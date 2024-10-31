from typing import Optional, Dict, List
from datetime import datetime, timedelta
from loguru import logger


class EmbeddingCache:
    """Simple in-memory cache for embeddings"""

    def __init__(self, ttl_seconds: int = 3600):
        """Initialize cache

        Args:
            ttl_seconds (int): Time-to-live in seconds for cache entries
        """
        self.cache: Dict[str, tuple[List[float], datetime]] = {}
        self.ttl = timedelta(seconds=ttl_seconds)

    async def get(self, key: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        try:
            if key in self.cache:
                embedding, timestamp = self.cache[key]
                if datetime.now() - timestamp < self.ttl:
                    return embedding
                else:
                    # Remove expired entry
                    del self.cache[key]
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {str(e)}")
            return None

    async def set(self, key: str, embedding: List[float]) -> None:
        """Set embedding in cache"""
        try:
            self.cache[key] = (embedding, datetime.now())
        except Exception as e:
            logger.error(f"Error setting cache: {str(e)}")

    async def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
