from abc import ABC, abstractmethod
from typing import List, Optional
from loguru import logger


class BaseEmbedding(ABC):
    """Base class for embedding implementations"""

    def __init__(self, embedding_size: int = 364, cache_enabled: bool = True):
        """Initialize embedding model

        Args:
            embedding_size (int): Size of the embedding vector
            cache_enabled (bool): Whether to enable embedding caching
        """
        self.embedding_size = embedding_size
        self.cache_enabled = cache_enabled
        self._validate_config()
        self._setup_model()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate configuration"""
        pass

    @abstractmethod
    def _setup_model(self) -> None:
        """Setup embedding model"""
        pass

    @abstractmethod
    async def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text

        Args:
            text (str): Input text to embed

        Returns:
            List[float]: Embedding vector
        """
        pass

    @abstractmethod
    async def clear_cache(self) -> None:
        """Clear embedding cache"""
        pass
