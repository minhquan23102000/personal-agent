from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
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
    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text

        Args:
            text (str): Input text to embed

        Returns:
            np.ndarray: Embedding vector
        """
        pass

    @abstractmethod
    async def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts

        Args:
            texts (List[str]): List of input texts

        Returns:
            List[np.ndarray]: List of embedding vectors
        """
        pass

    @abstractmethod
    async def get_entity_embedding(self, entity: str) -> np.ndarray:
        """Get embedding for an entity

        Args:
            entity (str): Entity text to embed

        Returns:
            np.ndarray: Embedding vector
        """
        pass

    @abstractmethod
    async def get_relationship_embedding(
        self, entity1: str, relationship: str, entity2: str
    ) -> np.ndarray:
        """Get embedding for an entity relationship

        Args:
            entity1 (str): First entity
            relationship (str): Relationship type
            entity2 (str): Second entity

        Returns:
            np.ndarray: Embedding vector
        """
        pass

    @abstractmethod
    async def clear_cache(self) -> None:
        """Clear embedding cache"""
        pass
