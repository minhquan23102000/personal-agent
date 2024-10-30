from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer as STModel
from loguru import logger

from src.memory.embeddings.base import BaseEmbedding
from src.memory.embeddings.cache import EmbeddingCache


class SentenceTransformerEmbedding(BaseEmbedding):
    """SentenceTransformer embedding implementation"""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        embedding_size: int = 384,
        cache_enabled: bool = False,
        device: str | None = None,
    ):
        """Initialize SentenceTransformer embedding

        Args:
            model_name (str): Model name to use (from HuggingFace)
            embedding_size (int): Size of the embedding vector
            cache_enabled (bool): Whether to enable embedding caching
            device (str): Device to use for inference (cpu/cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model: Optional[STModel] = None
        self.cache = EmbeddingCache() if cache_enabled else None
        super().__init__(embedding_size=embedding_size, cache_enabled=cache_enabled)

    def _validate_config(self) -> None:
        """Validate configuration"""
        if not isinstance(self.model_name, str):
            raise ValueError(f"model_name must be a string, got {self.model_name}")
        if not isinstance(self.device, str) and self.device is not None:
            raise ValueError(f"device must be a string or None, got {self.device}")

    def _setup_model(self) -> None:
        """Setup SentenceTransformer model"""
        try:
            self.model = STModel(self.model_name, device=self.device)
            logger.info(f"SentenceTransformer model initialized: {self.model_name}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SentenceTransformer model: {str(e)}"
            )

    async def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text"""
        if not self.model:
            raise RuntimeError("SentenceTransformer model not initialized")

        try:
            # Check cache first
            if self.cache_enabled and (cached := await self.cache.get(text)):
                return cached

            # Get embedding
            embedding = self.model.encode(
                [text], convert_to_numpy=True, show_progress_bar=False
            ).tolist()[0]

            # Cache the result
            if self.cache_enabled:
                await self.cache.set(text, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {str(e)}")
            raise

    async def clear_cache(self) -> None:
        """Clear embedding cache"""
        if self.cache_enabled and self.cache:
            await self.cache.clear()
