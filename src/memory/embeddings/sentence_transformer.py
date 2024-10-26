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
        cache_enabled: bool = True,
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
            raise ValueError("model_name must be a string")
        if not isinstance(self.device, str):
            raise ValueError("device must be a string")

    def _setup_model(self) -> None:
        """Setup SentenceTransformer model"""
        try:
            self.model = STModel(self.model_name, device=self.device)
            logger.info(f"SentenceTransformer model initialized: {self.model_name}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SentenceTransformer model: {str(e)}"
            )

    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text"""
        if not self.model:
            raise RuntimeError("SentenceTransformer model not initialized")

        try:
            # Check cache first
            if self.cache_enabled and (cached := await self.cache.get(text)):
                return cached

            # Get embedding
            embedding = self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

            # Cache the result
            if self.cache_enabled:
                await self.cache.set(text, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {str(e)}")
            raise

    async def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts"""
        if not self.model:
            raise RuntimeError("SentenceTransformer model not initialized")

        try:
            # Check cache first
            embeddings = []
            texts_to_embed = []
            indices_to_embed = []

            for i, text in enumerate(texts):
                if self.cache_enabled and (cached := await self.cache.get(text)):
                    embeddings.append(cached)
                else:
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)

            if texts_to_embed:
                batch_embeddings = self.model.encode(
                    texts_to_embed,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                for i, embedding in zip(indices_to_embed, batch_embeddings):
                    embeddings.insert(i, embedding)

                    # Cache the result
                    if self.cache_enabled:
                        await self.cache.set(texts_to_embed[i], embedding)

            return embeddings
        except Exception as e:
            logger.error(f"Error getting batch embeddings: {str(e)}")
            raise

    async def get_entity_embedding(self, entity: str) -> np.ndarray:
        """Get embedding for an entity"""
        return await self.get_text_embedding(entity)

    async def get_relationship_embedding(
        self, entity1: str, relationship: str, entity2: str
    ) -> np.ndarray:
        """Get embedding for an entity relationship"""
        relationship_text = f"{entity1} {relationship} {entity2}"
        return await self.get_text_embedding(relationship_text)

    async def clear_cache(self) -> None:
        """Clear embedding cache"""
        if self.cache_enabled and self.cache:
            await self.cache.clear()
