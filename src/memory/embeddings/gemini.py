from typing import List, Optional
import numpy as np
import google.generativeai as genai
from loguru import logger

from src.memory.embeddings.base import BaseEmbedding
from src.memory.embeddings.cache import EmbeddingCache


class GeminiEmbedding(BaseEmbedding):
    """Google Gemini embedding implementation"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "models/embedding-001",
        embedding_size: int = 768,
        cache_enabled: bool = True,
    ):
        """Initialize Gemini embedding

        Args:
            api_key (str): Google API key
            model (str): Model name to use
            embedding_size (int): Size of the embedding vector
            cache_enabled (bool): Whether to enable embedding caching
        """
        self.api_key = api_key
        self.model_name = model_name
        self._setup_model()
        self.cache = EmbeddingCache() if cache_enabled else None
        super().__init__(embedding_size=embedding_size, cache_enabled=cache_enabled)

    def _validate_config(self) -> None:
        """Validate configuration"""
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string")
        if not isinstance(self.model_name, str):
            raise ValueError("model_name must be a string")

    def _setup_model(self) -> None:
        """Setup Gemini client"""
        try:
            genai.configure(api_key=self.api_key)
            self.model: genai.GenerativeModel = genai.GenerativeModel(self.model_name)
            logger.info(f"Gemini model initialized: {self.model_name}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")

    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text"""
        try:
            # Check cache first
            if self.cache_enabled and (cached := await self.cache.get(text)):
                return cached

            # Get embedding
            result = self.model.embed_content(content=text)
            embedding = np.array(result.embedding, dtype=np.float32)

            # Cache the result
            if self.cache_enabled:
                await self.cache.set(text, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {str(e)}")
            raise

    async def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts"""
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
                # Gemini doesn't support true batching, so we process sequentially
                for i, text in zip(indices_to_embed, texts_to_embed):
                    result = self.model.embed_content(content=text)
                    embedding = np.array(result.embedding, dtype=np.float32)
                    embeddings.insert(i, embedding)

                    # Cache the result
                    if self.cache_enabled:
                        await self.cache.set(text, embedding)

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
