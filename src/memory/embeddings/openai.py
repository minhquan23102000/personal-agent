from typing import List, Optional, Dict
import numpy as np
from openai import AsyncOpenAI
from loguru import logger

from src.memory.embeddings.base import BaseEmbedding
from src.memory.embeddings.cache import EmbeddingCache


class OpenAIEmbedding(BaseEmbedding):
    """OpenAI embedding implementation"""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        embedding_size: int = 364,
        cache_enabled: bool = True,
    ):
        """Initialize OpenAI embedding

        Args:
            api_key (str): OpenAI API key
            model (str): Model name to use
            embedding_size (int): Size of the embedding vector
            cache_enabled (bool): Whether to enable embedding caching
        """
        self.api_key = api_key
        self.model = model
        self.client: Optional[AsyncOpenAI] = None
        self.cache = EmbeddingCache() if cache_enabled else None
        super().__init__(embedding_size=embedding_size, cache_enabled=cache_enabled)

    def _validate_config(self) -> None:
        """Validate configuration"""
        if not isinstance(self.api_key, str):
            raise ValueError("api_key must be a string")
        if not isinstance(self.model, str):
            raise ValueError("model must be a string")

    def _setup_model(self) -> None:
        """Setup OpenAI client"""
        try:
            self.client = AsyncOpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized with model: {self.model}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")

    async def get_text_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

        try:
            # Check cache first
            if self.cache_enabled and (cached := await self.cache.get(text)):
                return cached

            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)

            # Cache the result
            if self.cache_enabled:
                await self.cache.set(text, embedding)

            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding for text: {str(e)}")
            raise

    async def get_batch_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get embeddings for a batch of texts"""
        if not self.client:
            raise RuntimeError("OpenAI client not initialized")

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
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=texts_to_embed,
                )

                for i, emb_data in zip(indices_to_embed, response.data):
                    embedding = np.array(emb_data.embedding, dtype=np.float32)
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
