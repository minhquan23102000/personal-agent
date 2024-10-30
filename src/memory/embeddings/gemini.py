from dataclasses import dataclass, field
from typing import List
import google.generativeai as genai
from src.memory.embeddings.base import BaseEmbedding
from src.util.rotating_list import RotatingList
from src.config import get_google_api_key


@dataclass
class GeminiEmbeddingConfig:
    api_key: str | list[str] = field(default_factory=get_google_api_key)
    model_name: str = "models/text-embedding-004"
    embedding_size: int = 768
    cache_enabled: bool = True


class GeminiEmbedding(BaseEmbedding):
    """Gemini embedding implementation"""

    def __init__(self, config: GeminiEmbeddingConfig):
        """Initialize Gemini embedding model

        Args:
            config (GeminiEmbeddingConfig): Configuration for Gemini embedding
        """
        self.config = config
        super().__init__(
            embedding_size=config.embedding_size, cache_enabled=config.cache_enabled
        )

    def _validate_config(self) -> None:
        """Validate configuration"""
        if not self.config.api_key:
            raise ValueError("API key is required")
        if self.config.embedding_size > 768:
            raise ValueError("Embedding size must be <= 768 dimensions")

    def _setup_model(self) -> None:
        """Setup Gemini embedding model"""
        if isinstance(self.config.api_key, list):
            self.rotating_api_keys = RotatingList(self.config.api_key)
        else:
            self.rotating_api_keys = RotatingList([self.config.api_key])

    async def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding for a text using Gemini API

        Args:
            text (str): Input text to embed

        Returns:
            List[float]: Embedding vector
        """
        genai.configure(api_key=self.rotating_api_keys.rotate())

        result = genai.embed_content(
            content=text,
            model=self.config.model_name,
            output_dimensionality=self.config.embedding_size,
        )
        return result["embedding"]

    async def clear_cache(self) -> None:
        """Clear embedding cache"""
        # Implement caching logic if needed
        pass
