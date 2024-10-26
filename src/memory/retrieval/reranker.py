from typing import List, Tuple, TypeVar, Generic, Callable
from sentence_transformers import CrossEncoder
import numpy as np
from loguru import logger

T = TypeVar("T")  # Generic type for items to be reranked


class CrossEncoderReranker:
    """Cross-encoder based reranker for improving ranking quality"""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        """Initialize cross encoder reranker

        Args:
            model_name (str): Name of the cross-encoder model
            device (str): Device to use for inference (cpu/cuda)
        """
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info(f"Initialized cross-encoder reranker with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {str(e)}")
            raise

    async def rerank(
        self,
        query: str,
        items: List[T],
        text_extractor: Callable[[T], str],
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> List[Tuple[T, float]]:
        """Rerank items using cross-encoder

        Args:
            query (str): Query text
            items (List[T]): List of items to rerank
            text_extractor (callable): Function to extract text from item for comparison
            top_k (int, optional): Number of top items to return
            threshold (float, optional): Minimum score threshold

        Returns:
            List[Tuple[T, float]]: Reranked list of (item, score) tuples
        """
        try:
            if not items:
                return []

            # Prepare text pairs for cross-encoder
            text_pairs = [[query, text_extractor(item)] for item in items]

            # Get cross-encoder scores
            cross_scores = self.model.predict(text_pairs)

            # Combine with original scores
            reranked_items = [
                (item, cross_score) for item, cross_score in zip(items, cross_scores)
            ]

            # Sort by combined score
            reranked_items.sort(key=lambda x: x[1], reverse=True)

            # Apply threshold if specified
            if threshold is not None:
                reranked_items = [
                    (item, score)
                    for item, score in reranked_items
                    if score >= threshold
                ]

            # Apply top_k if specified
            if top_k is not None:
                reranked_items = reranked_items[:top_k]

            return reranked_items

        except Exception as e:
            logger.error(f"Error in cross-encoder reranking: {str(e)}")
            raise
