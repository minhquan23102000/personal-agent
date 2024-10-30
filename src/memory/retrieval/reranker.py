from typing import List, Tuple, TypeVar, Generic, Callable, Optional
from dataclasses import dataclass
from rerankers import Reranker, Document
import numpy as np
from loguru import logger
from src.config import JINA_API_KEY

T = TypeVar("T")  # Generic type for items to be reranked


@dataclass
class RerankerConfig:
    """Configuration for reranker"""

    model_name: str = (
        "jina-reranker-v2-base-multilingual"  # Default to standard cross-encoder
    )
    model_type: Optional[str] = "jina"  # Auto-detect model type
    api_key: Optional[str] = JINA_API_KEY  # For API-based rerankers
    device: Optional[str] = None  # Device for model inference
    lang: str = "en"  # Language for multilingual models
    verbose: int = 1  # Verbosity level
    max_length: Optional[int] = None  # Maximum length of input text


class AdvancedReranker:
    """Advanced reranker using the rerankers library supporting multiple architectures"""

    def __init__(self, config: RerankerConfig):
        """Initialize reranker

        Args:
            config (RerankerConfig): Configuration for the reranker
        """

        kwargs = {
            "lang": config.lang,
            "verbose": config.verbose,
        }

        # Add API key if provided
        if config.api_key:
            kwargs["api_key"] = config.api_key

        # Add model type if specified
        if config.model_type:
            kwargs["model_type"] = config.model_type

        # Add device if specified
        if config.device:
            kwargs["device"] = config.device

        # Add max length if specified
        if config.max_length:
            kwargs["max_length"] = config.max_length

        self.model = Reranker(config.model_name, **kwargs)
        logger.info(f"Initialized reranker with model: {config.model_name}")

    def rerank(
        self,
        query: str,
        items: List[T],
        text_extractor: Callable[[T], str],
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        metadata_extractor: Optional[Callable[[T], dict]] = None,
    ) -> List[Tuple[T, float]]:
        """Rerank items using the loaded reranker model

        Args:
            query (str): Query text
            items (List[T]): List of items to rerank
            text_extractor (callable): Function to extract text from item for comparison
            top_k (int, optional): Number of top items to return
            threshold (float, optional): Minimum score threshold
            metadata_extractor (callable, optional): Function to extract metadata from items

        Returns:
            List[Tuple[T, float]]: Reranked list of (item, score) tuples
        """
        try:
            if not items:
                return []

            # Create Document objects for reranking
            docs = []
            for idx, item in enumerate(items):
                doc_kwargs = {"text": text_extractor(item), "doc_id": idx}

                # Add metadata if extractor provided
                if metadata_extractor:
                    doc_kwargs["metadata"] = metadata_extractor(item)

                docs.append(Document(**doc_kwargs))

            # Get reranking results
            results = self.model.rank(query=query, docs=docs)

            # Create mapping from doc_id back to original items
            id_to_item = {idx: item for idx, item in enumerate(items)}

            # Process results
            reranked_items = [
                (id_to_item[result.document.doc_id], result.score)
                for result in results.results
            ]

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
            logger.error(f"Error in reranking: {str(e)}")
            raise
