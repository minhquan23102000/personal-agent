import pytest
from typing import List, Tuple
from dataclasses import dataclass

from src.memory.retrieval.reranker import AdvancedReranker, RerankerConfig


@dataclass
class TestDocument:
    """Test document class containing text and metadata"""

    text: str
    metadata: dict = None


@pytest.fixture
def reranker():
    """Fixture for creating a reranker instance"""
    config = RerankerConfig()
    return AdvancedReranker(config=config)


@pytest.fixture
def sample_documents() -> List[TestDocument]:
    """Fixture providing sample documents for testing"""
    return [
        TestDocument(
            text="Javascript is a high-level programming language.",
            metadata={"category": "programming"},
        ),
        TestDocument(
            text="Machine learning is a subset of artificial intelligence.",
            metadata={"category": "ai"},
        ),
        TestDocument(
            text="Natural language processing helps computers understand human language.",
            metadata={"category": "nlp"},
        ),
        TestDocument(
            text="Deep learning models require significant computational resources.",
            metadata={"category": "ai"},
        ),
        TestDocument(
            text="My name is John Doe.",
            metadata={"name": "John Doe"},
        ),
        TestDocument(
            text="I live in New York City.",
            metadata={"location": "New York City"},
        ),
    ]


def test_basic_reranking(reranker, sample_documents):
    """Test basic reranking functionality"""
    query = "What is machine learning?"

    results = reranker.rerank(
        query=query, items=sample_documents, text_extractor=lambda x: x.text
    )

    assert len(results) == len(sample_documents)
    assert all(isinstance(score, float) for _, score in results)

    # Verify scores are in descending order
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)

    # Verify AI-related documents are ranked higher
    top_result = results[0][0]
    print(results)
    assert "artificial intelligence" in top_result.text.lower()


def test_top_k_filtering(reranker, sample_documents):
    """Test top-k filtering of results"""
    query = "What is machine learning?"
    top_k = 2

    results = reranker.rerank(
        query=query,
        items=sample_documents,
        text_extractor=lambda x: x.text,
        top_k=top_k,
    )

    assert len(results) == top_k
    assert all(isinstance(score, float) for _, score in results)


def test_threshold_filtering(reranker, sample_documents):
    """Test threshold-based filtering"""
    query = "Tell me about programming"
    threshold = 0.5

    results = reranker.rerank(
        query=query,
        items=sample_documents,
        text_extractor=lambda x: x.text,
        threshold=threshold,
    )

    assert all(score >= threshold for _, score in results)


def test_empty_items(reranker):
    """Test handling of empty input list"""
    query = "test query"

    results = reranker.rerank(query=query, items=[], text_extractor=lambda x: x.text)

    assert results == []


def test_with_metadata(reranker, sample_documents):
    """Test reranking with metadata extraction"""
    query = "Tell me about AI"

    results = reranker.rerank(
        query=query,
        items=sample_documents,
        text_extractor=lambda x: x.text,
        metadata_extractor=lambda x: x.metadata,
    )

    assert len(results) == len(sample_documents)
    # Verify AI-related documents are ranked higher
    top_docs = [doc for doc, _ in results[:2]]
    assert any(doc.metadata["category"] == "ai" for doc in top_docs)


def test_error_handling():
    """Test error handling with invalid configuration"""
    with pytest.raises(Exception):
        config = RerankerConfig(
            model_name="invalid_model_name", model_type="invalid_type"
        )
        AdvancedReranker(config=config)
