from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any, List, Literal, Optional
from datetime import datetime
from dataclasses import dataclass

from src.memory.models import (
    ConversationMemory,
    Knowledge,
    EntityRelationship,
    ConversationSummary,
    MessageType,
    ShortTermMemory,
)


@dataclass
class BaseDatabase(ABC):
    """Abstract base class for database implementations"""

    connection_config: dict

    def __post_init__(self):
        """Initialize database with connection configuration"""
        self._validate_config()
        self._setup_connection()

    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the connection configuration"""
        pass

    @abstractmethod
    def _setup_connection(self) -> None:
        """Setup initial database connection and configuration"""
        pass

    @abstractmethod
    async def get_connection(self) -> AsyncGenerator[Any, None]:
        """Get database connection"""
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize database schema"""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connection"""
        pass

    @abstractmethod
    async def store_short_term_memory(
        self,
        conversation_id: str,
        user_info: str,
        last_conversation_summary: str,
        recent_goal_and_status: str,
        important_context: str,
        agent_beliefs: str,
        agent_info: str,
        environment_info: str,
        summary_embedding: List[float],
    ) -> ShortTermMemory:
        """Store short-term memory state with embedding"""
        pass

    @abstractmethod
    async def get_short_term_memory(
        self, conversation_id: Optional[str] = None
    ) -> Optional[ShortTermMemory]:
        """Retrieve current short-term memory state"""
        pass

    # Abstract Storage Operations
    @abstractmethod
    async def store_conversation(
        self,
        sender: str,
        message_content: str,
        message_type: MessageType,
        conversation_id: str,
    ) -> ConversationMemory:
        """Store conversation in database"""
        pass

    @abstractmethod
    async def store_knowledge(
        self,
        text: str,
        entities: List[str],
        keywords: List[str],
        text_embedding: List[float],
        entity_embeddings: List[float] | None,
    ) -> Knowledge:
        """Store knowledge with embeddings"""
        pass

    @abstractmethod
    async def store_entity_relationship(
        self,
        relationship_text: str,
        embedding: List[float],
    ) -> EntityRelationship:
        """Store entity relationship"""
        pass

    @abstractmethod
    async def store_conversation_summary(
        self,
        conversation_id: str,
        prompt: str,
        conversation_summary: str,
        improve_prompt: str,
        reward_score: float,
        feedback_text: Optional[str] = None,
        example: Optional[str] = None,
        improvement_suggestion: Optional[str] = None,
    ) -> ConversationSummary:
        """Store conversation summary"""
        pass

    # Abstract Retrieval Operations
    @abstractmethod
    async def get_conversation_context(
        self, conversation_id: str, limit: int = 10
    ) -> List[ConversationMemory]:
        """Retrieve conversation context"""
        pass

    @abstractmethod
    async def search_similar_knowledge(
        self,
        query_embedding: List[float],
        vector_column: Literal[
            "text_embedding", "entity_embeddings"
        ] = "text_embedding",
        limit: int = 5,
    ) -> List[Knowledge]:
        """Search for similar knowledge"""
        pass

    @abstractmethod
    async def search_similar_entities(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[EntityRelationship]:
        """Search for similar entities"""
        pass

    @abstractmethod
    async def get_conversation_summary(
        self, conversation_id: str
    ) -> Optional[ConversationSummary]:
        """Get conversation summary"""
        pass

    @abstractmethod
    async def get_best_performing_prompts(
        self, limit: int = 5
    ) -> List[ConversationSummary]:
        """Get best performing prompts"""
        pass

    @abstractmethod
    async def update_conversation_feedback(
        self,
        conversation_id: str,
        feedback_text: str,
        reward_score: float,
        improvement_suggestion: Optional[str] = None,
        example: Optional[str] = None,
    ) -> Optional[ConversationSummary]:
        """Update conversation feedback"""
        pass

    @abstractmethod
    async def get_latest_conversation_summary(self) -> Optional[ConversationSummary]:
        """Get the most recent conversation summary"""
        pass

    @abstractmethod
    async def search_similar_short_term_memories(
        self,
        query_embedding: List[float],
        limit: int = 5,
    ) -> List[ShortTermMemory]:
        """Search for similar short-term memories"""
        pass
