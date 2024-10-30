from typing import Optional, List, Tuple, TypeVar, Callable, Generic, Any, TYPE_CHECKING
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

from src.memory.models import (
    ConversationMemory,
    Knowledge,
    EntityRelationship,
    ConversationSummary,
    MessageType,
    ShortTermMemory,
)
from src.memory.database.base import BaseDatabase
from src.memory.embeddings.base import BaseEmbedding
from src.memory.retrieval.reranker import AdvancedReranker, RerankerConfig
from src.memory.database.sqlite import SQLiteDatabase
from src.memory.embeddings.sentence_transformer import SentenceTransformerEmbedding
from src.memory.memory_toolkit.static_flow.end_conversation import (
    reflection_conversation,
)
from src.memory.embeddings import GeminiEmbedding, GeminiEmbeddingConfig

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent

T = TypeVar("T")


class MemoryManager:
    def __init__(
        self,
        db_uri: str,
        database: BaseDatabase | None = None,
        embedding_model: BaseEmbedding | None = None,
        similarity_threshold: float = 0.7,
        max_search_knowledge_results: int = 5,
        max_search_entity_results: int = 10,
        use_reranker: bool = True,
    ):
        """Initialize MemoryManager

        Args:
            database (BaseDatabase): Database implementation
            embedding_model (BaseEmbedding): Embedding model implementation
            similarity_threshold (float): Minimum similarity score threshold
            max_results (int): Maximum number of results to return
            use_reranker (bool): Whether to use cross-encoder reranking
        """
        if embedding_model:
            self.embedding_model = embedding_model
        else:
            self.embedding_model = GeminiEmbedding(config=GeminiEmbeddingConfig())

        if database:
            self.db = database
        else:
            self.db = SQLiteDatabase(
                embedding_size=self.embedding_model.embedding_size, db_uri=db_uri
            )

        self.similarity_threshold = similarity_threshold
        self.max_search_knowledge_results = max_search_knowledge_results
        self.max_search_entity_results = max_search_entity_results
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = AdvancedReranker(config=RerankerConfig())

    def set_agent(self, agent: "BaseAgent") -> None:
        self.agent = agent

    async def reflection_conversation(
        self, user_feedback: str = "No user feedback provided."
    ) -> None:
        await reflection_conversation(memory_manager=self, user_feedback=user_feedback)

    async def store_conversation(
        self,
        sender: str,
        message_content: str,
        message_type: MessageType,
        conversation_id: str,
    ) -> ConversationMemory:
        """Store a conversation turn in short-term memory"""
        try:
            return await self.db.store_conversation(
                sender=sender,
                message_content=message_content,
                message_type=message_type,
                conversation_id=conversation_id,
            )
        except Exception as e:
            logger.error(f"Error storing conversation: {str(e)}")
            raise

    async def store_knowledge(
        self,
        text: str,
        entities: List[str],
        keywords: List[str],
    ) -> Knowledge:
        """Store new knowledge with embeddings"""
        try:
            text_embedding = await self.embedding_model.get_text_embedding(text)
            if entities:
                entity_embeddings = await self.embedding_model.get_text_embedding(
                    ", ".join(entities)
                )
            else:
                entity_embeddings = None
            return await self.db.store_knowledge(
                text=text,
                entities=entities,
                keywords=keywords,
                text_embedding=text_embedding,
                entity_embeddings=entity_embeddings,
            )
        except Exception as e:
            logger.error(f"Error storing knowledge: {str(e)}")
            raise

    async def search_similar_knowledge(
        self, query: str, limit: int = 5, threshold: float | None = None
    ) -> List[Knowledge]:
        """Search for similar knowledge using vector similarity"""
        try:
            query_embedding = await self.embedding_model.get_text_embedding(query)
            rs = await self.db.search_similar_knowledge(
                query_embedding=query_embedding,
                limit=limit,
            )
            # rerank results
            rs = await self._rerank_results(
                query=query,
                items=rs,
                text_extractor=lambda x: x.text,
                limit=limit,
                threshold=threshold,
            )
            return rs
        except Exception as e:
            logger.error(f"Error searching knowledge: {str(e)}")
            raise

    async def store_entity_relationship(
        self,
        relationship_text: str,
    ) -> EntityRelationship:
        """Store entity relationship with embedding"""
        try:
            embedding = await self.embedding_model.get_text_embedding(relationship_text)
            return await self.db.store_entity_relationship(
                relationship_text=relationship_text,
                embedding=embedding,
            )
        except Exception as e:
            logger.error(f"Error storing entity relationship: {str(e)}")
            raise

    async def search_similar_entities(
        self, query: str, limit: int = 10, threshold: float | None = None
    ) -> List[EntityRelationship]:
        """Search for similar entity relationships using vector similarity"""
        try:
            query_embedding = await self.embedding_model.get_text_embedding(query)
            rs = await self.db.search_similar_entities(
                query_embedding=query_embedding,
                limit=limit,
            )
            # rerank results
            rs = await self._rerank_results(
                query=query,
                items=rs,
                text_extractor=lambda x: x.relationship_text,
                limit=limit,
                threshold=threshold,
            )
            return rs
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            raise

    async def store_conversation_summary(
        self,
        conversation_id: str,
        prompt: str,
        conversation_summary: str,
        improve_prompt: str,
        reward_score: float,
        feedback_text: str = "",
        example: str = "",
        improvement_suggestion: str = "",
    ) -> ConversationSummary:
        """Store conversation summary with feedback and improvements"""
        try:
            return await self.db.store_conversation_summary(
                conversation_id=conversation_id,
                prompt=prompt,
                conversation_summary=conversation_summary,
                improve_prompt=improve_prompt,
                reward_score=reward_score,
                feedback_text=feedback_text,
                example=example,
                improvement_suggestion=improvement_suggestion,
            )
        except Exception as e:
            logger.error(f"Error storing conversation summary: {str(e)}")
            raise

    async def get_conversation_summary(
        self, conversation_id: str
    ) -> Optional[ConversationSummary]:
        """Retrieve conversation summary by conversation ID"""
        try:
            return await self.db.get_conversation_summary(
                conversation_id=conversation_id,
            )
        except Exception as e:
            logger.error(f"Error retrieving conversation summary: {str(e)}")
            raise

    async def update_conversation_feedback(
        self,
        conversation_id: str,
        feedback_text: str,
        reward_score: float,
        improvement_suggestion: Optional[str] = None,
        example: Optional[str] = None,
    ) -> Optional[ConversationSummary]:
        """Update conversation feedback and improvement suggestions"""
        try:
            return await self.db.update_conversation_feedback(
                conversation_id=conversation_id,
                feedback_text=feedback_text,
                reward_score=reward_score,
                improvement_suggestion=improvement_suggestion,
                example=example,
            )
        except Exception as e:
            logger.error(f"Error updating conversation feedback: {str(e)}")
            raise

    async def get_best_performing_prompts(
        self, limit: int = 5
    ) -> List[ConversationSummary]:
        """Retrieve the best performing prompts based on reward scores"""
        try:
            return await self.db.get_best_performing_prompts(
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Error retrieving best performing prompts: {str(e)}")
            raise

    async def _rerank_results(
        self,
        query: str,
        items: List[T],
        text_extractor: Callable[[T], str],
        limit: int,
        threshold: float | None = None,
    ) -> List[T]:
        """Two-stage reranking: similarity search followed by cross-encoder

        Args:
            query: Search query text
            items: List of items to rerank
            text_extractor: Function to extract text from item for reranking
            limit: Maximum number of results to return

        Returns:
            List of reranked items
        """
        if not self.use_reranker:
            return items[:limit]

        # Second stage: Rerank results
        reranked_items = self.reranker.rerank(
            query=query,
            items=items,  # Initial scores don't matter for reranking
            text_extractor=text_extractor,
            top_k=limit,
            threshold=threshold or self.similarity_threshold,
        )

        return [item for item, _ in reranked_items]

    async def query_knowledge(
        self,
        query: str,
        threshold: float | None = None,
    ) -> Tuple[List[Knowledge], List[EntityRelationship]]:
        """Query knowledge and related entities based on text query"""
        try:
            # Get query embedding
            query_embedding = await self.embedding_model.get_text_embedding(query)

            # First stage: Similarity search for knowledge
            knowledge_entries = await self.db.search_similar_knowledge(
                query_embedding=query_embedding,
                limit=self.max_search_knowledge_results,  # Get more for reranking
            )

            if not knowledge_entries:
                return [], []

            # Get related entities
            all_entities = {
                entity for entry in knowledge_entries for entity in entry.entities
            }

            all_entities_text = ", ".join(all_entities)
            all_entities_embedding = await self.embedding_model.get_text_embedding(
                all_entities_text
            )

            # First stage: Similarity search for entity relationships in a single query
            entity_relationships = await self.db.search_similar_entities(
                query_embedding=all_entities_embedding,
                limit=self.max_search_entity_results,
            )

            # Second stage: Rerank both result sets
            reranked_knowledge = await self._rerank_results(
                query=query,
                items=knowledge_entries,
                text_extractor=lambda x: x.text,
                limit=self.max_search_knowledge_results,
                threshold=threshold,
            )

            reranked_relationships = await self._rerank_results(
                query=query,
                items=entity_relationships,
                text_extractor=lambda x: x.relationship_text,
                limit=self.max_search_entity_results,
                threshold=threshold,
            )

            return reranked_knowledge, reranked_relationships

        except Exception as e:
            logger.error(f"Error in query_knowledge: {str(e)}")
            raise

    async def query_entities(
        self,
        query: str,
        threshold: float | None = None,
    ) -> Tuple[List[EntityRelationship], List[Knowledge]]:
        """Query entities and related knowledge based on entity query"""
        try:
            query_embedding = await self.embedding_model.get_text_embedding(query)

            # First stage: Similarity search for entities
            entity_relationships = await self.db.search_similar_entities(
                query_embedding=query_embedding,
                limit=self.max_search_entity_results,
            )

            if not entity_relationships:
                return [], []

            # Get related knowledge based on entity relationships
            entities_text = ", ".join(r.relationship_text for r in entity_relationships)
            entities_embedding = await self.embedding_model.get_text_embedding(
                entities_text
            )

            # First stage: Similarity search for related knowledge
            related_knowledge = await self.db.search_similar_knowledge(
                query_embedding=entities_embedding,
                vector_column="entity_embeddings",  # Added missing comma
                limit=self.max_search_knowledge_results,
            )

            # Second stage: Rerank both results
            reranked_relationships = await self._rerank_results(
                query=query,
                items=entity_relationships,
                text_extractor=lambda x: x.relationship_text,
                limit=self.max_search_entity_results,
                threshold=threshold,
            )

            reranked_knowledge = await self._rerank_results(
                query=query,
                items=related_knowledge,
                text_extractor=lambda x: x.text,
                limit=self.max_search_knowledge_results,
                threshold=threshold,
            )

            return reranked_relationships, reranked_knowledge

        except Exception as e:
            logger.error(f"Error in query_entities: {str(e)}")
            raise

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
    ) -> ShortTermMemory:
        """Store short-term memory state with embedding"""
        try:
            # Create a summary text for embedding
            summary_text = f"{last_conversation_summary}"
            summary_embedding = await self.embedding_model.get_text_embedding(
                summary_text
            )

            return await self.db.store_short_term_memory(
                conversation_id=conversation_id,
                user_info=user_info,
                last_conversation_summary=last_conversation_summary,
                recent_goal_and_status=recent_goal_and_status,
                important_context=important_context,
                agent_beliefs=agent_beliefs,
                agent_info=agent_info,
                environment_info=environment_info,
                summary_embedding=summary_embedding,
            )
        except Exception as e:
            logger.error(f"Error storing short-term memory: {str(e)}")
            raise

    async def search_similar_short_term_memories(
        self,
        query: str,
        limit: int = 3,
        threshold: float | None = None,
    ) -> List[ShortTermMemory]:
        """Search for similar short-term memories based on query text"""
        try:
            # Get query embedding
            query_embedding = await self.embedding_model.get_text_embedding(query)

            # Get similar memories
            memories = await self.db.search_similar_short_term_memories(
                query_embedding=query_embedding,
                limit=limit,
            )

            # Rerank if needed
            if self.use_reranker:
                memories = await self._rerank_results(
                    query=query,
                    items=memories,
                    text_extractor=lambda x: f"{x.last_conversation_summary}",
                    limit=limit,
                    threshold=threshold,
                )

            return memories

        except Exception as e:
            logger.error(f"Error searching short-term memories: {str(e)}")
            raise

    async def get_short_term_memory(
        self, conversation_id: Optional[str] = None
    ) -> Optional[ShortTermMemory]:
        """Retrieve the current short-term memory state

        Returns:
            Optional[ShortTermMemory]: The current memory state if it exists
        """
        try:
            return await self.db.get_short_term_memory(conversation_id=conversation_id)
        except Exception as e:
            logger.error(f"Error retrieving short-term memory: {str(e)}")
            raise

    async def get_latest_conversation_summary(self) -> Optional[ConversationSummary]:
        """Retrieve the most recent conversation summary"""
        try:
            return await self.db.get_latest_conversation_summary()
        except Exception as e:
            logger.error(f"Error retrieving latest conversation summary: {str(e)}")
            raise
