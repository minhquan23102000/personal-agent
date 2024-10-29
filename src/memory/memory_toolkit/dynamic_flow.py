from typing import List, Optional, TYPE_CHECKING
from loguru import logger
from mirascope.core import BaseToolKit, toolkit_tool
import asyncio

from src.memory.memory_manager import MemoryManager


class DynamicMemoryToolKit(BaseToolKit):
    """Toolkit for managing memory operations including storage and retrieval."""

    __namespace__ = "long_term_knowledge_base"

    memory_manager: MemoryManager

    @toolkit_tool
    async def store_knowledge(
        self,
        knowledge_text: List[str],
        entities: List[str],
        relationship_text: List[str],
    ) -> str:
        """Save important information to your long-term memory. Whenever you ensures the information is retained even across different conversations or sessions and for long term benefit.

        You can use this function in following scenarios:
        - When you learn something new
        - When you have a realization
        - When you have a new idea
        - When you have a new feeling
        - When you encounter important information.

        To store knowledge, provide the list of facts, knowledges or informations you want to remember, a list of relevant entities names associated with it, and descriptions of how those entities relate to each other (For example: "John is a friend of Mary", "Paris is in France", "Water is wet", "human have emotions", "emotions guide human behavior").
        """

        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            new_relationships = []
            existing_relationships = []

            for knowledge in knowledge_text:
                similar_knowledge, _ = await self.memory_manager.query_knowledge(
                    query=knowledge, threshold=0.9
                )
                if similar_knowledge:
                    continue
                else:
                    text_embedding = (
                        await self.memory_manager.embedding_model.get_text_embedding(
                            knowledge
                        )
                    )

                    entity_text = ", ".join(entities)
                    entity_embeddings = (
                        await self.memory_manager.embedding_model.get_text_embedding(
                            entity_text
                        )
                    )

                    await self.memory_manager.store_knowledge(
                        text=knowledge,
                        entities=entities,
                        keywords=entities,
                        text_embedding=text_embedding,
                        entity_embeddings=entity_embeddings,
                    )

            for rel_text in relationship_text:
                similar_rel, _ = await self.memory_manager.query_entities(
                    rel_text, threshold=0.9
                )

                if similar_rel:
                    existing_relationships.append(similar_rel)
                else:
                    rel_embedding = (
                        await self.memory_manager.embedding_model.get_text_embedding(
                            rel_text
                        )
                    )

                    await self.memory_manager.store_entity_relationship(
                        relationship_text=rel_text, embedding=rel_embedding
                    )
                    new_relationships.append(rel_text)

            return "Successfully stored knowledge and relationships."

        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            return f"Error storing knowledge: {str(e)}"

    @toolkit_tool
    async def search_knowledge_facts(self, query: str) -> str:
        """Retrieve and search relevant knowledge from your long-term memory.

        You can use this function in following scenarios:
        - When you need to recall important information
        - When you need to find a specific fact
        - When you need to find related information
        - When you think that your current memory is incomplete or lacking knowledge to complete a task or answer a question.

        Search the knowledge base using the given query to locate pertinent information and any associated entities.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _search():
                knowledge_entries, relationships = (
                    await self.memory_manager.query_knowledge(query=query)
                )

                results = []
                if knowledge_entries:
                    results.append("Relevant Knowledge:")
                    for entry in knowledge_entries:
                        results.append(f"- {entry.text}")

                if relationships:
                    results.append("\nEntity Relationships:")
                    for rel in relationships:
                        results.append(f"- {rel.relationship_text}")

                return (
                    "\n".join(results) if results else "No relevant information found."
                )

            return await _search()

        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return f"Error searching knowledge: {str(e)}"

    @toolkit_tool
    async def search_entities_facts(self, entities: List[str]) -> str:
        """Retrieve and search knowledge, facts of entities orrelationships between entities.

        You can use this function in following scenarios:
        - When you need to find connections between entities
        - When you need to find relationships between entities
        - When you need to find information, knowledge or facts about entities
        - When you think that your current memory is incomplete or lacking knowledge to complete a task or answer a question.

        To search for relationships between entities, provide a list of entity names.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _search_entities():
                query = ", ".join(entities)
                relationships, related_knowledge = (
                    await self.memory_manager.query_entities(query=query)
                )

                results = []
                if relationships:
                    results.append("Entity Relationships:")
                    for rel in relationships:
                        results.append(f"- {rel.relationship_text}")

                if related_knowledge:
                    results.append("\nRelated Knowledge:")
                    for entry in related_knowledge:
                        results.append(f"- {entry.text}")

                return (
                    "\n".join(results)
                    if results
                    else "No relationships found between these entities."
                )

            return await _search_entities()

        except Exception as e:
            logger.error(f"Error searching relationships: {e}")
            return f"Error searching relationships: {str(e)}"

    @toolkit_tool
    async def store_entities_relationship(
        self,
        entities: List[str],
        relationship_text: List[str],
    ) -> str:
        """Store relationships between entities in your long-term memory.

        You can use this function in following scenarios:
        - When you discover connections between entities
        - When you learn how entities are related to each other
        - When you want to explicitly store relationships without associated knowledge
        - When you need to update or add new relationships between known entities

        To use this tool, provide a list of entity names and a list of descriptions of how the entities are related. (For example: "John is a friend of Mary", "Paris is in France", "Water is wet", "human have emotions", "emotions guide human behavior").
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            for rel_text in relationship_text:
                similar_rel, _ = await self.memory_manager.query_entities(
                    rel_text, threshold=0.93
                )

                if similar_rel:
                    continue

                rel_embedding = (
                    await self.memory_manager.embedding_model.get_text_embedding(
                        rel_text
                    )
                )

                await self.memory_manager.store_entity_relationship(
                    relationship_text=rel_text,
                    embedding=rel_embedding,
                )

            return f"Successfully stored relationship between entities."

        except Exception as e:
            logger.error(f"Error storing entity relationship: {e}")
            return f"Error storing entity relationship: {str(e)}"

    @toolkit_tool
    async def recall_similar_conversation_contexts(self, query: str) -> str:
        """Search for similar conversation contexts from your memory to help understand the current situation better.

        You can use this function in following scenarios:
        - When you need to recall how you handled similar conversations in the past
        - When you want to understand the context of similar situations
        - When you need to maintain consistency in your responses across similar conversations
        - When you want to learn from past interactions to improve current responses
        - When you need to understand patterns in user behavior or conversation flow

        The function will search through past conversation contexts using semantic similarity and return the most relevant ones.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _search_contexts():
                similar_memories = (
                    await self.memory_manager.search_similar_short_term_memories(
                        query=query,
                        limit=3,
                        threshold=0.7,
                    )
                )

                if not similar_memories:
                    return "No similar conversation contexts found."

                results = ["Similar Conversation Contexts:"]

                for memory in similar_memories:
                    context_summary = [
                        f"\nConversation id: {memory.conversation_id}",
                        f"Summary: {memory.last_conversation_summary}",
                        f"Goal & Status: {memory.recent_goal_and_status}",
                        f"Important Context: {memory.important_context}",
                        f"User Info: {memory.user_info}",
                        f"Agent Beliefs: {memory.agent_beliefs}",
                        f"Environment Info: {memory.environment_info}",
                        f"Conversation ended at: {memory.timestamp}",
                    ]
                    results.extend(context_summary)

                return "\n".join(results)

            return await _search_contexts()

        except Exception as e:
            logger.error(f"Error searching conversation contexts: {e}")
            return f"Error searching conversation contexts: {str(e)}"

    # @toolkit_tool
    # async def recall_conversation_by_id(self, conversation_id: str) -> str:
    #     """Retrieve the specific conversation context using its ID.

    #     You can use this function in following scenarios:
    #     - When you need to recall the exact details of a specific conversation
    #     - When you want to reference or continue a previous conversation
    #     - When you need to verify or check specific details from a past interaction
    #     - When you need to maintain continuity across conversation sessions

    #     Provide the conversation_id to get the detailed context of that specific conversation.
    #     """
    #     try:
    #         if not self.memory_manager:
    #             return "Error: Memory manager not available"

    #         memory = await self.memory_manager.get_short_term_memory(
    #             conversation_id=conversation_id
    #         )

    #         if not memory:
    #             return f"No conversation context found for ID: {conversation_id}"

    #         context_details = [
    #             f"Conversation Context for ID: {conversation_id}",
    #             f"\nSummary: {memory.last_conversation_summary}",
    #             f"Goal & Status: {memory.recent_goal_and_status}",
    #             f"Important Context: {memory.important_context}",
    #             f"User Info: {memory.user_info}",
    #             f"Agent Beliefs: {memory.agent_beliefs}",
    #             f"Agent Info: {memory.agent_info}",
    #             f"Environment Info: {memory.environment_info}",
    #             f"\nTimestamp: {memory.timestamp}",
    #         ]

    #         return "\n".join(context_details)

    #     except Exception as e:
    #         logger.error(f"Error retrieving conversation context: {e}")
    #         return f"Error retrieving conversation context: {str(e)}"


def get_memory_toolkit(memory_manager: "MemoryManager") -> DynamicMemoryToolKit:
    """Get configured memory toolkit instance."""
    return DynamicMemoryToolKit(memory_manager=memory_manager)
