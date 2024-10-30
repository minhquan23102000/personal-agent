from typing import List, Optional, TYPE_CHECKING
from loguru import logger
from mirascope.core import BaseToolKit, toolkit_tool
import asyncio
import traceback
from src.memory.memory_manager import MemoryManager


class DynamicMemoryToolKit(BaseToolKit):
    """Toolkit for managing memory operations including storage and retrieval. Use this toolkit to store and retrieve knowledge, facts, entities, relationships, and conversation contexts."""

    __namespace__ = "long_term_memory_database"

    memory_manager: MemoryManager

    duplicate_threshold: float = 0.97

    @toolkit_tool
    async def store_update_knowledge(
        self,
        knowledge_text: List[str],
        entities: List[str],
        relationship_text: List[str],
    ) -> str:
        """Save important information, knowledge, facts to your long-term memory.

        You can use this function in the following scenarios:
        - When you learn something new
        - When you have a realization
        - When you have a new idea
        - When you have a new feeling
        - When you encounter important information

        Args:
            self: self.
            knowledge_text (List[str]): A list of knowledge strings to be stored.
            entities (List[str]): A list of entities related to the knowledge.
            relationship_text (List[str]): A list of concise relationship descriptions between the entities, formatted as "entity1" "relationship" "entity2".
        """

        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            new_relationships = []
            existing_relationships = []

            if knowledge_text:

                for knowledge in knowledge_text:
                    similar_knowledge = (
                        await self.memory_manager.search_similar_knowledge(
                            query=knowledge, threshold=self.duplicate_threshold
                        )
                    )
                    if similar_knowledge:
                        continue
                    else:
                        await self.memory_manager.store_knowledge(
                            text=knowledge,
                            entities=entities,
                            keywords=entities,
                        )

            if relationship_text:
                for rel_text in relationship_text:
                    similar_rel = await self.memory_manager.search_similar_entities(
                        query=rel_text, threshold=self.duplicate_threshold
                    )

                if similar_rel:
                    existing_relationships.append(similar_rel)
                else:
                    await self.memory_manager.store_entity_relationship(
                        relationship_text=rel_text
                    )
                    new_relationships.append(rel_text)

            return "Successfully stored knowledge and entities relationships."

        except Exception as e:
            logger.error(
                f"Error storing knowledge: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error storing knowledge: {str(e)}"

    @toolkit_tool
    async def search_knowledge_facts(self, query: str) -> str:
        """Search knowledge, facts from your long-term memory.

        You can use this function in the following scenarios:
        - When you need to recall important information
        - When you need to find a specific fact
        - When you need to find related information

        Args:
            query (str): The query string used to search for relevant knowledge.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            knowledge_entries, relationships = (
                await self.memory_manager.query_knowledge(
                    query=query, threshold=self.duplicate_threshold
                )
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

            return "\n".join(results) if results else "No relevant information found."

        except Exception as e:
            logger.error(
                f"Error searching knowledge: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error searching knowledge: {str(e)}. Traceback: {traceback.format_exc()}"

    @toolkit_tool
    async def search_entities_facts(self, entities: List[str]) -> str:
        """Search knowledge, facts of entities or relationships between entities.

        You can use this function in the following scenarios:
        - When you need to find connections between entities
        - When you need to find relationships between entities
        - When you need to find information, knowledge, or facts about entities

        Args:
            entities (List[str]): A list of entity names to search for relationships and knowledge.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            query = ", ".join(entities)
            relationships, related_knowledge = await self.memory_manager.query_entities(
                query=query
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

        except Exception as e:
            logger.error(
                f"Error searching relationships: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error searching relationships: {str(e)}. Traceback: {traceback.format_exc()}"

    @toolkit_tool
    async def store_update_entities_facts(
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

        Args:
            self: self.
            entities (List[str]): A list of entity names to store relationships for.
            relationship_text (List[str]): A list of concise relationship descriptions between the entities, formatted as "entity1" "relationship" "entity2".
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            for rel_text in relationship_text:
                similar_rel = await self.memory_manager.search_similar_entities(
                    query=rel_text, threshold=self.duplicate_threshold
                )

                if similar_rel:
                    continue

                await self.memory_manager.store_entity_relationship(
                    relationship_text=rel_text
                )

            return f"Successfully stored entities relationships."

        except Exception as e:
            logger.error(
                f"Error storing entity relationship: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error storing entity relationship: {str(e)}. Traceback: {traceback.format_exc()}"

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

        Args:
            query (str): The query string to search for similar conversation contexts.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            similar_memories = (
                await self.memory_manager.search_similar_short_term_memories(
                    query=query,
                    limit=3,
                    threshold=0.65,
                )
            )

            if not similar_memories:
                return "No similar conversation contexts found."

            results = [
                f"Top {len(similar_memories)} Similar Conversation Contexts (May not be relevant, use with caution):"
            ]

            for memory in similar_memories:
                context_summary = [
                    f"\nConversation id: {memory.conversation_id}\n",
                    f"Summary:\n{memory.last_conversation_summary}\n",
                    f"Goal & Status:\n{memory.recent_goal_and_status}\n",
                    f"Important Context:\n{memory.important_context}\n",
                    f"User Info:\n{memory.user_info}\n",
                    f"Agent Beliefs:\n{memory.agent_beliefs}\n",
                    f"Environment Info:\n{memory.environment_info}\n",
                    f"Conversation ended at: {memory.timestamp}\n",
                ]
                results.extend(context_summary)

            return "\n".join(results)

        except Exception as e:
            logger.error(
                f"Error searching conversation contexts: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error searching conversation contexts: {str(e)}. Traceback: {traceback.format_exc()}"

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
