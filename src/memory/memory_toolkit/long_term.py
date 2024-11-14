from typing import List, Optional, TYPE_CHECKING
from loguru import logger
from mirascope.core import BaseToolKit, toolkit_tool
import asyncio
import traceback
from src.memory.memory_manager import MemoryManager, SearchResult
import inspect


class LongtermMemoryToolKit(BaseToolKit):
    """Toolkit for managing memory operations including storage and retrieval. Use this toolkit to store and retrieve knowledge, facts, entities, relationships, and conversation contexts."""

    __namespace__ = "long_term_memory_database"

    memory_manager: MemoryManager

    similarity_threshold: float = 0.7

    @toolkit_tool
    async def search_knowledge_facts(self, query: str) -> str:
        """Search knowledge, facts from your long-term memory in external long term memory database.

        You can use this function in the following scenarios:
        - When you need to recall important information
        - When you need to find a specific fact
        - When you need to find related knowledges

        Use this when you do not have context in the current moment for a specific topic user asking or for task you are working on.

        Args:
            query (str): The query string used to search for relevant knowledge.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            knowledge_entries, relationships = (
                await self.memory_manager.query_knowledge(
                    query=query, threshold=self.similarity_threshold
                )
            )

            results = []
            if knowledge_entries:
                results.append("Relevant Knowledge:")
                for entry in knowledge_entries:
                    results.append(
                        f"- {entry.item.text} (similarity: {entry.score:.2f})"
                    )

            if relationships:
                results.append("\nEntity Relationships:")
                for rel in relationships:
                    results.append(
                        f"- {rel.item.relationship_text} (similarity: {rel.score:.2f})"
                    )

            return "\n".join(results) if results else "No relevant information found."

        except Exception as e:
            logger.error(
                f"Error searching knowledge: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error searching knowledge: {str(e)}. Traceback: {traceback.format_exc()}"

    @toolkit_tool
    async def recall_similar_conversation_contexts(self, context_query: str) -> str:
        """Search for similar conversation contexts from your memory to help understand the current situation better.

        You can use this function in following scenarios:
        - When you need to recall how you handled similar conversations in the past
        - When you want to understand the context of similar situations

        Use this when you do not have context in the current moment for a specific topic user asking or for task you are working on.

        Args:
            context_query (str): Summary of the current conversation context as a context query string.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            similar_memories = (
                await self.memory_manager.search_similar_context_memories(
                    query=context_query,
                    limit=3,
                    threshold=max(self.similarity_threshold - 0.3, 0.1),
                )
            )

            if not similar_memories:
                return "No similar conversation contexts found."

            results = [
                f"Top {len(similar_memories)} Similar Conversation Contexts (May not be relevant, use with caution):"
            ]

            for memory in similar_memories:
                context_summary = inspect.cleandoc(
                    f"""
                    <>
                    Conversation id: {memory.item.conversation_id}
                    Summary: {memory.item.last_conversation_summary}
                    Goal & Status: {memory.item.recent_goal_and_status}
                    Important Context: {memory.item.important_context}
                    User Info: {memory.item.user_info}
                    Agent Beliefs: {memory.item.agent_beliefs}
                    Environment Info: {memory.item.environment_info}
                    Conversation ended at: {memory.item.timestamp}
                    Conversation similarity score: {memory.score:.2f}
                    </>
                    """
                )

                results.append(context_summary)

            return "\n".join(results)

        except Exception as e:
            logger.error(
                f"Error searching conversation contexts: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error searching conversation contexts: {str(e)}. Traceback: {traceback.format_exc()}"

    @toolkit_tool
    async def recall_conversation_details(self, conversation_id: str) -> str:
        """Retrieve all messages between user and assistant for a specific conversation.

        You can use this function in following scenarios:
        - When you need to understand the full context of a past conversation
        - When you need to analyze how a particular conversation evolved

        Args:
            conversation_id (str): The unique identifier of the conversation to retrieve. This should be a valid conversation ID from the recent conversations context.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            # Get conversation details for user and assistant only
            conversation_messages = await self.memory_manager.get_conversation_details(
                conversation_id=conversation_id,
                senders=["user", "assistant"],
                limit=70,  # Reasonable limit for conversation length
            )

            if not conversation_messages:
                return f"No conversation found with ID: {conversation_id}."

            # Format conversation in chronological order
            messages = sorted(conversation_messages, key=lambda x: x.turn_id)

            # Build formatted conversation string
            conversation_lines = [
                f"Conversation ID: {conversation_id}\n",
                "Conversation History:",
            ]

            for msg in messages:
                # Format timestamp to be more readable
                timestamp = msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                role = "User" if msg.sender == "user" else "Assistant"
                conversation_lines.append(
                    f"\n[{timestamp}] {role}:\n{msg.message_content}"
                )

            return "\n".join(conversation_lines)

        except Exception as e:
            logger.error(
                f"Error retrieving conversation details: {e}. Traceback: {traceback.format_exc()}"
            )
            return f"Error retrieving conversation details: {str(e)}"


def get_memory_toolkit(memory_manager: "MemoryManager") -> LongtermMemoryToolKit:
    """Get configured memory toolkit instance."""
    return LongtermMemoryToolKit(memory_manager=memory_manager)
