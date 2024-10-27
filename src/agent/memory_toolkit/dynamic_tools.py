from typing import List, Optional
from loguru import logger
from mirascope.core import BaseToolKit, toolkit_tool
import asyncio

from src.memory.memory_manager import MemoryManager


class DynamicMemoryToolKit(BaseToolKit):
    """Toolkit for managing memory operations including storage and retrieval."""

    __namespace__ = "memory_tools"

    memory_manager: Optional[MemoryManager]

    @toolkit_tool
    async def store_knowledge(
        self,
        knowledge_text: str,
        entities: List[str],
        keywords: List[str],
        relationship_text: List[str],
    ) -> str:
        """Store important information in memory for future use. Remember to use this when you want to store important information.


        knowledge_text (str): The main knowledge text content to store. Can be a sentence or a paragraph, or multiple paragraphs. Make sure to it cover all context of the information.
        entities (List[str]): List of entities identified in the text.
        keywords (List[str]): List of keywords from the text.
        relationship_text (List[str]): List of relationship statements (e.g. "Paris is the capital of France").

        Returns:
            str: Status message about the storage operation
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _store():
                # Search similar knowledge if exists return existing knowledge
                similar_knowledge, _ = await self.memory_manager.query_knowledge(
                    query=knowledge_text, threshold=0.9
                )
                if similar_knowledge:
                    return f"Knowledge already exists: {similar_knowledge[0].text}"

                # Get embeddings for text and entities
                text_embedding = (
                    await self.memory_manager.embedding_model.get_text_embedding(
                        knowledge_text
                    )
                )

                entity_text = " ".join(entities)
                entity_embeddings = (
                    await self.memory_manager.embedding_model.get_text_embedding(
                        entity_text
                    )
                )

                # Store knowledge
                await self.memory_manager.store_knowledge(
                    text=knowledge_text,
                    entities=entities,
                    keywords=keywords,
                    text_embedding=text_embedding,
                    entity_embeddings=entity_embeddings,
                )

                # Store entity relationships
                for rel_text in relationship_text:
                    rel_embedding = (
                        await self.memory_manager.embedding_model.get_text_embedding(
                            rel_text
                        )
                    )
                    await self.memory_manager.store_entity_relationship(
                        relationship_text=rel_text, embedding=rel_embedding
                    )

                return f"Successfully stored knowledge: {knowledge_text}"

            return await _store()

        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            return f"Error storing knowledge: {str(e)}"

    @toolkit_tool
    async def search_knowledge(self, query: str) -> str:
        """Search for relevant knowledge in memory.

        Args:
            query (str): The search query text

        Returns:
            str: Formatted string containing relevant knowledge and relationships
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _search():
                # Query both knowledge and relationships
                knowledge_entries, relationships = (
                    await self.memory_manager.query_knowledge(query=query)
                )

                # Format results
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
    async def search_entity_relationships(self, entities: List[str]) -> str:
        """Search for relationships between specified entities in memory.

        Args:
            entities (List[str]): List of entities to search relationships for

        Returns:
            str: Formatted string containing relevant entity relationships
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _search_entities():
                # Create query from entities
                query = ", ".join(entities)
                relationships, related_knowledge = (
                    await self.memory_manager.query_entities(query=query)
                )

                # Format results
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


def get_memory_toolkit(memory_manager: MemoryManager) -> DynamicMemoryToolKit:
    """Get configured memory toolkit instance.

    Args:
        memory_manager (MemoryManager): Memory manager instance to use for the toolkit

    Returns:
        DynamicMemoryToolKit: Configured DynamicMemoryToolKit instance
    """
    return DynamicMemoryToolKit(memory_manager=memory_manager)
