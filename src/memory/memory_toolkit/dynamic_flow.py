from typing import List, Optional, TYPE_CHECKING
from loguru import logger
from mirascope.core import BaseToolKit, toolkit_tool
import asyncio

from src.memory.memory_manager import MemoryManager


class DynamicMemoryToolKit(BaseToolKit):
    """Toolkit for managing memory operations including storage and retrieval."""

    __namespace__ = "long_term_memory_operations"

    memory_manager: MemoryManager

    @toolkit_tool
    async def remember_knowledge(
        self,
        knowledge_text: str,
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

        To store knowledge, provide the text you want to remember, a list of relevant entities associated with it, and descriptions of how those entities relate to each other.
        """

        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _store():

                return_msg = ""
                new_relationships = []
                existing_relationships = []

                similar_knowledge, _ = await self.memory_manager.query_knowledge(
                    query=knowledge_text, threshold=0.9
                )
                if similar_knowledge:
                    return_msg += (
                        f"Knowledge already exists: {similar_knowledge[0].text}\n"
                    )
                else:
                    text_embedding = (
                        await self.memory_manager.embedding_model.get_text_embedding(
                            knowledge_text
                        )
                    )

                    entity_text = ", ".join(entities)
                    entity_embeddings = (
                        await self.memory_manager.embedding_model.get_text_embedding(
                            entity_text
                        )
                    )

                    await self.memory_manager.store_knowledge(
                        text=knowledge_text,
                        entities=entities,
                        keywords=entities,
                        text_embedding=text_embedding,
                        entity_embeddings=entity_embeddings,
                    )

                    return_msg += f"Stored new knowledge {knowledge_text} sucessfully."

                for rel_text in relationship_text:
                    similar_rel, _ = await self.memory_manager.query_entities(
                        rel_text, threshold=0.9
                    )

                    if similar_rel:
                        existing_relationships.append(similar_rel)
                    else:
                        rel_embedding = await self.memory_manager.embedding_model.get_text_embedding(
                            rel_text
                        )

                        await self.memory_manager.store_entity_relationship(
                            relationship_text=rel_text, embedding=rel_embedding
                        )
                        new_relationships.append(rel_text)

                if new_relationships:
                    return_msg += f"Stored new entity relationships sucessfully: {new_relationships}\n"

                if existing_relationships:
                    return_msg += f"Entity relationships already exist: {existing_relationships}\n"

                return return_msg

            return await _store()

        except Exception as e:
            logger.error(f"Error storing knowledge: {e}")
            return f"Error storing knowledge: {str(e)}"

    @toolkit_tool
    async def search_knowledge(self, query: str) -> str:
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
    async def search_entity_relationships(self, entities: List[str]) -> str:
        """Retrieve and search knowledge of entities orrelationships between entities.

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
    async def remember_entities_relationship(
        self,
        entities: List[str],
        relationship_text: str,
    ) -> str:
        """Store relationships between entities in your long-term memory.

        You can use this function in following scenarios:
        - When you discover connections between entities
        - When you learn how entities are related to each other
        - When you want to explicitly store relationships without associated knowledge
        - When you need to update or add new relationships between known entities

        To use this tool, provide a list of entity names and a description of how the entities are related.
        """
        try:
            if not self.memory_manager:
                return "Error: Memory manager not available"

            async def _store_relationship():
                similar_rel, _ = await self.memory_manager.query_entities(
                    relationship_text, threshold=0.95
                )

                if similar_rel:
                    return f"Relationship already exists: {similar_rel[0].relationship_text}"

                rel_embedding = (
                    await self.memory_manager.embedding_model.get_text_embedding(
                        relationship_text
                    )
                )

                await self.memory_manager.store_entity_relationship(
                    relationship_text=relationship_text,
                    embedding=rel_embedding,
                )

                return f"Successfully stored relationship between entities {', '.join(entities)}: {relationship_text}"

            return await _store_relationship()

        except Exception as e:
            logger.error(f"Error storing entity relationship: {e}")
            return f"Error storing entity relationship: {str(e)}"


def get_memory_toolkit(memory_manager: "MemoryManager") -> DynamicMemoryToolKit:
    """Get configured memory toolkit instance."""
    return DynamicMemoryToolKit(memory_manager=memory_manager)
