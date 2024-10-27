from typing import List, Optional, TYPE_CHECKING
from loguru import logger
from mirascope.core import BaseToolKit, toolkit_tool
import asyncio

from src.memory.memory_manager import MemoryManager


class DynamicMemoryToolKit(BaseToolKit):
    """Toolkit for managing memory operations including storage and retrieval."""

    __namespace__ = "memory_tools"

    memory_manager: MemoryManager

    @toolkit_tool
    async def store_knowledge(
        self,
        knowledge_text: str,
        entities: List[str],
        relationship_text: List[str],
    ) -> str:
        """Store Knowledge in Your Long Term Memory. Call this method when you learn new knowledge in the conversation.
        This method allows you to remember important knowledge and information for future use.
        It checks if the knowledge already exists in memory to avoid duplicates. If the knowledge
        is new, it stores the text along with associated entities and their relationships.
        For Example:
            result = await dynamic_memory_toolkit.store_knowledge(
                knowledge_text="Python is a programming language.",
                entities=["Python", "Programming Language"],
                relationship_text=["Python is used for web development.", "Python is popular for data science."]
            )
            print(result)  # Outputs: Successfully stored knowledge: Python is a programming language.

        Args:
            self: The DynamicMemoryToolKit instance. Ignore this parameter.
            knowledge_text (str): The text of the knowledge to be stored.
            entities (List[str]): A list of entities related to the knowledge.
            relationship_text (List[str]): A list of relationship descriptions related to the entities.
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
        """Search for relevant knowledge in memory.
        This function allows you to search for knowledge entries stored in the memory manager based on a given query. It retrieves relevant knowledge and associated entity relationships, providing a structured response.
        Ensure that the query is specific enough to yield relevant results.
        Use keywords that are likely to match the stored knowledge for better search accuracy.
        Handle the returned string appropriately, especially if it indicates no results were found.
        For Example:
            To search for knowledge related to "machine learning", you would call,
            result = await search_knowledge(query="machine learning")
            print(result)

        Args:
            query (str): The search term or phrase used to find relevant knowledge in memory.
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
        """Search for relationships between specified entities in memory.
        This function allows you to find and retrieve relationships between specified entities stored in the memory manager. It provides insights into how entities are connected and any related knowledge that may enhance understanding.
        Provide a clear and concise list of entities to ensure accurate results.
        Use entity names that are commonly recognized within the context of your application.
        Be aware that the function may return no results if the entities are not related or if they do not exist in memory.
        For Example:
            To search for relationships between entities "Alice" and "Bob", you would call,
            result = await search_entity_relationships(entities=["Alice", "Bob"])
            print(result)

        Args:
            entities (List[str]): A list of entity names for which relationships are to be searched. Ensure that the entities are accurately spelled and relevant to the context of the search.
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


def get_memory_toolkit(memory_manager: "MemoryManager") -> DynamicMemoryToolKit:
    """Get configured memory toolkit instance."""
    return DynamicMemoryToolKit(memory_manager=memory_manager)
