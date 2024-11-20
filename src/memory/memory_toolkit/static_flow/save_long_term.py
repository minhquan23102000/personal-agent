import traceback
from typing import List
from pydantic import BaseModel, Field
from mirascope.core import prompt_template, gemini, litellm
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import ValidationError
from loguru import logger
from src.core.prompt.error_prompt import format_error_message
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent
    from src.memory.memory_manager import MemoryManager


class BaseLongTermMemory(BaseModel):
    knowledge_text: List[str] = Field(
        description="List of important knowledge or facts extracted from the conversation"
    )
    entities: List[str] = Field(description="List of important entities mentioned")
    entities_relationships: List[str] = Field(
        description="List of relationships between entities in format 'object relationship subject'"
    )


@prompt_template(
    """
    MESSAGES: {history}
    
    USER:
    Extract important knowledge, entities, and relationships from the conversation above that should be stored in long-term memory.
    Focus on factual information, knowledge, key insights, and meaningful relationships between concepts or entities.
    
    Format each knowledge item as a separate, clear statement.
    Format relationships as short and concise "object" "relationship" "subject".
    
    !! Important: Capture all essential information in this single interaction. Because you will only have one chance to store the information.
    
    Additional Context: 
    {additional_context}
    """
)
def base_long_term_memory_prompt(history, additional_context): ...


async def save_long_term_memory(
    agent: "BaseAgent",
    additional_context: str = "",
    include_message_history: bool = True,
) -> None:
    """Extract and save important information to long-term memory."""
    try:
        if not agent.memory_manager:
            return

        prompt = base_long_term_memory_prompt(
            history=agent._build_prompt(include_history=include_message_history),
            additional_context=additional_context,
        )

        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.reflection_model,
            response_model=BaseLongTermMemory,
            json_mode=True,
        )
        def call(*, errors: list[ValidationError] | None = None):
            config = {}
            agent.rotate_api_key()

            config["messages"] = prompt

            if errors:
                config["computed_fields"] = {
                    "previous_errors": f"Previous Errors: {format_error_message(errors)}"
                }

            return config

        memory_content = call()

        # Store the extracted information using DynamicMemoryToolKit methods

        await store_update_knowledge(
            memory_manager=agent.memory_manager,
            knowledge_text=memory_content.knowledge_text,
            entities=memory_content.entities,
            entities_relationship=memory_content.entities_relationships,
        )

        logger.info("Successfully saved long-term memory content")

    except Exception as e:
        logger.error(f"Error saving long-term memory: {e}")
        raise e


async def store_update_knowledge(
    memory_manager: "MemoryManager",
    knowledge_text: List[str],
    entities: List[str],
    entities_relationship: List[str],
) -> str:
    """Save important information, knowledge, facts to  long-term memory.

    Args:
        self: self.
        knowledge_text (List[str]): A list of knowledge strings to be stored. Each knowledge should be clear separated from each other, unless they have the same context. Do not join multiple unrelated knowledge into one sentence.
        entities (List[str]): A list of entities related to the knowledge.
        entities_relationship (List[str]): A list of concise and very short relationship between the entities, formatted as "entity1" "relationship" "entity2". Focus on the relationship between the entities, not the facts or knowledges.
    """

    duplicate_threshold = 0.93
    try:

        new_relationships = []
        existing_relationships = []

        return_msg = ""
        if knowledge_text:

            for knowledge in knowledge_text:

                await memory_manager.store_knowledge(
                    text=knowledge,
                    entities=entities,
                    keywords=entities,
                )

        if entities_relationship:
            for rel_text in entities_relationship:

                await memory_manager.store_entity_relationship(
                    relationship_text=rel_text
                )
                new_relationships.append(rel_text)

        return f"Successfully stored knowledge: {knowledge_text}, entities: {entities}, entites relationships: {entities_relationship} in Long Term Database."

    except Exception as e:
        logger.error(
            f"Error storing knowledge: {e}. Traceback: {traceback.format_exc()}"
        )
        return f"Error storing knowledge: {str(e)}"
