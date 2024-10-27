import inspect
from mirascope.core import prompt_template, litellm, Messages
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING
from src.memory.memory_toolkit.static_flow.conversation_summary import (
    BaseConversationSummary,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import AfterValidator, ValidationError

from loguru import logger

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


class BaseShortTermMemoryUpdate(BaseModel):
    """Model for short-term memory updates."""

    user_info: str = Field(
        description="Identify any new details about the user's preferences, background, or needs that emerged during the conversation."
    )
    recent_goal_and_status: str = Field(
        description="Document the current goals the user has set and their progress or status."
    )
    important_context: str = Field(
        description="Capture any significant contextual elements that should be retained for future reference."
    )
    agent_beliefs: str = Field(
        description="Adjust the agent's understanding of the user's intentions and the world based on insights gained from the conversation."
    )
    agent_info: str = Field(
        description="Confirm and update details about the agent, including its personality, role, name, gender, language, style, age, profile, history experience, habbits, hobbies, likes, hates, and beyond. Remember agent should have their personal identity, and not easily to change."
    )


def short_term_memory_prompt(history, summary, current_memory):
    return [
        Messages.System(
            inspect.cleandoc(
                f"""
                ### Instructions:
                Extract and update an agent's short-term memory based on conversation history. The updated memory should reflect the latest user information, goals, context, and agent beliefs, ensuring relevant information is retained for future interactions.
                
                ### Output Style
                Concise and factual. Use bullet points or a similar structured format to present the updated memory. Compress the information as much as possible, but do not lose any important details.

                ### Output Rules
                * Focus on the most recent and relevant information.
                * Provide a clear status of ongoing tasks.
                * Maintain important context for future interactions.
                * Do not include sensitive information (e.g., passwords, financial details).

                ### Supplementary Information
                * **Recency Bias:** Prioritize recent information as it is likely to be more relevant.
                * **Priming:**  Consider how previous interactions might influence the current conversation.
                * **Contextual Awareness:**  Pay attention to the broader context of the conversation.
                
            
                ### Current Short-Term Memory:
                {current_memory}
                
                ### Conversation History:
                {history}
                
                ### Conversation Summary:
                {summary}
            
                """
            )
        )
    ]


async def generate_updated_short_term_memory(
    agent: "BaseAgent", summary: BaseConversationSummary
) -> BaseShortTermMemoryUpdate:
    """Get updated short-term memory state after conversation."""
    try:
        prompt = short_term_memory_prompt(
            history=agent.history,
            summary=summary,
            current_memory=agent.short_term_memory,
        )

        @retry(
            stop=stop_after_attempt(3),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.model_name,
            response_model=BaseShortTermMemoryUpdate,
            json_mode=True,
        )
        def call(*, errors: list[ValidationError] | None = None):
            config = {}

            config["messages"] = short_term_memory_prompt(
                history=agent.history,
                summary=summary,
                current_memory=agent.short_term_memory,
            )

            if errors:
                config["computed_fields"] = {
                    "previous_errors": f"Previous Errors: {errors}"
                }

            return config

        response = call()

        return response
    except Exception as e:
        logger.error(f"Error generating updated short-term memory: {e}")
        raise
