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
        description="Identify any new details about the user's preferences, personality, background, needs, hypothesis about user, or anything else important beyond that emerged during the conversation. Should be a list of bullet points. Should be compressed as much as possible."
    )
    recent_goal_and_status: str = Field(
        description="Document the current goals the user has set and their progress or status. Should be a list of bullet points. Should be compressed as much as possible."
    )
    important_context: str = Field(
        description="Capture any significant contextual elements that should be retained for future reference. Should be compressed as much as possible. Should be a list of bullet points."
    )
    agent_beliefs: str = Field(
        description="Adjust the your understanding of the user's intentions and the world based on insights gained from the conversation. What you have learned about the user and the world? Should be a list of bullet points. Should be compressed as much as possible."
    )
    agent_info: str = Field(
        description="How the conversation has changed you? An updated of your personality, role, name, gender, language, style, age, profile, history experience, habbits, hobbies, likes, hates, and anything beyond. Should be a list of bullet points. Should be compressed as much as possible."
    )


@prompt_template(
    """
    MESSAGES: {history}
    
    USER:
    Extract and update your short-term memory based on conversation history. Think of what is important, what is not important. Then write an updated short-term memory.
    """
)
def short_term_memory_prompt(history, current_memory): ...


async def generate_updated_short_term_memory(
    agent: "BaseAgent", summary: str
) -> BaseShortTermMemoryUpdate:
    """Get updated short-term memory state after conversation."""
    try:
        prompt = short_term_memory_prompt(
            history=agent.history,
            current_memory=agent.short_term_memory,
        )

        @retry(
            stop=stop_after_attempt(3),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.slow_model_name,
            response_model=BaseShortTermMemoryUpdate,
            json_mode=True,
        )
        def call(*, errors: list[ValidationError] | None = None):
            config = {}

            config["messages"] = prompt

            if errors:
                config["computed_fields"] = {
                    "previous_errors": f"Previous Errors: {errors}"
                }

            return config

        response = call()

        response.important_context += f"\n\nLast conversation summary: {summary}"

        return response
    except Exception as e:
        logger.error(f"Error generating updated short-term memory: {e}")
        raise
