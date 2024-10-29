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
        description="Identify any new details about the user's preferences, personality, background, needs, styles, habits, beliefs, your relationship with user or anything you understand about user."
    )
    recent_goal_and_status: str = Field(
        description="Document the all the goals the user has set and their progress or status. Should be a list of bullet points in short and concise. If goal is completed, remove it from the list."
    )
    important_context: str = Field(
        description="Capture any significant contextual elements that should be retained for future reference. Should be compressed and short as much as possible. Should be a list of bullet points in short and concise."
    )
    agent_beliefs: str = Field(
        description="Adjust the your belife and understanding of the world based on insights gained from the conversation. Should be a list of bullet points."
    )
    agent_info: str = Field(
        description="How the conversation has changed you? Describe detailed of your personality, role, name, gender, language, style, age, profile, historical background, relationship with user and anything beyond."
    )
    environment_info: str = Field(
        description="Describe the environment you are in. Resource, tools you have, anything you can experience, observe, realize, understand, feel, etc. Objects, people, time, space, etc."
    )


@prompt_template(
    """
    MESSAGES: {history}
    
    USER:
    Extract and update your short-term memory based on conversation history. Think of what need to remember or update? And what need to forget or delete. Then write an updated short-term memory. Think of it is as writting a note for your future self. This is very important, as it updated your future behavior and decision making.
    
    USER FEEDBACK:
    {user_feedback}
    
    Updated to your current short-term memory:
    {current_memory}
    """
)
def short_term_memory_prompt(history, current_memory, user_feedback): ...


async def generate_updated_short_term_memory(
    agent: "BaseAgent", summary: str, user_feedback: str = "No user feedback provided."
) -> BaseShortTermMemoryUpdate:
    """Get updated short-term memory state after conversation."""
    try:
        prompt = short_term_memory_prompt(
            history=agent._build_prompt(include_short_term_memory=False),
            current_memory=agent.short_term_memory,
            user_feedback=user_feedback,
        )

        @retry(
            stop=stop_after_attempt(3),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.reasoning_model_name,
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

        return response
    except Exception as e:
        logger.error(f"Error generating updated short-term memory: {e}")
        raise
