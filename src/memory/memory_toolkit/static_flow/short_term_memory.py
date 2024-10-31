import inspect
from mirascope.core import prompt_template, litellm, Messages
from pydantic import BaseModel, Field
from typing import TYPE_CHECKING
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import AfterValidator, ValidationError

from loguru import logger

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent

from src.agent.error_prompt import format_error_message


class BaseShortTermMemoryUpdate(BaseModel):
    """Model for short-term memory updates."""

    user_info: str = Field(
        description="Any new details regarding the user's preferences, personality, background, needs, styles, habits, beliefs, your relationship with the user, how you addressed the user, or any other relevant information you have about the user."
    )
    important_context: str = Field(
        description="Capture any significant contextual elements that should be retained for future reference. Should be compressed and short as much as possible. Should be a list of bullet points in short and concise."
    )
    agent_beliefs: str = Field(
        description="Adjust the your belife and understanding of the world based on insights gained from the conversation. Should be a list of bullet points."
    )
    agent_info: str = Field(
        description="How the conversation has changed you? Note and update your personality, role, name, gender, language, style, age, profile, historical background, anything beyond."
    )
    recent_goal_and_status: str = Field(
        description="Document the all the goals the user has set and their progress or status. Should be a list of bullet points in short and concise. If goal is completed, remove it from the list."
    )
    environment_info: str = Field(
        description="Describe the environment (both digital environment or physical environment) you are in. Anything you can experience, observe, realize, understand, feel, etc. Objects, resources, people, time, space, etc. Only note what is most meaningful and useful to you."
    )


@prompt_template(
    """
    MESSAGES: {history}
    
    USER:
    Extract and update your short-term memory based on conversation history. Think of what need to remember or update? And what need to forget or delete? Then write an updated short-term memory. Think of it is as writting a note for your future self. This is very important, as it updated your future behavior, personality, and decision making. So be very careful and thoughtful.
    
    USER FEEDBACK:
    <>
    {user_feedback}
    </>
    
    Updated to your current short-term memory:
    <>
    {current_memory}
    </>
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
            model=agent.slow_model_name,
            response_model=BaseShortTermMemoryUpdate,
            json_mode=True,
        )
        def call(*, errors: list[ValidationError] | None = None):
            config = {}

            config["messages"] = prompt

            if errors:
                config["computed_fields"] = {
                    "previous_errors": f"Previous Errors: {format_error_message(errors)}"
                }

            return config

        response = call()

        return response
    except Exception as e:
        logger.error(f"Error generating updated short-term memory: {e}")
        raise
