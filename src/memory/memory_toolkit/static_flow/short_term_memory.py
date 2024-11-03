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

from src.core.prompt.error_prompt import format_error_message


class BaseShortTermMemoryUpdate(BaseModel):
    """Model for short-term memory updates."""

    user_info: str = Field(
        description="Provide a comprehensive overview of the user's preferences, personality traits, background, needs, styles, habits, beliefs, and our relationship, ensuring all relevant details are included for a better understanding of the user."
    )
    how_to_address_user: str = Field(
        description="How you address the user? Give a concise and short note. Keep language simple and direct."
    )
    important_context: str = Field(
        description="Capture any significant contextual elements that should be retained for future reference. Should be a list of bullet points in short and concise."
    )
    agent_beliefs: str = Field(
        description="Create a bullet-point list that reflects how your beliefs and understanding of the world have evolved based on insights gained from our conversation."
    )
    agent_info: str = Field(
        description="Reflect on how this conversation has influenced your personality and characteristics. Update your profile to include changes in your name, gender, language, style, age, role, historical background and anything important about you based on the discussion."
    )
    recent_goal_and_status: str = Field(
        description="Document the all the goals the user has set and their progress or status. Should be a list of bullet points in short and concise. If goal is completed, remove it from the list."
    )
    environment_info: str = Field(
        description="Provide a detailed description of your surroundings, including sensory experiences, notable objects, tools or resources, the presence of people and their interactions, as well as your thoughts, feelings, and any insights or hypotheses about the environment."
    )


@prompt_template(
    """
    MESSAGES: {history}
    
    USER:
    USER FEEDBACK:
    <>
    {user_feedback}
    </>
    
    USER:
    Review the recent conversation history to extract and summarize key points that are essential for future interactions. Create a concise list highlighting these important details while identifying and eliminating any irrelevant or outdated information. Avoid adding unnecessary details. Focus on clarity and relevance to enhance future behavior and decision-making. Treat this as a note for your future self.
    
    This is very important, as it updated your future behavior, and decision making. So be very careful and thoughtful. 
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
            current_memory=agent.context_memory,
            user_feedback=user_feedback,
        )

        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.reflection_model,
            response_model=BaseShortTermMemoryUpdate,
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

        response = call()

        return response
    except Exception as e:
        logger.error(f"Error generating updated short-term memory: {e}")
        raise
