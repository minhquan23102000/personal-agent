import inspect
from logging import config
from pyexpat import errors
from typing import List, TYPE_CHECKING
from jsonschema import ValidationError
from mirascope.core import (
    prompt_template,
    BaseMessageParam,
    litellm,
    BaseDynamicConfig,
    Messages,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import AfterValidator, ValidationError
from pydantic import BaseModel, Field

from loguru import logger
from tenacity import retry, stop_after_attempt

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent

from src.agent.error_prompt import format_error_message


class BaseSelfReflection(BaseModel):
    """Model for self-reflection response."""

    thoughts: str = Field(
        description="Critique and thoughts about the conversation and your performance"
    )
    feelings: str = Field(
        description="Feelings about the conversation and your performance. Describe it in short keywords format with uppercase."
    )
    reward_score: float = Field(
        description="Performance score from 0-10",
        ge=0,
        le=10,
    )
    strengths: List[str] = Field(description="What worked well in the conversation")
    areas_for_improvement: List[str] = Field(description="What could be improved")
    specific_examples: List[str] = Field(
        description="Concrete examples from the conversation"
    )

    improved_prompt: str = Field(
        description="Revise the system prompt to resolve identified issues. You can add, update, or remove parts of the system prompt. This is very important, as it updated your future behavior, personality, and decision making. So be very careful and thoughtful."
    )


@prompt_template(
    """
    MESSAGES: {history}
    
    USER:
    Analyze the provided conversation history and your system prompt. Evaluating your performance in the conversation on a scale of 0-10. Identify strengths and successes, as well as areas for improvement. Specific examples from the conversation to illustrate your points. Finally, suggest an improved system prompt that addresses any identified issues. This is very important, as it updated your future behavior, personality, and decision making. So be very careful and thoughtful.
    
    USER FEEDBACK:
    {user_feedback}
    
    AGENT (YOUR) SYSTEM PROMPT:
    {system_prompt}
    """
)
def base_self_reflection_prompt(system_prompt, history, user_feedback): ...


async def perform_self_reflection(
    agent: "BaseAgent", user_feedback: str = ""
) -> BaseSelfReflection:
    """Perform structured self-reflection and generate improvements."""
    try:
        user_feedback = user_feedback or "No user feedback provided."

        @retry(
            stop=stop_after_attempt(3),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.slow_model_name,
            response_model=BaseSelfReflection,
            json_mode=True,
        )
        def call(*, errors: list[ValidationError] | None = None):
            config = {}

            config["messages"] = base_self_reflection_prompt(
                system_prompt=agent.system_prompt,
                history=agent._build_prompt(include_system_prompt=True),
                user_feedback=user_feedback,
            )

            if errors:
                config["computed_fields"] = {
                    "previous_errors": f"Previous Errors: {format_error_message(errors)}"
                }

            return config

        response = call()

        return response
    except Exception as e:
        logger.error(f"Error performing self-reflection: {e}")
        raise
