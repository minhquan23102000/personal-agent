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


class BaseSelfReflection(BaseModel):
    """Model for self-reflection response."""

    strengths: List[str] = Field(description="What worked well in the conversation")
    areas_for_improvement: List[str] = Field(description="What could be improved")
    specific_examples: List[str] = Field(
        description="Concrete examples from the conversation"
    )
    reward_score: float = Field(
        description="Performance score from 0-10",
        ge=0,
        le=10,
    )
    improved_prompt: str = Field(
        description="Enhanced system prompt addressing identified issues"
    )


def base_self_reflection_prompt(system_prompt, history):
    return [
        Messages.System(
            inspect.cleandoc(
                f"""
                
                ### Instructions:
                Analyze the provided conversation by evaluating its performance on a scale of 0-10, focusing on task completion, user satisfaction, information accuracy, and conversation flow, human-like conversation, and overall helpfulness. Identify strengths and successes, as well as areas for improvement. Provide specific examples from the conversation to illustrate your points. Finally, suggest an improved system prompt that addresses any identified issues.
                
                * Focus on providing actionable and constructive feedback.
                * The revised system prompt must be directly relevant to the identified areas for improvement
                * Analytical, objective, detailed, and specific. Use clear and concise language, avoiding jargon.
                * Evaluation should be based on observable evidence from the conversation.
                
                ### Supplementary Information:
                * **First Principles Thinking:**  Break down the conversation into its fundamental components (user intent, LLM response, information exchange) to identify core issues.
                * **Occam's Razor:** Favor simpler explanations and solutions when addressing areas for improvement.
                * **User-Centered Design:** Prioritize the user's needs and expectations throughout the analysis and prompt revision.
                * **Cognitive Biases:** Be mindful of potential biases in the conversation and strive for an objective evaluation.  Consider how biases might affect both the user and the LLM.
                * **User Experience Design:** Consider using principles from user experience design, communication theory, and AI interaction standards to enhance evaluation accuracy and depth.
               
                
                Current System Prompt: {system_prompt}
                Conversation Logs: {history}
                """
            )
        )
    ]


async def perform_self_reflection(agent: "BaseAgent") -> BaseSelfReflection:
    """Perform structured self-reflection and generate improvements."""
    try:

        @retry(
            stop=stop_after_attempt(3),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.model_name, response_model=BaseSelfReflection, json_mode=True
        )
        def call(*, errors: list[ValidationError] | None = None):
            config = {}

            config["messages"] = base_self_reflection_prompt(
                system_prompt=agent.system_prompt, history=agent.history
            )

            if errors:
                config["computed_fields"] = {
                    "previous_errors": f"Previous Errors: {errors}"
                }

            return config

        response = call()

        return response
    except Exception as e:
        logger.error(f"Error performing self-reflection: {e}")
        raise
