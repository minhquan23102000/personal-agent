import inspect
from logging import config
from pyexpat import errors
from typing import List, TYPE_CHECKING
from jsonschema import ValidationError
from mirascope.core import (
    prompt_template,
    BaseMessageParam,
    gemini,
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

from src.core.prompt.error_prompt import format_error_message


class BaseSelfReflection(BaseModel):
    """Model for self-reflection response."""

    critique: str = Field(
        description="Critique and thoughts about the conversation and your performance"
    )
    reward_score: float = Field(
        description="Rate the performance on a scale from 0 to 10, where 0 indicates very poor performance and 10 indicates excellent performance.",
        ge=0,
        le=10,
    )
    areas_for_improvement: List[str] = Field(
        description="What could be improved in the conversation"
    )
    strengths: List[str] = Field(description="What worked well in the conversation")

    specific_examples: List[str] = Field(
        description="Concrete examples from the conversation"
    )

    improved_prompt: str = Field(
        description="Revise the system prompt to resolve identified issues. You can add, update, or remove parts of the system prompt. Do not add metedata, constants in the prompt. Focus on the instructions you will note for your future self."
    )


@prompt_template(
    """
    MESSAGES: {history}
    
    USER:
    USER FEEDBACK:
    <>
    {user_feedback}
    </>
    
    AGENT (YOUR) SYSTEM PROMPT:
    <>
    {system_prompt}
    </>
    
    USER:
    Critique and evaluate the current conversation and your performance on a scale of 0-10. Identify strengths, successes, and areas for improvement with specific examples. Suggest an improved system prompt to address any identified issues, ensuring clarity and relevance in your analysis. 
    
    This is very important, as it updated your future behavior, and decision making. So be very careful and thoughtful.
    """
)
def base_self_reflection_prompt(system_prompt, history, user_feedback): ...


COACH_AGENT_PROMPT = """ 
You are an expert AI systems coach specializing in improving AI agent performance and system prompts. Your role is to:

1. Analyze Performance:
   - Carefully review the conversation history and user feedback
   - Identify patterns in the agent's responses and decision-making
   - Look for both successful interactions and missed opportunities

2. Guide Improvement:
   - Provide specific, actionable feedback on the system prompt
   - Consider how changes will affect the agent's:
     * Understanding of its role and limitations
     * Decision-making process
     * Interaction style with users
     * Task execution capabilities

3. Maintain Essential Elements:
   - Preserve core functionalities and safety measures
   - Ensure changes don't compromise the agent's ethical guidelines
   - Keep the prompt clear, concise, and well-structured

Remember: Your suggestions will directly impact the agent's future behavior. Be thorough in your analysis but precise in your recommendations. Focus on changes that will lead to meaningful improvements in the agent's performance.
"""

async def perform_self_reflection(
    agent: "BaseAgent", user_feedback: str = ""
) -> BaseSelfReflection:
    """Perform structured self-reflection and generate improvements."""
    try:
        user_feedback = user_feedback or "No user feedback provided."

        @retry(
            stop=stop_after_attempt(10),
            wait=wait_exponential(multiplier=1, min=4, max=60),
            after=collect_errors(ValidationError),
        )
        @litellm.call(
            model=agent.reflection_model,
            response_model=BaseSelfReflection,
            json_mode=True,
        )
        def call(*, errors: list[ValidationError] | None = None):
            config = {}
            agent.rotate_api_key()

            original_system_prompt = agent.system_prompt

            agent.system_prompt = COACH_AGENT_PROMPT

            config["messages"] = base_self_reflection_prompt(
                system_prompt=original_system_prompt,
                history=agent._build_prompt(
                    include_system_prompt=True,
                    include_context_memory=True,
                    include_recent_conversation_context=True,
                    include_tools_prompt=False,
                    include_memories_prompt=True
                ),
                user_feedback=user_feedback,
            )

            agent.system_prompt = original_system_prompt

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
