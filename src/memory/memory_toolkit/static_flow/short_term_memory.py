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

    user_info: str = Field(description="Updated user information")
    recent_goal_and_status: str = Field(description="Current goals and their status")
    important_context: str = Field(description="Key contextual information")
    agent_beliefs: str = Field(description="Updated agent beliefs")
    agent_info: str = Field(description="Agent's current information and identity")


def short_term_memory_prompt(history, summary, current_memory):
    return [
        Messages.User(
            inspect.cleandoc(
                f"""
        Analyze the conversation history to extract and update the agent's short-term memory with the following key information:

        1. **User Information**: Identify any new details about the user's preferences, background, or needs that emerged during the conversation.
        2. **Recent Goals**: Document the current goals the user has set and their progress or status.
        3. **Important Context**: Capture any significant contextual elements that should be retained for future reference.
        4. **Agent Beliefs**: Adjust the agent's understanding of the user's intentions and the world based on insights gained from the conversation.
        5. **Agent Info**: Confirm and update details about the agent, including its personality, role, name, gender, language, style, age, etc.

        Ensure the updates focus on the most recent and relevant information, provide a clear status of ongoing tasks, and maintain important context for future interactions.
        
        Current Short-Term Memory:
        {current_memory}
        
        Conversation History:
        {history}
        
        Conversation Summary:
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
