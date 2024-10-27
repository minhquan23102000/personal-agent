from pydantic import BaseModel, Field
from mirascope.core import prompt_template
from typing import List
from src.agent.base_agent import BaseAgent
from loguru import logger


class BaseConversationSummary(BaseModel):
    """Model for conversation summary response."""

    summary: str = Field(
        description="Concise summary of the conversation, keep most important details."
    )
    key_points: List[str] = Field(
        description="List of main points from the conversation"
    )
    outcomes: List[str] = Field(description="List of decisions or outcomes reached")


@prompt_template(
    """
    Please provide a structured summary of the following conversation.
    Focus on extracting key points, decisions made, and outcomes reached.
    Keep the summary compressed as much as possible.
    
    Conversation History:
    {history}
    
    Provide your response in a structured format with:
    1. A concise overall summary
    2. Key discussion points
    3. Specific outcomes or decisions reached
    """
)
def base_conversation_summary_prompt(history): ...


async def generate_conversation_summary(agent: BaseAgent) -> BaseConversationSummary:
    """Generate a structured summary of the current conversation."""
    prompt = base_conversation_summary_prompt(history=agent.history)
    response = await agent._custom_llm_call(
        query=prompt,
        response_model=BaseConversationSummary,
        json_mode=True,
    )
    return response
