import inspect
from pydantic import BaseModel, Field
from mirascope.core import prompt_template, Messages, litellm
from typing import List, TYPE_CHECKING
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import AfterValidator, ValidationError

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


class BaseConversationSummary(BaseModel):

    summary: str = Field(
        description="Concise summary of the conversation, keep most important details."
    )
    key_points: List[str] = Field(
        description="List of main points from the conversation"
    )
    outcomes: List[str] = Field(description="List of decisions or outcomes reached")


def base_conversation_summary_prompt(history):
    return [
        Messages.System(
            inspect.cleandoc(
                f"""
            Create a structured summary of the conversation below. The information should be compressed as much as possible while still retaining the most important details.
            
            {history}
            """
            )
        )
    ]


async def generate_conversation_summary(agent: "BaseAgent") -> BaseConversationSummary:
    """Generate a structured summary of the current conversation."""
    prompt = base_conversation_summary_prompt(history=agent.history)

    # response = await agent._custom_llm_call(
    #     query=prompt,
    #     response_model=BaseConversationSummary,
    #     json_mode=True,
    # )

    @retry(
        stop=stop_after_attempt(3),
        after=collect_errors(ValidationError),
    )
    @litellm.call(
        model=agent.model_name, response_model=BaseConversationSummary, json_mode=True
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
