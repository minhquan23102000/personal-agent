from dataclasses import dataclass
from typing import TYPE_CHECKING, List
from mirascope.core import (
    BaseMessageParam,
    Messages,
    prompt_template,
    litellm,
)
from loguru import logger

from src.core.reasoning.base import BaseReasoningEngine
from src.core.prompt.tool_prompt import get_list_tools_name

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


@dataclass
class ChatEngine(BaseReasoningEngine):
    """Simple chat engine implementation with tool usage capability."""

    name: str = "Chat"
    description: str = (
        "Simple chat engine for direct conversation with tool usage capability"
    )
    state_prompt: str = """
    This agent uses a simple chat framework:
    1. Process user input
    2. Respond directly or use tools when needed
    3. Provide clear and concise responses
    """

    model_name: str = "gemini/gemini-1.5-flash-002"
    temperature: float = 0.7

    async def run(self, agent: "BaseAgent") -> None:
        """Execute the chat engine's main loop."""

        # Check if tools were used in the response
        use_tool_call, _ = await agent._default_step(include_tools=True)

        # If tools were used, process the tool output
        if use_tool_call:
            use_tool_call, _ = await agent._default_step(include_tools=False)
