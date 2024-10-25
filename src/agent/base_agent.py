from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from mirascope.core import BaseDynamicConfig, BaseMessageParam, BaseTool, Messages


@dataclass
class BaseAgent(ABC):
    """Base class for all agents.

    This class provides the basic structure and interface that all agents should follow.
    It includes abstract methods for defining prompts and tools, as well as basic state
    management through message history.

    Attributes:
        history: List of message parameters representing the conversation history
        max_history: Maximum number of messages to keep in history (None for unlimited)
        system_prompt: System message to use for all interactions
    """

    history: List[BaseMessageParam] = field(default_factory=list)
    max_history: Optional[int] = None
    system_prompt: str = "You are an AI assistant."

    @abstractmethod
    def get_tools(self) -> List[Type[BaseTool]]:
        """Get the list of tools available to this agent.

        Returns:
            List of tool classes that this agent can use
        """
        return []

    @abstractmethod
    def build_prompt(self, query: str) -> Messages.Type:
        """Build the prompt for the agent using the current state.

        Args:
            query: The user's input query

        Returns:
            A Messages.Type object containing the full prompt
        """
        return [
            Messages.System(self.system_prompt),
            *self.history,
            Messages.User(query),
        ]

    def _update_history(self, query: str, response: BaseDynamicConfig) -> None:
        """Update conversation history with new messages.

        Args:
            query: The user's input query
            response: The agent's response
        """
        self.history.append(Messages.User(query))
        self.history.append(response.message_param)

        if self.max_history and len(self.history) > self.max_history:
            # Keep only the most recent messages up to max_history
            self.history = self.history[-self.max_history :]

    def _build_config(self, query: str) -> Dict[str, Any]:
        """Build the configuration for the LLM call.

        Args:
            query: The user's input query

        Returns:
            Dictionary containing the messages and tools configuration
        """
        return {"messages": self.build_prompt(query), "tools": self.get_tools()}

    async def _process_tools(
        self, tools: List[BaseTool], response: BaseDynamicConfig
    ) -> None:
        """Process and execute tools called by the agent.

        Args:
            tools: List of tools to execute
            response: The agent's response containing tool calls
        """
        tools_and_outputs = []
        for tool in tools:
            print(f"[Calling Tool '{tool._name()}' with args {tool.args}]")
            output = await tool.call()
            tools_and_outputs.append((tool, output))

        self.history.extend(response.tool_message_params(tools_and_outputs))

    @abstractmethod
    async def _call(self, query: str) -> BaseDynamicConfig:
        """Make the actual call to the LLM.

        This method should be implemented by subclasses to use their specific
        LLM provider.

        Args:
            query: The user's input query

        Returns:
            The LLM's response
        """
        pass

    async def step(self, query: str) -> str:
        """Execute one step of the agent's reasoning process.

        Args:
            query: The user's input query

        Returns:
            The agent's final response content
        """
        response = await self._call(query)
        self._update_history(query, response)

        if tools := response.tools:
            await self._process_tools(tools, response)
            return await self.step("")  # Continue the conversation

        return response

    async def run(self) -> None:
        """Run the agent in an interactive loop."""
        while True:
            query = input("User: ")
            if query.lower() in ["exit", "quit"]:
                break

            print("Assistant: ", end="", flush=True)
            response = await self.step(query)
            print(response)
