from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, List, Optional, Type, Tuple

from loguru import logger
from mirascope.core import BaseDynamicConfig, BaseMessageParam, BaseTool, Messages

from src.memory.memory_manager import MemoryManager
from src.memory.models import MessageType, ShortTermMemory
import uuid
import random
from rich import print


@dataclass
class BaseAgent(ABC):
    """Base class for all agents with memory integration.

    Attributes:
        history: List of message parameters representing the conversation history
        max_history: Maximum number of messages to keep in history (None for unlimited)
        system_prompt: System message to use for all interactions
        memory_manager: Manager for handling agent's memory operations
        conversation_id: ID of current conversation
        agent_id: Unique identifier for this agent
    """

    history: List[BaseMessageParam] = field(default_factory=list)
    max_history: Optional[int] = None
    system_prompt: str = "You are an AI assistant."
    memory_manager: Optional[MemoryManager] = None
    conversation_id: Optional[int] = None
    agent_id: str = field(
        default_factory=lambda: f"Agent ID: {random.randint(1000, 9999)}"
    )

    def __post_init__(self):
        pass

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

    async def _update_history(self, query: str, response: BaseDynamicConfig) -> None:
        """Update conversation history with new messages."""
        try:
            # Add messages to history
            user_message = Messages.User(query)
            self.history.append(user_message)

            # Get message param safely
            message_param = getattr(response, "message_param", None)
            if message_param:
                self.history.append(message_param)

            # Trim history if needed
            if self.max_history and len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

            # Store in memory if available
            if self.memory_manager:
                await self.memory_manager.store_conversation(
                    sender="user",
                    message_content=query,
                    message_type=MessageType.TEXT,
                    conversation_id=self.conversation_id,
                )
                if message_param:
                    await self.memory_manager.store_conversation(
                        sender=self.agent_id,
                        message_content=str(message_param),
                        message_type=MessageType.TEXT,
                        conversation_id=self.conversation_id,
                    )
        except Exception as e:
            logger.error(f"Error updating history: {e}")
            raise

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
        """Process and execute tools called by the agent."""
        try:
            tools_and_outputs = []
            for tool in tools:
                logger.info(f"Calling Tool '{tool._name()}' with args {tool.args}")
                output = await tool.call()
                tools_and_outputs.append((tool, output))

            # Get tool messages safely
            tool_messages = getattr(response, "tool_message_params", None)
            if tool_messages and callable(tool_messages):
                messages = tool_messages(tools_and_outputs)
                self.history.extend(messages)

                # Store tool interactions in memory
                if self.memory_manager:
                    for msg in messages:
                        await self.memory_manager.store_conversation(
                            sender=self.agent_id,
                            message_content=str(msg),
                            message_type=MessageType.TOOL,
                            conversation_id=self.conversation_id,
                        )
        except Exception as e:
            logger.error(f"Error processing tools: {e}")
            raise

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
        """Execute one step of the agent's reasoning process."""
        try:
            response = await self._call(query)
            await self._update_history(query, response)

            # Get tools safely using dict access
            tools = response.get("tools", []) if isinstance(response, dict) else []

            if tools:
                await self._process_tools(tools, response)
                return await self.step("")  # Continue the conversation

            # Return string response
            return str(getattr(response, "content", "")) if response else ""

        except Exception as e:
            logger.error(f"Error in agent step: {e}")
            error_msg = f"An error occurred: {str(e)}"
            return error_msg

    async def run(self) -> None:
        """Run the agent in an interactive loop."""
        # Initialize conversation before starting
        await self.initialize_conversation()

        while True:
            query = input("User: ")
            if query.lower() in ["exit", "quit"]:
                break

            print("Assistant: ", end="", flush=True)
            response = await self.step(query)
            print(response)

    @abstractmethod
    async def generate_conversation_summary(self) -> str:
        """Generate summary of the current conversation."""
        pass

    @abstractmethod
    async def perform_self_reflection(self) -> Tuple[float, str, str]:
        """Perform self-reflection and return reward score, feedback and improved prompt."""
        pass

    @abstractmethod
    async def update_short_term_memory(self) -> None:
        """Update agent's short-term memory state."""
        pass

    async def initialize_conversation(self) -> None:
        """Initialize conversation by loading best prompt and short-term memory."""
        if not self.memory_manager:
            logger.warning("No memory manager available - using default initialization")
            return

        try:
            # 1. Load best performing system prompt
            best_prompts = await self.memory_manager.get_best_performing_prompts(
                limit=1
            )
            if best_prompts:
                best_prompt = best_prompts[0]
                self.system_prompt = best_prompt.improve_prompt
                logger.info(
                    f"Loaded best system prompt with score {best_prompt.reward_score}"
                )

            # 2. Load short-term memory
            short_term_memory = await self.memory_manager.get_short_term_memory()
            if short_term_memory:
                # Update system prompt with context from short-term memory
                self.system_prompt = self._enhance_prompt_with_memory(
                    self.system_prompt, short_term_memory
                )
                logger.info("Enhanced system prompt with short-term memory context")

        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            # Use default prompt if initialization fails
            logger.info("Using default system prompt")

    def _enhance_prompt_with_memory(
        self, base_prompt: str, memory: ShortTermMemory
    ) -> str:
        """Enhance the system prompt with context from short-term memory.

        Args:
            base_prompt: Original system prompt
            memory: Current short-term memory state

        Returns:
            Enhanced prompt incorporating memory context
        """
        # Build enhanced prompt with memory context

        short_memory_prompt = inspect.cleandoc(
            f"""
            User Information: \n{memory.user_info}\n
            Recent Goals and Status: \n{memory.recent_goal_and_status}\n
            Important Context: \n{memory.important_context}\n
            Current Beliefs: \n{memory.agent_beliefs}\n
            Last Conversation Summary: \n{memory.last_conversation_summary}\n
            """
        )

        enhanced_prompt = [
            "# System Prompt\n",
            base_prompt,
            "",
            "# Context from previous interactions:\n",
            short_memory_prompt,
        ]

        return "\n".join(enhanced_prompt)
