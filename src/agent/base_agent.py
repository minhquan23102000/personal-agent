from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import inspect
from typing import Any, Callable, Dict, List, Optional, Type, Tuple

from loguru import logger
from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    BaseTool,
    Messages,
    prompt_template,
    litellm,
)
from mirascope.core.openai import OpenAICallResponse
import pydantic


from src.memory.memory_manager import MemoryManager
from src.memory.models import MessageType, ShortTermMemory


import uuid
import random
from rich import print


def generate_agent_id():
    random_id = str(uuid.uuid4())
    hash_id = hashlib.sha256(random_id.encode()).hexdigest()[:8]
    return hash_id


@dataclass
class BaseAgent:
    """Base class for all agents with memory integration.

    Attributes:
        history: List of message parameters representing the conversation history
        max_history: Maximum number of messages to keep in history (None for unlimited)
        system_prompt: System message to use for all interactions
        memory_manager: Manager for handling agent's memory operations
        conversation_id: ID of current conversation
        agent_id: Unique identifier for this agent
    """

    model_name: str = "gemini/gemini-1.5-flash-002"
    history: List[BaseMessageParam] = field(default_factory=list)
    max_history: Optional[int] = None
    system_prompt: str = "You are an AI agent."
    temperature: float = 0.5
    memory_manager: Optional[MemoryManager] = None
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = field(default_factory=generate_agent_id)
    short_term_memory: Optional[ShortTermMemory] = None
    tools: List[Type[BaseTool] | Callable] = field(default_factory=list)

    def __post_init__(self):
        if self.memory_manager:
            dynamic_toolkit = self.memory_manager.get_dynamic_memory_toolkit()
            self.add_tools(dynamic_toolkit.create_tools())
            self.memory_manager.set_agent(self)

    def add_tools(self, tools: List[Type[BaseTool] | Callable]) -> None:
        self.tools.extend(tools)

    def get_tools(self) -> List[Type[BaseTool] | Callable]:
        """Get the list of tools available to this agent.

        Returns:
            List of tool classes that this agent can use
        """
        return self.tools

    def build_prompt(
        self,
        query: str | List[BaseMessageParam] | Messages.Type,
        include_history: bool = True,
    ) -> Messages.Type:
        """Build the prompt for the agent using the current state.

        Args:
            query: The user's input query

        Returns:
            A Messages.Type object containing the full prompt
        """
        system_prompt = self._enhance_prompt_with_memory()

        if include_history:
            messages = [Messages.System(system_prompt), *self.history]
        else:
            messages = []

        if isinstance(query, str):
            return [
                *messages,
                Messages.User(query),
            ]
        else:
            return [
                *messages,
                *query,
            ]

    async def _update_history(self, query: str, response: OpenAICallResponse) -> None:
        """Update conversation history with new messages and store in memory.

        Args:
            query: The user's input query
            response: The agent's response
        """
        try:
            # 1. Create message objects
            user_message = Messages.User(query)
            self.history.append(user_message)

            # Get message param safely
            assistant_message = getattr(response, "message_param", None)
            if assistant_message:
                self.history.append(assistant_message)

            # 2. Trim history if needed
            if self.max_history and len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

            # 3. Store messages in memory if available
            if self.memory_manager:
                # Store user message
                await self.memory_manager.store_conversation(
                    sender="user",
                    message_content=query,
                    message_type=MessageType.TEXT,
                    conversation_id=self.conversation_id,
                )

                # Store assistant response if available
                if assistant_message:
                    # Extract content safely
                    content = str(getattr(assistant_message, "content", ""))
                    if content:
                        await self.memory_manager.store_conversation(
                            sender=self.agent_id,
                            message_content=content,
                            message_type=MessageType.TEXT,
                            conversation_id=self.conversation_id,
                        )

                # Store any function calls or tool usage
                function_call = getattr(assistant_message, "function_call", None)
                if function_call:
                    await self.memory_manager.store_conversation(
                        sender=self.agent_id,
                        message_content=f"Tool call: {function_call}",
                        message_type=MessageType.TOOL,
                        conversation_id=self.conversation_id,
                    )

        except Exception as e:
            logger.error(f"Error updating history: {e}")
            # Don't raise - allow conversation to continue even if storage fails
            logger.warning("Continuing conversation without storing messages")

    def _build_call_config(
        self,
        query: str | List[BaseMessageParam] | Messages.Type,
        call_params: dict = {},
        include_history: bool = True,
        include_tools: bool = True,
        custom_tools: List[Type[BaseTool] | Callable] = [],
    ) -> Dict[str, Any]:
        """Build the configuration for the LLM call.

        Args:
            query (str): The user's input query.
            call_params (dict, optional): Additional parameters for the LLM call. Defaults to an empty dictionary.
            include_history (bool, optional): Flag to include conversation history. Defaults to True.
            include_tools (bool, optional): Flag to include tools in the configuration. Defaults to True.
            custom_tools (List[Type[BaseTool] | Callable], optional): A list of custom tools to include. Defaults to an empty list.

        Returns:
            Dict[str, Any]: A dictionary containing the messages and tools configuration.
        """
        if call_params.get("temperature"):
            call_params["temperature"] = self.temperature

        messages = self.build_prompt(query, include_history)

        tools = []
        if include_tools:
            tools = self.get_tools()

        if custom_tools:
            tools.extend(custom_tools)

        return {
            "messages": messages,
            "tools": tools,
            "call_params": call_params,
        }

    async def _process_tools(
        self, tools: List[BaseTool], response: OpenAICallResponse
    ) -> None:
        """Process and execute tools called by the agent."""
        try:
            tools_and_outputs = []
            for tool in tools:
                logger.info(f"Calling Tool '{tool._name()}' with args {tool.args}")
                # output = (
                #     tool.call()
                #     if not inspect.iscoroutinefunction(tool.call)
                #     else await tool.call()
                # )
                output = await tool.call()
                logger.info(f"Tool output: {output}")
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

    async def _custom_llm_call(
        self,
        query: str | List[BaseMessageParam] | Messages.Type,
        response_model: pydantic.BaseModel | None = None,
        call_params: dict = {},
        include_history: bool = False,
        include_tools: bool = False,
        custom_tools: List[Type[BaseTool] | Callable] = [],
        json_mode: bool = False,
    ):

        config = self._build_call_config(
            query, call_params, include_history, include_tools, custom_tools
        )

        if response_model is not None:

            @litellm.call(
                model=self.model_name,
                response_model=response_model,
                json_mode=json_mode,
            )  # type: ignore
            async def lite_llm_call(self):
                return config

            return await lite_llm_call(self)

        else:

            @litellm.call(model=self.model_name, **config)
            async def lite_llm_call(self):
                return config

            return await lite_llm_call(self)

    async def _defalt_llm_call(
        self,
        query: str,
        call_params: dict = {},
    ) -> OpenAICallResponse:
        """Make a call to the LLM using litellm.

        Args:
            config: Either a string query or a dict containing dynamic configuration
            response_model: Optional Pydantic model for response validation
            call_params: Additional parameters for the LLM call

        Returns:
            The LLM's response
        """

        # Create the LLM call function
        config = self._build_call_config(query, call_params)

        @litellm.call(model=self.model_name)
        async def lite_llm_call(self):
            return config

        return lite_llm_call(self)

    async def step(self, query: str):
        """Execute one step of the agent's reasoning process."""
        try:
            response = await self._defalt_llm_call(query)
            await self._update_history(query, response)

            if tools := response.tools:
                await self._process_tools(tools, response)
                return await self.step("")  # Continue the conversation

            # Return string response
            return str(getattr(response, "content", "")) if response else response

        except Exception as e:
            logger.error(f"Error in agent step: {e}")
            error_msg = f"An error occurred: {str(e)}"
            return error_msg

    async def run(self) -> None:
        """Run the agent in an interactive loop."""
        try:
            # Initialize conversation
            await self.initialize_conversation()

            while True:
                query = input("User: ")
                if query.lower() in ["exit", "quit"]:
                    break

                print("Assistant: ", end="", flush=True)
                response = await self.step(query)
                print(response)

        except Exception as e:
            logger.error(f"Error in run: {e}")
            print(f"An error occurred: {str(e)}")

        # Handle conversation end
        if self.memory_manager:
            await self.memory_manager.reflection_conversation()

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
            self.short_term_memory = short_term_memory

            logger.info("Enhanced system prompt with short-term memory context")

        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            # Use default prompt if initialization fails
            logger.info("Using default system prompt")

    def _enhance_prompt_with_memory(self) -> str:
        """Enhance the system prompt with context from short-term memory.


        Returns:
            Enhanced prompt incorporating memory context
        """
        # Build enhanced prompt with memory context

        # Agent id
        agent_id = f"Your ID: {self.agent_id}\n"

        if self.short_term_memory:
            short_memory_prompt = inspect.cleandoc(
                f"""
                ## AGENT INFORMATION & IDENTITY: {self.short_term_memory.agent_info}
                ## AGENT BELIEFS: {self.short_term_memory.agent_beliefs}
                
                ## USER INFORMATION: {self.short_term_memory.user_info}
                
                ## LAST CONVERSATION SUMMARY: {self.short_term_memory.last_conversation_summary}
                ## RECENT GOALS AND STATUS: {self.short_term_memory.recent_goal_and_status}
                ## IMPORTANT CONTEXT: {self.short_term_memory.important_context}
                """
            )
        else:
            short_memory_prompt = "This is your first interaction with the user."

        enhanced_prompt = inspect.cleandoc(
            f"""
            # AGENT ID: {agent_id}
            
            # SYSTEM INSTRUCTIONS:
            
            {self.system_prompt}
            
            # CONTEXT FROM PREVIOUS INTERACTIONS:
            
            {short_memory_prompt}
            """
        )

        return enhanced_prompt
