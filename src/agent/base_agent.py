from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union
import traceback
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
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import AfterValidator, ValidationError

from src.memory.memory_manager import MemoryManager
from src.memory.models import MessageType, ShortTermMemory
from src.memory.memory_toolkit.dynamic_flow import get_memory_toolkit
from src.util.rotating_list import RotatingList
import os


def generate_agent_id() -> str:
    random_id = str(uuid.uuid4())
    return hashlib.sha256(random_id.encode()).hexdigest()[:8]


async def send_message_to_human(message: str) -> str:
    """A tool help you communication with human your creator."""
    print(f"[Agent Message]: {message}")
    answer = input("[Human Response]: ")
    print("[End Interaction]")

    if answer.lower() in ["exit", "quit"]:
        raise SystemExit

    return f"[Human Response]: {answer}"


@dataclass
class BaseAgent:
    """Base class for all agents with memory integration."""

    model_name: str = "gemini/gemini-1.5-flash-002"
    history: List[Messages.Type] = field(default_factory=list)
    max_history: Optional[int] = None
    system_prompt: str = "You are an AI agent."
    temperature: float = 0.5
    memory_manager: Optional[MemoryManager] = None
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = field(default_factory=generate_agent_id)
    short_term_memory: Optional[ShortTermMemory] = None
    tools: List[Union[Type[BaseTool], Callable]] = field(default_factory=list)
    api_keys: list[str] | None = None
    rotating_api_keys: RotatingList | None = None
    api_key_env_var: str | None = None

    def __post_init__(self):
        if self.memory_manager:
            dynamic_toolkit = get_memory_toolkit(self.memory_manager)
            self.add_tools(dynamic_toolkit.create_tools())
            self.memory_manager.set_agent(self)

        if self.api_keys:
            logger.info(f"Rotating through {len(self.api_keys)} api keys")
            logger.info(f"Api env var: {self.api_key_env_var}")
            if not self.api_key_env_var:
                raise ValueError("Api key env var is not set")

            self.rotating_api_keys = RotatingList(self.api_keys)

    def add_tools(self, tools: List[Union[Type[BaseTool], Callable]]) -> None:
        self.tools.extend(tools)

    def get_tools(self) -> List[Union[Type[BaseTool], Callable]]:
        """Get the list of tools available to this agent."""
        return self.tools

    def build_prompt(
        self,
        query: Messages.Type,
        include_history: bool = True,
    ) -> Messages.Type:
        """Build the prompt for the agent using the current state."""
        system_prompt = self._enhance_prompt_with_memory()

        messages = (
            [Messages.System(system_prompt), *self.history] if include_history else []
        )

        if isinstance(query, str):
            return [*messages, Messages.User(query)]
        else:
            if isinstance(query, list):
                return [*messages, *query]
            else:
                return [*messages, query]

    async def _update_history(
        self, query: Messages.Type, response: OpenAICallResponse
    ) -> None:
        """Update conversation history with new messages and store in memory."""
        try:
            if isinstance(query, str):
                query_message = Messages.User(query)
            else:
                query_message = query

            self.history.append(query_message)
            response_message = None
            try:
                response_message = response.message_param
            except Exception as e:
                logger.error(f"Error getting response message: {e}.")

            if response_message:
                self.history.append(response_message)

            if self.max_history and len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

            if self.memory_manager:
                await self._store_messages_in_memory(query, response_message)

        except Exception as e:
            logger.error(f"Error updating history: {e}")
            logger.warning("Continuing conversation without storing messages")

    async def _store_messages_in_memory(
        self,
        message: Messages.Type,
        assistant_message: Optional[BaseMessageParam] | dict,
    ) -> None:
        """Store messages in memory if available."""
        if self.memory_manager:
            if isinstance(message, str):
                sender = "user"
            else:
                sender = message.role
                message = str(message.content)

            await self.memory_manager.store_conversation(
                sender=sender,
                message_content=message,
                message_type=MessageType.TEXT,
                conversation_id=self.conversation_id,
            )

            if assistant_message:
                try:
                    if isinstance(assistant_message, dict):
                        assistant_message = BaseMessageParam(
                            role=assistant_message["role"],
                            content=str(assistant_message["tool_calls"]),
                        )

                    await self.memory_manager.store_conversation(
                        sender=assistant_message.role,
                        message_content=str(assistant_message.content),
                        message_type=MessageType.TEXT,
                        conversation_id=self.conversation_id,
                    )
                except Exception as e:
                    logger.error(
                        f"Error storing assistant message in memory: {e}. Message: {assistant_message}"
                    )

    def _build_call_config(
        self,
        query: Messages.Type,
        call_params: dict = {},
        include_history: bool = True,
        include_tools: bool = True,
        custom_tools: List[Union[Type[BaseTool], Callable]] = [],
        errors: list[ValidationError] | None = None,
    ) -> Dict[str, Any]:
        """Build the configuration for the LLM call."""
        if call_params.get("temperature"):
            call_params["temperature"] = self.temperature

        messages = self.build_prompt(query, include_history)

        tools = self.get_tools() if include_tools else []
        tools.extend(custom_tools)

        config = {
            "messages": messages,
            "tools": tools,
            "call_params": call_params,
        }

        if errors:
            config["computed_fields"] = {
                "previous_errors": f"Previous Errors: {errors}"
            }

        return config

    async def _process_tools(
        self, tools: List[BaseTool], response: OpenAICallResponse
    ) -> None:
        """Process and execute tools called by the agent."""
        try:
            tools_and_outputs = []
            for tool in tools:
                logger.info(f"Calling Tool '{tool._name()}' with args {tool.args}")
                output = await tool.call()
                logger.info(f"Tool output: {output}")
                tools_and_outputs.append((tool, output))

            tool_messages = getattr(response, "tool_message_params", None)
            if tool_messages and callable(tool_messages):
                messages = tool_messages(tools_and_outputs)
                self.history.extend(messages)

                if self.memory_manager:
                    await self._store_tool_interactions(messages)
        except Exception as e:
            logger.error(f"Error processing tools: {e}")
            raise

    async def _store_tool_interactions(self, messages: List[BaseMessageParam]) -> None:
        """Store tool interactions in memory."""
        if self.memory_manager:
            for msg in messages:
                await self.memory_manager.store_conversation(
                    sender="Tool",
                    message_content=str(msg),
                    message_type=MessageType.TOOL,
                    conversation_id=self.conversation_id,
                )

    async def _default_llm_call(
        self,
        query: Messages.Type,
        call_params: dict = {},
        *,
        errors: list[ValidationError] | None = None,
    ) -> OpenAICallResponse:
        """Make a call to the LLM using litellm."""
        config = self._build_call_config(query, call_params, errors=errors)

        # @retry(
        #     stop=stop_after_attempt(3),
        #     wait=wait_exponential(multiplier=1, min=4, max=10),
        # )
        @litellm.call(model=self.model_name)
        async def lite_llm_call():
            # Rotate api key after each call
            if self.rotating_api_keys:
                if self.api_key_env_var:
                    api_key = self.rotating_api_keys.rotate()
                    os.environ[self.api_key_env_var] = api_key

            return config

        return await lite_llm_call()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        after=collect_errors(ValidationError),
    )
    async def step(
        self, query: Messages.Type, *, errors: list[ValidationError] | None = None
    ):
        """Execute one step of the agent's reasoning process."""
        try:
            response = await self._default_llm_call(query, errors=errors)

            if tools := getattr(response, "tools", None):
                await self._process_tools(tools, response)
                return await self.step("")  # Continue the conversation

            await self._update_history(query, response)

            return str(getattr(response, "content", "")) if response else response

        except Exception as e:
            config = self._build_call_config(query)
            logger.error(
                f"Error in agent step: {e}. Traceback: {traceback.format_exc()}."
            )
            logger.error(f"Formatted Config: {json.dumps(config, indent=4)}")

            raise e

    async def run(self, as_chat: bool = True) -> None:
        """Run the agent in an interactive loop."""
        try:
            await self.initialize_conversation()
            if not as_chat:
                self.add_tools([send_message_to_human])

            first_time = True

            while True:
                if as_chat:
                    query = input("User: ")
                    if query.lower() in ["exit", "quit"]:
                        break
                else:
                    if first_time:
                        query = (
                            "Note: this is an auto message, send by an system operator. "
                            " You are currently operating independently, evaluate the situation and determine the optimal next steps."
                            " Call 'send_message_to_human' tool to communicate with the human when necessary."
                        )
                        first_time = False

                    else:
                        query = ""

                    time.sleep(1)

                print("Assistant: ", end="", flush=True)
                response = await self.step(query)
                print(response)

        except Exception as e:
            logger.error(f"Error in run: {e}")
            raise e
        finally:
            if self.memory_manager:
                if len(self.history) >= 4:
                    await self.memory_manager.reflection_conversation()

    async def initialize_conversation(self) -> None:
        """Initialize conversation by loading best prompt and short-term memory."""
        logger.info("Initializing conversation...")

        if not self.memory_manager:
            logger.warning("No memory manager available - using default initialization")
            return

        try:
            latest_summary = await self.memory_manager.get_latest_conversation_summary()
            if latest_summary:
                self.system_prompt = latest_summary.improve_prompt
                logger.info(f"Loaded system prompt: {self.system_prompt}")

            self.short_term_memory = await self.memory_manager.get_short_term_memory()
            logger.info(
                f"Enhanced system prompt with short-term memory context: {self.short_term_memory}"
            )

            logger.debug(
                f"Enhanced system prompt with short-term memory context: {self._enhance_prompt_with_memory()}"
            )

        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            logger.info("Using default system prompt")

    def _enhance_prompt_with_memory(self) -> str:
        """Enhance the system prompt with context from short-term memory."""
        agent_id = f"Your ID: {self.agent_id}\n"

        if self.short_term_memory:
            # logger.debug("Populate short-term memory into system prompt")
            short_memory_prompt = f"""
                ## AGENT INFORMATION & IDENTITY: 
                {self.short_term_memory.agent_info}
                
                ## AGENT BELIEFS: 
                {self.short_term_memory.agent_beliefs}
                
                ## USER INFORMATION: 
                {self.short_term_memory.user_info}
                
                ## LAST CONVERSATION SUMMARY: 
                {self.short_term_memory.last_conversation_summary}
                
                ## RECENT GOALS AND STATUS: 
                {self.short_term_memory.recent_goal_and_status}
                
                ## IMPORTANT CONTEXT: 
                {self.short_term_memory.important_context}
                """

        else:
            short_memory_prompt = "This is your first interaction with the user."

        return inspect.cleandoc(
            f"""
            # AGENT ID: {agent_id}
            
            # SYSTEM INSTRUCTIONS:
            
            {self.system_prompt}
            
            {short_memory_prompt}
            """
        )
