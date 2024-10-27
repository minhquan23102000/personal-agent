from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import inspect
import uuid
from typing import Any, Callable, Dict, List, Optional, Type, Union

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


def generate_agent_id() -> str:
    random_id = str(uuid.uuid4())
    return hashlib.sha256(random_id.encode()).hexdigest()[:8]


def interact_with_human(message: str) -> str:
    """Facilitate clear and relevant communication with a human when needed, by identifying the context—whether it's a question, chat, report, message or etc. And ensuring the interaction is focused on the user's intent."""
    print(f"[Agent Message]: {message}")
    answer = input("[Human Response]: ")
    print("[End Interaction]")

    if answer.lower() in ["exit", "quit"]:
        raise SystemExit

    return answer


@dataclass
class BaseAgent:
    """Base class for all agents with memory integration."""

    model_name: str = "gemini/gemini-1.5-flash-002"
    history: List[BaseMessageParam] = field(default_factory=list)
    max_history: Optional[int] = None
    system_prompt: str = "You are an AI agent."
    temperature: float = 0.5
    memory_manager: Optional[MemoryManager] = None
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = field(default_factory=generate_agent_id)
    short_term_memory: Optional[ShortTermMemory] = None
    tools: List[Union[Type[BaseTool], Callable]] = field(default_factory=list)

    def __post_init__(self):
        if self.memory_manager:
            dynamic_toolkit = get_memory_toolkit(self.memory_manager)
            self.add_tools(dynamic_toolkit.create_tools())
            self.memory_manager.set_agent(self)

    def add_tools(self, tools: List[Union[Type[BaseTool], Callable]]) -> None:
        self.tools.extend(tools)

    def get_tools(self) -> List[Union[Type[BaseTool], Callable]]:
        """Get the list of tools available to this agent."""
        return self.tools

    def build_prompt(
        self,
        query: Union[str, List[BaseMessageParam], Messages.Type],
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
            return [*messages, *query]

    async def _update_history(
        self, query: str | BaseMessageParam, response: OpenAICallResponse
    ) -> None:
        """Update conversation history with new messages and store in memory."""
        try:
            if isinstance(query, str):
                user_message = Messages.User(query)
            else:
                user_message = query

            self.history.append(user_message)

            assistant_message = getattr(response, "message_param", None)
            if assistant_message:
                self.history.append(assistant_message)

            if self.max_history and len(self.history) > self.max_history:
                self.history = self.history[-self.max_history :]

            if self.memory_manager:
                await self._store_messages_in_memory(query, assistant_message)

        except Exception as e:
            logger.error(f"Error updating history: {e}")
            logger.warning("Continuing conversation without storing messages")

    async def _store_messages_in_memory(
        self,
        message: str | BaseMessageParam,
        assistant_message: Optional[BaseMessageParam],
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
                content = str(getattr(assistant_message, "content", ""))
                if content:
                    await self.memory_manager.store_conversation(
                        sender=sender,
                        message_content=content,
                        message_type=MessageType.TEXT,
                        conversation_id=self.conversation_id,
                    )

                function_call = getattr(assistant_message, "function_call", None)
                if function_call:
                    await self.memory_manager.store_conversation(
                        sender=sender,
                        message_content=f"Tool call: {function_call}",
                        message_type=MessageType.TOOL,
                        conversation_id=self.conversation_id,
                    )

    def _build_call_config(
        self,
        query: Union[str, List[BaseMessageParam], Messages.Type],
        call_params: dict = {},
        include_history: bool = True,
        include_tools: bool = True,
        custom_tools: List[Union[Type[BaseTool], Callable]] = [],
    ) -> Dict[str, Any]:
        """Build the configuration for the LLM call."""
        if call_params.get("temperature"):
            call_params["temperature"] = self.temperature

        messages = self.build_prompt(query, include_history)

        tools = self.get_tools() if include_tools else []
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
                    sender=msg.role,
                    message_content=str(msg),
                    message_type=MessageType.TOOL,
                    conversation_id=self.conversation_id,
                )

    async def _custom_llm_call(
        self,
        query: Messages.Type,
        response_model: Optional[Type[Any]] = None,
        call_params: dict = {},
        include_history: bool = False,
        include_tools: bool = False,
        custom_tools: List[Union[Type[BaseTool], Callable]] = [],
        json_mode: bool = False,
    ) -> Union[OpenAICallResponse, Any]:
        config = self._build_call_config(
            query, call_params, include_history, include_tools, custom_tools
        )

        if response_model is not None:

            # @retry(
            #     stop=stop_after_attempt(3),
            #     after=collect_errors(ValidationError),
            # )
            @litellm.call(
                model=self.model_name,
                response_model=response_model,
                json_mode=json_mode,
            )  # type: ignore
            async def lite_llm_call(*, errors: list[ValidationError] | None = None):
                if errors:
                    previous_errors = f"Previous Errors: {errors}"
                    print(previous_errors)
                    config.update(
                        {"computed_fields": {"previous_errors": previous_errors}}
                    )

                return config

            return await lite_llm_call()
        else:

            @retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=4, max=10),
            )
            @litellm.call(model=self.model_name, **config)
            async def lite_llm_call():
                return config

            return await lite_llm_call()

    async def _default_llm_call(
        self,
        query: str | BaseMessageParam | Messages.Type | List[BaseMessageParam],
        call_params: dict = {},
    ) -> OpenAICallResponse:
        """Make a call to the LLM using litellm."""
        config = self._build_call_config(query, call_params)

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=4, max=10),
        )
        @litellm.call(model=self.model_name)
        async def lite_llm_call():
            return config

        return await lite_llm_call()

    async def step(self, query: str | BaseMessageParam):
        """Execute one step of the agent's reasoning process."""
        try:
            response = await self._default_llm_call(query)
            await self._update_history(query, response)

            if tools := getattr(response, "tools", None):
                await self._process_tools(tools, response)
                return await self.step("")  # Continue the conversation

            return str(getattr(response, "content", "")) if response else response

        except Exception as e:
            logger.error(f"Error in agent step: {e}")
            return f"An error occurred: {str(e)}"

    async def run(self, as_chat: bool = True) -> None:
        """Run the agent in an interactive loop."""
        try:
            await self.initialize_conversation()

            while True:
                if as_chat:
                    query = input("User: ")
                    if query.lower() in ["exit", "quit"]:
                        break
                else:
                    query = Messages.Assistant("")

                print("Assistant: ", end="", flush=True)
                response = await self.step(query)
                print(response)

        except Exception as e:
            logger.error(f"Error in run: {e}")
            print(f"An error occurred: {str(e)}")

        if self.memory_manager:
            await self.memory_manager.reflection_conversation()

    async def initialize_conversation(self) -> None:
        """Initialize conversation by loading best prompt and short-term memory."""
        if not self.memory_manager:
            logger.warning("No memory manager available - using default initialization")
            return

        try:
            best_prompts = await self.memory_manager.get_best_performing_prompts(
                limit=1
            )
            if best_prompts:
                best_prompt = best_prompts[0]
                self.system_prompt = best_prompt.improve_prompt
                logger.info(
                    f"Loaded best system prompt with score {best_prompt.reward_score}"
                )

            self.short_term_memory = await self.memory_manager.get_short_term_memory()
            logger.info("Enhanced system prompt with short-term memory context")

        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            logger.info("Using default system prompt")

    def _enhance_prompt_with_memory(self) -> str:
        """Enhance the system prompt with context from short-term memory."""
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

        return inspect.cleandoc(
            f"""
            # AGENT ID: {agent_id}
            
            # SYSTEM INSTRUCTIONS:
            
            {self.system_prompt}
            
            # CONTEXT FROM PREVIOUS INTERACTIONS:
            
            {short_memory_prompt}
            """
        )
