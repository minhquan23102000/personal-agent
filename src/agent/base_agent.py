from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import datetime
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
    openai,
)

from mirascope.core.openai import OpenAICallResponse, OpenAITool
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import AfterValidator, ValidationError
import pydantic
from src.memory.memory_manager import MemoryManager
from src.memory.models import MessageType, ShortTermMemory
from src.memory.memory_toolkit.dynamic_flow import get_memory_toolkit
from src.util.rotating_list import RotatingList
import os
from rich import print
import rich
from src.agent.tools.prompt import build_prompt_from_list_tools, get_list_tools_name


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

    default_model_name: str = "gemini/gemini-1.5-flash-002"
    slow_model_name: str = "gemini/gemini-1.5-flash-002"
    history: List[BaseMessageParam | Messages.Type | OpenAICallResponse] = field(
        default_factory=list
    )
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

        logger.info(f"Agent ID: {self.agent_id}")
        logger.info(f"Action Model: {self.default_model_name}")
        logger.info(f"Reasoning Model: {self.slow_model_name}")
        logger.info(f"Temperature: {self.temperature}")

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

    def rotate_api_key(self) -> None:
        """Rotate the api key."""
        if self.rotating_api_keys:
            if self.api_key_env_var:
                api_key = self.rotating_api_keys.rotate()
                os.environ[self.api_key_env_var] = api_key

    def add_tools(self, tools: List[Union[Type[BaseTool], Callable]]) -> None:
        self.tools.extend(tools)

    def get_tools(self) -> List[Union[Type[BaseTool], Callable]]:
        """Get the list of tools available to this agent."""
        return self.tools

    def _build_system_prompt(self) -> str:

        return inspect.cleandoc(
            f"""
            # AGENT ID: {self.agent_id}

            # CURRENT TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

            # SYSTEM INSTRUCTIONS:

            {self.system_prompt}
            """
        ).strip()

    def _build_short_term_memory_prompt(self) -> str:
        """Enhance the system prompt with context from short-term memory."""

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
            Conversation ended at {self.short_term_memory.timestamp.strftime("%Y-%m-%d %H:%M:%S")}
            {self.short_term_memory.last_conversation_summary}

            ## RECENT GOALS AND STATUS: 
            {self.short_term_memory.recent_goal_and_status}

            ## IMPORTANT CONTEXT: 
            {self.short_term_memory.important_context}

            ## ENVIRONMENT INFORMATION: 
            {self.short_term_memory.environment_info}
            """

        else:
            short_memory_prompt = "This is your first interaction with the user."

        return inspect.cleandoc(short_memory_prompt).strip()

    @prompt_template(
        """
        SYSTEM: 
        {system_prompt}
        {short_term_memory_prompt}
        
        ## LIST OF TOOLS YOU CAN USE: 
        {tools_prompt}
        
        MESSAGES: {history}
        """
    )
    def _build_prompt(
        self,
        include_history: bool = True,
        include_short_term_memory: bool = True,
        include_system_prompt: bool = True,
        include_tools_prompt: bool = True,
    ) -> BaseDynamicConfig:
        """Build the prompt for the agent using the current state."""
        system_prompt = self._build_system_prompt() if include_system_prompt else ""
        short_term_memory_prompt = (
            self._build_short_term_memory_prompt() if include_short_term_memory else ""
        )
        history = self.history if include_history else []
        tools_prompt = (
            build_prompt_from_list_tools(self.get_tools())
            if include_tools_prompt and self.get_tools()
            else ""
        )
        return {
            "computed_fields": {
                "system_prompt": system_prompt,
                "short_term_memory_prompt": short_term_memory_prompt,
                "history": history,
                "tools_prompt": tools_prompt,
            }
        }

    class ReasoningAction(pydantic.BaseModel):
        """Thought and action reasoning."""

        thought: str = pydantic.Field(
            description="Your thought on the current situation, obervation."
        )
        send_message_to_user: bool = pydantic.Field(
            description="Return True if this action is send a message to the user and end your turn. False if you think that there are still an action, or tool use to take."
        )
        action: str = pydantic.Field(
            description="Provide short and concise the optimal actions to take."
        )
        tools: List[str] = pydantic.Field(
            description="List of tools use in the action. Else leave it empty if there are not necessary to use any tools. If the send_message_to_human is True, this field should be empty.",
        )
        message_to_user: str = pydantic.Field(
            default="",
            description="The message to send to the user if send_message_to_user is True. Else leave it empty string.",
        )

        @pydantic.field_validator("tools")
        def validate_tools(self, v: list[str]) -> list[str]:
            if (self.send_message_to_user or self.message_to_user) and v:
                raise ValueError(
                    "Tools use should be empty if send_message_to_user True or message_to_user is not empty string. If you think there are still an action or tool use to take, set send_message_to_user to False and message_to_user to empty string."
                )
            return v

        @pydantic.field_validator("send_message_to_user")
        def validate_send_message_to_user(self, v: bool) -> bool:
            if v and self.tools:
                raise ValueError(
                    "send_message_to_user should be False if tools is not empty. If there are still an action or tool use to take, set send_message_to_user to False and message_to_user to empty string."
                )
            return v

        @pydantic.field_validator("message_to_user")
        def validate_message_to_user(self, v: str) -> str:
            if self.send_message_to_user and not v:
                raise ValueError(
                    "Message to user should not be empty if send_message_to_user is True."
                )
            return v

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        after=collect_errors(ValidationError),
    )
    @litellm.call(
        model=default_model_name, response_model=ReasoningAction, json_mode=True
    )
    @prompt_template(
        """
        MESSAGES: 
        {history}
        
        USER:
        {previous_errors}
        
        From last message or output, give your thought on the current situation and observations, then decide what actions to do. You can decide multiple actions at once, but be aware your limitations, as multiple actions might not be feasible and can give errors.
        Actions can be executed tools you currently have access, send messages to human, or continue to think and plan or anything necessary to achieve the goal.
        
        - Write down your thought
        - Write down the optimal actions to take
        - Write down the tools you will use if any. The tools name should be a list of tool names {tools_names}. 
        - Write down if you should send a message to user and end your turn.
        - Write down the message to send to user if send_message_to_user is True.
        """
    )
    async def _reasoning_action(
        self, *, errors: list[ValidationError] | None = None
    ) -> BaseDynamicConfig:
        """Reasoning about the action to take."""
        self.rotate_api_key()
        history = self.history
        tools_names = get_list_tools_name(self.get_tools())

        return {
            "computed_fields": {
                "history": history,
                "tools_names": tools_names,
                "previous_errors": f"Previous errors: {errors}",
            }
        }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    @litellm.call(model=default_model_name)
    async def _default_call(
        self, query: BaseMessageParam | None = None
    ) -> BaseDynamicConfig:
        messages = self._build_prompt()
        tools = self.get_tools()
        self.rotate_api_key()

        if query:
            messages.append(query)

        return {
            "messages": messages,
            "tools": tools,
            "call_params": {"temperature": self.temperature},
        }

    async def _process_tools(
        self, tools: List[OpenAITool], response: OpenAICallResponse
    ) -> None:
        """Process and execute tools called by the agent."""

        tools_and_outputs = []
        for tool in tools:
            print(f"Calling Tool '{tool._name()}' with args {tool.args}")
            output = None
            try:
                output = await tool.call()
            except ValidationError as e:
                output = f"Error calling tool {tool._name()} invalid input: {e}"
            except Exception as e:
                output = f"Error calling tool {tool._name()}: {e}"

            print(f"Tool output: {output}")
            tools_and_outputs.append((tool, output))

        tool_messages = response.tool_message_params

        messages = tool_messages(tools_and_outputs)

        self.history.extend(messages)

        if self.memory_manager:
            await self._store_tool_interactions(messages)

    def format_reasoning_response(self, response: ReasoningAction) -> str:
        """Format the reasoning response."""
        return f"Thought: {response.thought}\nAction: {response.action}\nExecute tools: {response.tools}\nShould I stop and send message: {response.send_message_to_user}"

    async def step(
        self,
        query: Messages.Type,
    ):
        """Execute one step of the agent's reasoning process."""
        try:
            await self.store_turn_message(query, "user")
            self.history.append(query)

            # chain of thought reasoning action loop
            reach_to_human = False
            message_to_user = ""
            while not reach_to_human or not message_to_user:
                # reasoning step
                reasoning_response = await self._reasoning_action()  # type: ignore

                # condition to stop and send back message
                reach_to_human = reasoning_response.send_message_to_user
                message_to_user = reasoning_response.message_to_user

                formatted_reasoning_response = self.format_reasoning_response(
                    reasoning_response
                )
                self.history.append(Messages.Assistant(formatted_reasoning_response))
                print(formatted_reasoning_response, sep="\n\n")
                await self.store_turn_message(
                    Messages.Assistant(formatted_reasoning_response), "assistant"
                )

                if reasoning_response.tools:
                    action_query = Messages.User(
                        f"Let's executing the action with tools: {reasoning_response.tools}"
                    )
                    self.history.append(action_query)

                    tool_action_response = await self._default_call()  # type: ignore

                    self.history.append(tool_action_response.message_param)
                    await self.store_turn_message(
                        tool_action_response.message_param, "assistant"
                    )

                    while not tool_action_response.tools:
                        action_query = Messages.User(
                            f"You should use the tools: {tool_action_response.tools} to execute the action."
                        )
                        self.history.append(action_query)

                        tool_action_response = await self._default_call()  # type: ignore
                        self.history.append(tool_action_response.message_param)
                        await self.store_turn_message(
                            tool_action_response.message_param, "assistant"
                        )

                    if tools := tool_action_response.tools:
                        await self._process_tools(tools, tool_action_response)

                # response = await self._default_call(Messages.User(""))  # type: ignore
                # self.history.append(response.message_param)  # type: ignore
                # await self.store_turn_message(response.message_param, "assistant")  # type: ignore
                # if tools := response.tools:
                #     await self._process_tools(tools, response)
                # response = await self._default_call(Messages.User(""))
                time.sleep(1)

            return message_to_user

        except Exception as e:
            logger.error(
                f"Error in agent step: {e}. Traceback: {traceback.format_exc()}."
            )
            print(self.history)

            raise e

    async def run(self) -> None:
        """Run the agent in an interactive loop."""
        try:
            await self.initialize_conversation()

            while True:
                print("[User]: ", end="", flush=True)
                query = input("")
                if query.lower() in ["exit", "quit"]:
                    break

                query = Messages.User(query)
                print("[Assistant]: ", end="", flush=True)

                response = await self.step(query)
                print(response)

        except Exception as e:
            logger.error(f"Error in run: {e}")
            raise e
        finally:
            print(self.history)
            print("-" * 50)
            if self.memory_manager:
                if len(self.history) >= 4:
                    user_feedback = input(
                        f"Please provide your feedback for the conversation with {self.agent_id}: "
                    )
                    await self.memory_manager.reflection_conversation(user_feedback)

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

            logger.debug(
                f"Enhanced system prompt with short-term memory context:\n{self._build_system_prompt()}"
            )

        except Exception as e:
            logger.error(f"Error initializing conversation: {e}")
            logger.info("Using default system prompt")

    def convert_message_to_base_message_param(
        self, message: BaseMessageParam | Messages.Type | str | dict, role: str
    ) -> BaseMessageParam:
        if isinstance(message, BaseMessageParam):
            return message

        if isinstance(message, dict):
            return BaseMessageParam(
                role=role,
                content=str(message["tool_calls"]),
            )

        return BaseMessageParam(role=role, content=str(message))

    async def store_turn_message(
        self,
        message: Messages.Type | str | dict,
        role: str = "assistant",
        message_type: MessageType = MessageType.TEXT,
    ) -> None:
        """Store a single turn message in memory."""
        if self.memory_manager:
            try:
                message = self.convert_message_to_base_message_param(message, role)
            except Exception as e:
                logger.error(f"Error converting message to base message param: {e}")
                return

            await self.memory_manager.store_conversation(
                sender=message.role,
                message_content=str(message.content),
                message_type=message_type,
                conversation_id=self.conversation_id,
            )

    async def _store_tool_interactions(self, messages: List) -> None:
        """Store tool interactions in memory."""
        if self.memory_manager:

            for msg in messages:
                try:
                    await self.memory_manager.store_conversation(
                        sender="Tool",
                        message_content=f"{msg['name']}: {msg['content']}",
                        message_type=MessageType.TOOL,
                        conversation_id=self.conversation_id,
                    )
                except Exception as e:
                    logger.error(f"Error storing tool interaction: {e}")
