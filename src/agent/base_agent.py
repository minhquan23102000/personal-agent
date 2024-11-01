from dataclasses import dataclass, field
import datetime
import hashlib
import inspect
import time
import uuid
from typing import Callable, List, Optional, Type, Union
import traceback
from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    BaseTool,
    Messages,
    prompt_template,
    litellm,
)

from mirascope.core.openai import OpenAICallResponse, OpenAITool
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import ValidationError
import pydantic
from src.inputer.base import BaseInputer
from src.memory.memory_manager import MemoryManager
from src.memory.models import MessageType, ShortTermMemory
from src.memory.memory_toolkit.dynamic_flow import get_memory_toolkit
from src.util.rotating_list import RotatingList
import os
import pyperclip
from src.agent.tools.prompt import build_prompt_from_list_tools, get_list_tools_name
from src.agent.error_prompt import format_error_message
from src.log import ConsolePrinter, MultiPrinter
from src.inputer.console_inputer import ConsoleInputer


GEMINI_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]


def generate_agent_id() -> str:
    random_id = str(uuid.uuid4())
    return hashlib.sha256(random_id.encode()).hexdigest()[:8]


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
    max_thought_without_tools: int = 4
    max_retries: int = 3
    printer: MultiPrinter = field(
        default_factory=lambda: MultiPrinter([ConsolePrinter()])
    )
    inputer: BaseInputer = field(default_factory=lambda: ConsoleInputer())

    def __post_init__(self):
        self.printer.print_system_message(f"Agent ID: {self.agent_id}")
        self.printer.print_system_message(f"Default Model: {self.default_model_name}")
        self.printer.print_system_message(f"Slow Model: {self.slow_model_name}")
        self.printer.print_system_message(f"Temperature: {self.temperature}")

        if self.memory_manager:
            dynamic_toolkit = get_memory_toolkit(self.memory_manager)
            self.add_tools(dynamic_toolkit.create_tools())
            self.memory_manager.set_agent(self)

        if self.api_keys:
            self.printer.print_system_message(
                f"Rotating through {len(self.api_keys)} api keys"
            )
            self.printer.print_system_message(f"Api env var: {self.api_key_env_var}")
            if not self.api_key_env_var:
                raise ValueError("Api key env var is not set")

            self.rotating_api_keys = RotatingList(self.api_keys)

        self.printer.print_system_message(
            f"Tools: {get_list_tools_name(self.get_tools())}"
        )

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
<>
{self.system_prompt}
</>
            """
        ).strip()

    def _build_short_term_memory_prompt(self) -> str:
        """Enhance the system prompt with context from short-term memory."""

        if self.short_term_memory:
            short_memory_prompt = inspect.cleandoc(
                f"""
## AGENT INFORMATION & IDENTITY: 
{self.short_term_memory.agent_info}

## AGENT BELIEFS: 
{self.short_term_memory.agent_beliefs}

## USER INFORMATION: 
{self.short_term_memory.user_info}

## HOW TO ADDRESS USER: 
{self.short_term_memory.how_to_address_user}

## LAST CONVERSATION SUMMARY on {self.short_term_memory.timestamp.strftime("%Y-%m-%d %H:%M:%S")}:
{self.short_term_memory.last_conversation_summary}

## RECENT GOALS AND STATUS: 
{self.short_term_memory.recent_goal_and_status}

## IMPORTANT CONTEXT: 
{self.short_term_memory.important_context}

## ENVIRONMENT INFORMATION: 
{self.short_term_memory.environment_info}
                """
            )

        else:
            short_memory_prompt = "This is your first interaction with the user."

        return short_memory_prompt.strip()

    @prompt_template(
        """
        SYSTEM: 
        {system_prompt}
        {short_term_memory_prompt}
        
        # LIST OF TOOLS YOU CAN ACCESS (REMEMBER TO USE THEM WHEN NECESSARY): 
        <>
        {tools_prompt}
        </>
        
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

        feeling: str = pydantic.Field(
            description="Note your feeling about the current situation, obervation. Using short keywords to describe your feeling, format in uppercase. If you don't have any feeling, just leave this empty string."
        )
        thought: str = pydantic.Field(
            description="Note your thought or plan for the current situation, obervation. Be concise and clear."
        )
        goal_completed: bool = pydantic.Field(
            description="True if you think you have completed the final goal instructed by user or you have the final answer. False otherwise.",
            default=False,
        )
        talk_to_user: bool = pydantic.Field(
            description="Decide if you should halt actions and talk to the user. Feel free to talk to me if you need something."
        )
        action: str = pydantic.Field(
            description="Identify the most effective actions to take. It is possible to combine multiple actions, but be mindful that some may not be feasible or easy to execute simultaneously, unless you are confident."
        )
        tools: List[str] = pydantic.Field(
            description="Consider the tools you will utilize for the action you plan to take. Only include the tools if you believe they will be necessary for this specific action. Do not add them indiscriminately; instead, think critically and plan carefully before making your selection. DO NOT ADD MULTIPLE TOOLS IF YOU THINK YOU CAN USE ONE TOOL TO COMPLETE THE ACTION. If no tools are deemed necessary, simply leave that section empty."
        )

    @prompt_template(
        """
        MESSAGES: 
        {messages}
        
        USER:
        {previous_errors}
        
        ## TOOLS YOU CAN ACCESS:
        <>
        {tools_names:list}
        </>
        
        Reflect on the previous message or output, noting your thoughts and feelings regarding the current situation and observations. Subsequently, determine the appropriate actions to take.

        - Record your feelings.
        - Record your thoughts.
        - Indicate whether you have completed the final goal.
        - Decide if you should halt actions and talk to the user.
        - Identify the optimal actions to pursue.
        - Note the tool you will use for the action you are going to take.
        """
    )
    def _build_reasoning_prompt(
        self, *, errors: list[ValidationError] | None = None
    ) -> BaseDynamicConfig:
        messages = self._build_prompt()

        if errors:
            previous_errors = f"Previous Errors: <>{format_error_message(errors)}</>"
        else:
            previous_errors = ""

        return {
            "computed_fields": {
                "messages": messages,
                "tools_names": get_list_tools_name(self.get_tools()),
                "previous_errors": previous_errors,
            },
        }

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        after=collect_errors(ValidationError),
    )
    @litellm.call(
        model=default_model_name,
        response_model=ReasoningAction,
        json_mode=True,
    )
    async def _reasoning_action(
        self, *, errors: list[ValidationError] | None = None
    ) -> BaseDynamicConfig:
        """Reasoning about the action to take."""
        self.rotate_api_key()

        reasoning_prompt = self._build_reasoning_prompt(errors=errors)
        if errors:
            self.printer.print_system_message(
                f"Error in reasoning: {format_error_message(errors)}", type="error"
            )

        return {
            "messages": reasoning_prompt,
        }

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    @litellm.call(model=default_model_name)
    async def _default_call(
        self, query: BaseMessageParam | None = None, include_tools: bool = True
    ) -> BaseDynamicConfig:
        """Default call to the agent."""
        messages = self._build_prompt()
        tools = self.get_tools() if include_tools else []
        self.rotate_api_key()

        if query:
            messages.append(query)

        return {
            "messages": messages,
            "tools": tools,
        }

    async def _process_tools(
        self, tools: List[OpenAITool], response: OpenAICallResponse
    ):
        """Process and execute tools called by the agent."""
        tools_and_outputs = []
        self.printer.print_system_message(
            f"Agent is executing tools: {get_list_tools_name(tools)}"
        )

        for tool in tools:
            self.printer.print_system_message(
                f"Calling Tool '{tool._name()}' with args {tool.args}"
            )
            try:
                output = await tool.call()
                self.printer.print_system_message(f"Tool output: {output}")
                tools_and_outputs.append((tool, output))
            except ValidationError as e:
                error_msg = (
                    f"Error calling tool {tool._name()} invalid input: {e.errors()}"
                )
                self.printer.print_system_message(error_msg, type="error")
                tools_and_outputs.append((tool, error_msg))
            except Exception as e:
                error_msg = f"Error calling tool {tool._name()}: {e}. Traceback: {traceback.format_exc()}"
                self.printer.print_system_message(error_msg, type="error")
                tools_and_outputs.append((tool, error_msg))

        # Get tool messages and add them to history
        tool_messages = response.tool_message_params(tools_and_outputs)

        # Store tool interactions in memory if available
        if self.memory_manager:
            await self._store_tool_interactions(tool_messages)

        return tool_messages

    def format_reasoning_response(self, response: ReasoningAction) -> str:
        """Format the reasoning response."""
        return inspect.cleandoc(
            f"""
            My feeling: {response.feeling}
            My thought: {response.thought}
            Do I reach the final goal: {"Yes" if response.goal_completed else "No"}
            Should I talk to {self.short_term_memory.how_to_address_user if self.short_term_memory else "user"}: {"Yes" if response.talk_to_user else "No"}
            My Action: {response.action}
            Available tools for action: {response.tools}
            """
        )

    async def _assitant_turn_message(self, message: BaseMessageParam) -> None:
        self.history.append(message)
        await self.store_turn_message(message, "assistant")

    @retry(
        stop=stop_after_attempt(max_retries),
        after=collect_errors(ValidationError),
    )
    async def _action_step(
        self,
        action: str,
        tools: List[str],
        *,
        errors: list[ValidationError] | None = None,
    ) -> None:
        """Execute the action with tools."""

        if errors:
            correct_action_query = Messages.User(
                inspect.cleandoc(
                    f"""
                !! This is an remind auto message !! 
                There are some error in the last step when indicate that you pass the wrong parameters when use tool. 
                Error: <> {format_error_message(errors)} </>."
                Correct the parameters indicated in the error message and re-execute the action: {action} immediately without further discussion.
                """
                )
            )
            self.history.append(
                correct_action_query
            )  # append to history for agent learn from the mistaske.
            self.printer.print_user_message(correct_action_query)

        # call to execute the action
        tool_action_response = await self._default_call()  # type: ignore
        await self._assitant_turn_message(tool_action_response.message_param)
        self.printer.print_agent_message(tool_action_response.content)

        # agent response with tools call, process the tools call and return the output to history
        if tool_action_response.tools:
            tool_output_messages = await self._process_tools(
                tool_action_response.tools, tool_action_response
            )
            self.history.extend(tool_output_messages)

    async def _react_loop(self) -> None:
        """React loop to reason and act."""
        # chain of thought reasoning action loop
        num_iterate_without_tools = 0
        while True:
            # reasoning step
            reasoning_response = await self._reasoning_action()  # type: ignore
            formatted_reasoning_response = self.format_reasoning_response(
                reasoning_response
            )
            await self._assitant_turn_message(
                Messages.Assistant(formatted_reasoning_response)
            )

            # print the reasoning response
            self.printer.print_agent_message(formatted_reasoning_response)

            # if agent decide to talk to user, break the chain of thought
            if reasoning_response.talk_to_user:
                break

            # if there are tools to execute, execute the action with tools, else continue the react chain of thought
            # if reasoning_response.tools:
            #     await self._action_step(
            #         reasoning_response.action, reasoning_response.tools
            #     )
            #     num_iterate_without_tools = 0
            # else:
            #     # self.printer.print_user_message(action_query)
            #     action_without_tools = await self._default_call(include_tools=False)  # type: ignore
            #     await self._assitant_turn_message(action_without_tools.message_param)
            #     self.printer.print_agent_message(action_without_tools.content)

            #     num_iterate_without_tools += 1

            # agent execute the action
            await self._action_step(reasoning_response.action, reasoning_response.tools)

            # if agent decide to complete the task, break the chain of thought
            if reasoning_response.goal_completed:
                break

            # agent plan and think not action, count the times
            if not reasoning_response.tools:
                num_iterate_without_tools += 1
            else:
                num_iterate_without_tools = 0

            if (
                num_iterate_without_tools > self.max_thought_without_tools
            ):  # stop after 5 times of thought without doing any action
                break

            time.sleep(2)

    async def step(
        self,
        query: Messages.Type,
    ):
        """Execute one step of the agent's reasoning process."""
        try:
            await self.store_turn_message(query, "user")
            self.history.append(query)

            # react loop
            await self._react_loop()

            # final call to talk to human or when complete task, after react loop complete
            final_response = await self._default_call(
                include_tools=False
            )  # type: ignore
            await self._assitant_turn_message(final_response.message_param)

            return final_response.content

        except Exception as e:
            self.printer.print_system_message(
                f"Error in agent step: {e}. Traceback: {traceback.format_exc()}",
                type="error",
            )
            return str(e)

    async def run(self) -> None:
        """Run the agent in an interactive loop."""
        try:
            await self.initialize_conversation()
            self.printer.print_system_message("Type pcb to paste clipboard content.")
            self.printer.print_system_message(
                "Type exit or quit to end the conversation."
            )

            while True:
                query = self.inputer.input("[User]: ")

                # Check if the input is empty and if so, try to get the clipboard content
                if query == "pcb":
                    query = pyperclip.paste()

                if query.lower() in ["exit", "quit"]:
                    break

                query = Messages.User(query)
                self.printer.print_user_message(query.content)

                response = await self.step(query)
                self.printer.print_agent_message(response)

        except Exception as e:
            self.printer.print_system_message(
                f"Error in run: {e}. Traceback: {traceback.format_exc()}", type="error"
            )
            raise e
        finally:
            self.printer.print_history(self.history)

            if self.memory_manager:
                if len(self.history) >= 4:
                    user_feedback = self.inputer.input(
                        f"Please provide your feedback for the conversation with {self.agent_id}: "
                    )
                    await self.memory_manager.reflection_conversation(user_feedback)

    async def initialize_conversation(self) -> None:
        """Initialize conversation by loading best prompt and short-term memory."""
        self.printer.print_system_message("Initializing conversation...")

        if not self.memory_manager:
            self.printer.print_system_message(
                "No memory manager available - using default initialization"
            )
            return

        latest_summary = await self.memory_manager.get_latest_conversation_summary()
        if latest_summary:
            self.system_prompt = latest_summary.improve_prompt
            self.printer.print_system_message(
                f"Loaded system prompt: {self.system_prompt}"
            )

        self.short_term_memory = await self.memory_manager.get_short_term_memory()

        self.printer.print_system_message(
            f"Prompt after initializing:\n{self._build_prompt()[0].content}"
        )

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
                self.printer.print_system_message(
                    f"Error converting message to base message param: {e}", type="error"
                )
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
                    self.printer.print_system_message(
                        f"Error storing tool interaction: {e}", type="error"
                    )
