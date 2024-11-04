from dataclasses import dataclass, field
import hashlib
import inspect
import uuid
from typing import Any, Callable, List, Optional, Type, Union
import traceback
from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    BaseTool,
    Messages,
    prompt_template,
    litellm,
    gemini,
)

from mirascope.core.openai import OpenAICallResponse
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import ValidationError
from src.memory.memory_manager import MemoryManager
from src.memory.models import ConversationSummary, MessageType, ShortTermMemory
from src.memory.memory_toolkit.long_term import get_memory_toolkit
from src.util.rotating_list import RotatingList
import os
import pyperclip
from src.core.prompt.tool_prompt import (
    build_prompt_from_list_tools,
    get_list_tools_name,
)
from src.core.prompt.error_prompt import format_error_message
from src.core.prompt.system import (
    build_system_prompt,
    build_short_term_memory_prompt,
    build_recent_conversation_context,
)
from src.core.prompt.tool_prompt import (
    build_prompt_from_list_tools,
    get_list_tools_name,
)
from src.interface import ConsoleInterface, BaseInterface

from src.core.reasoning.base import BaseReasoningEngine
from src.memory.memory_toolkit.note_taking import (
    NoteTakingToolkit,
    format_notes,
    save_notes,
    load_notes,
)


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

    default_model: str = "gemini/gemini-1.5-flash-002"
    reflection_model: str = "gemini/gemini-1.5-pro-002"

    agent_id: str = field(default_factory=generate_agent_id)
    system_prompt: str = "You are an AI agent."
    temperature: float = 0.5
    max_retries: int = 5

    memory_manager: Optional[MemoryManager] = None
    context_memory: Optional[ShortTermMemory] = None
    tools: List[Union[Type[BaseTool], Callable]] = field(default_factory=list)

    reasoning_engine: Optional[BaseReasoningEngine] = None
    note_taking_toolkit: NoteTakingToolkit = field(default_factory=NoteTakingToolkit)

    api_keys: list[str] | None = None
    rotating_api_keys: RotatingList | None = None
    api_key_env_var: str | None = None

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_history: Optional[int] = None
    history: List[BaseMessageParam | Messages.Type | gemini.GeminiMessageParam] = field(
        default_factory=list
    )

    interface: BaseInterface = field(default_factory=lambda: ConsoleInterface())

    recent_conversations: List[ConversationSummary] = field(default_factory=list)
    num_recent_conversations: int = 4

    def __post_init__(self) -> None:
        """Initialize the agent after dataclass initialization."""
        self._initialize_agent()
        self._setup_api_keys()
        self._initialize_tools()

    def _initialize_agent(self) -> None:
        """Print initial agent configuration."""
        self.interface.print_system_message(f"Agent ID: {self.agent_id}")
        self.interface.print_system_message(f"Default Model: {self.default_model}")
        self.interface.print_system_message(f"Slow Model: {self.reflection_model}")
        self.interface.print_system_message(f"Temperature: {self.temperature}")

        if self.reasoning_engine:
            self.interface.print_system_message(
                f"Reasoning Engine: {self.reasoning_engine.__class__.__name__}"
            )

        self._initialize_note_taking_toolkit()

    def _setup_api_keys(self) -> None:
        """Setup API key rotation if configured."""
        if self.api_keys:
            if not self.api_key_env_var:
                raise ValueError("Api key env var is not set")

            self.interface.print_system_message(
                f"Rotating through {len(self.api_keys)} api keys"
            )
            self.interface.print_system_message(f"Api env var: {self.api_key_env_var}")
            self.rotating_api_keys = RotatingList(self.api_keys)

    def _initialize_tools(self) -> None:
        """Initialize tools and memory toolkit."""
        if self.memory_manager:
            dynamic_toolkit = get_memory_toolkit(self.memory_manager)
            self.add_tools(dynamic_toolkit.create_tools())
            self.memory_manager.set_agent(self)

            self.add_tools(self.note_taking_toolkit.create_tools())

    def _initialize_note_taking_toolkit(self) -> None:
        """Initialize the note taking toolkit."""
        self.note_taking_toolkit.notes = load_notes(self.agent_id)

    def rotate_api_key(self) -> None:
        """Rotate the api key."""
        if self.rotating_api_keys:
            if self.api_key_env_var:
                api_key = self.rotating_api_keys.rotate()
                os.environ[self.api_key_env_var] = str(api_key)

                # self.interface.print_system_message(
                #     f"Current api key: {os.environ[self.api_key_env_var]}"
                # )

    def add_tools(self, tools: List[Union[Type[BaseTool], Callable]]) -> None:
        self.tools.extend(tools)

    def get_tools(self) -> List[Union[Type[BaseTool], Callable]]:
        """Get the list of tools available to this agent."""
        return self.tools

    async def initialize_conversation(self) -> None:
        """Initialize conversation by loading best prompt and short-term memory."""
        self.interface.print_system_message("Initializing conversation...")

        if not self.memory_manager:
            self.interface.print_system_message(
                "No memory manager available - using default initialization"
            )
            return

        # Load latest summary for system prompt
        latest_summary = await self.memory_manager.get_latest_conversation_summary()
        if latest_summary:
            self.system_prompt = latest_summary.improve_prompt
            self.interface.print_system_message(
                f"Loaded system prompt: {self.system_prompt}"
            )

        # Load recent conversations for context
        self.recent_conversations = (
            await self.memory_manager.get_recent_conversation_summaries(
                limit=self.num_recent_conversations
            )
        )

        # Load short term memory
        self.context_memory = await self.memory_manager.get_short_term_memory()

        self.interface.print_system_message(
            f"Prompt after initializing:\n{self._build_prompt()[0].content}"
        )

    @prompt_template(
        """
        SYSTEM: 
    
        {system_prompt}
        
        {short_term_memory_prompt}
        
        {recent_conversation_context}
        
        # AVAILABLE TOOLS (REMEMBER TO USE THEM WHEN NECESSARY): 
        
        {tools_prompt}
        
        # AGENT'S NOTES (IMPORTANT INFORMATION, KNOWLEDGES, IDEAS, PLANS, ETC. IN THE CURRENT CONTEXT):
        
        {notes_prompt}
        
        MESSAGES: {history}
        """
    )
    def _build_prompt(
        self,
        include_history: bool = True,
        include_short_term_memory: bool = True,
        include_system_prompt: bool = True,
        include_tools_prompt: bool = True,
        include_notes_prompt: bool = True,
        include_recent_conversation_context: bool = True,
    ) -> BaseDynamicConfig:
        """Build the prompt for the agent using the current state."""
        system_prompt = (
            build_system_prompt(self, self.recent_conversations)
            if include_system_prompt
            else ""
        )
        short_term_memory_prompt = (
            build_short_term_memory_prompt(self) if include_short_term_memory else ""
        )

        history = self.history if include_history else []
        tools_prompt = (
            build_prompt_from_list_tools(self.get_tools())
            if include_tools_prompt and self.get_tools()
            else ""
        )

        notes_prompt = (
            format_notes(self.note_taking_toolkit.notes)
            if include_notes_prompt and self.note_taking_toolkit
            else ""
        )

        recent_conversation_context = (
            build_recent_conversation_context(self.recent_conversations)
            if include_recent_conversation_context
            else ""
        )

        return {
            "computed_fields": {
                "system_prompt": system_prompt,
                "short_term_memory_prompt": short_term_memory_prompt,
                "history": history,
                "tools_prompt": tools_prompt,
                "notes_prompt": notes_prompt,
                "recent_conversation_context": recent_conversation_context,
            }
        }

    async def _assitant_turn_message(self, message: BaseMessageParam) -> None:
        self.history.append(message)
        await self.store_turn_message(message, "assistant")

    @litellm.call(model=default_model)
    def _default_call(
        self, query: BaseMessageParam | None = None, include_tools: bool = True
    ) -> BaseDynamicConfig:
        """Default call to the agent with the current state and context history."""
        messages = self._build_prompt()
        tools = self.get_tools() if include_tools else []
        self.rotate_api_key()

        if query:
            messages.append(query)

        return {
            "messages": messages,
            "tools": tools,
        }

    async def _process_tools(self, tools: List[BaseTool], response: OpenAICallResponse):
        """Process and execute tools called by the agent."""
        tools_and_outputs = []
        self.interface.print_system_message(
            f"Agent is executing tools: {get_list_tools_name(tools)}"  # type: ignore
        )

        for tool in tools:
            output = await self._execute_tool(tool)
            tools_and_outputs.append((tool, output))

        tool_messages = response.tool_message_params(tools_and_outputs)

        if self.memory_manager:
            await self._store_tool_interactions(tool_messages)

        return tool_messages

    async def _execute_tool(self, tool: BaseTool) -> Any:
        """Execute a single tool and handle its output."""
        self.interface.print_system_message(
            f"Calling Tool '{tool._name()}' with args {tool.args}"
        )
        try:
            # if tool is async, it will be called as await tool.call()
            if inspect.iscoroutinefunction(tool.call):
                output = await tool.call()
            else:
                output = tool.call()

            self.interface.print_system_message(f"Tool output: {output}")
            return output
        except ValidationError as e:
            error_msg = f"Error calling tool {tool._name()} invalid input: {e.errors()}"
            self.interface.print_system_message(error_msg, type="error")
            return error_msg
        except Exception as e:
            error_msg = f"Error calling tool {tool._name()}: {e}. Traceback: {traceback.format_exc()}"
            self.interface.print_system_message(error_msg, type="error")
            return error_msg

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        after=collect_errors(ValidationError),
    )
    async def _default_step(
        self, include_tools: bool = True, *, errors: list[ValidationError] | None = None
    ) -> tuple[bool, OpenAICallResponse]:
        """Execute the default step of the agent. With current state and context history. Return True if the step agent use tool call."""
        use_tool_call = False
        correct_action_query = None

        if errors:

            correct_action_query = Messages.User(
                inspect.cleandoc(
                    f"""
                !! This is an remind auto message !! 
                There are some error in the last step when indicate that you pass the wrong parameters when use tool. 
                Error: <> {format_error_message([errors[-1]])} </>."
                Correct the parameters indicated in the error message and re-execute the action immediately without further discussion.
                """
                )
            )
            self.history.append(correct_action_query)
            self.interface.print_user_message(correct_action_query.content)

        # response call
        response = self._default_call(include_tools=include_tools)  # type: ignore
        self.interface.print_agent_message(response.content)
        # store agent response to history
        await self._assitant_turn_message(response.message_param)  # type: ignore

        # tool call
        if response.tools:
            tool_output_messages = await self._process_tools(
                response.tools, response
            )  # type: ignore
            # store message to history before tool, make sure the tool call is successful before add to history
            self.history.extend(tool_output_messages)  # type: ignore
            use_tool_call = True

        return use_tool_call, response

    async def step(
        self,
        query: Messages.Type,
    ):
        """Execute one step of the agent's reasoning process."""
        try:

            await self.store_turn_message(query, "user")
            self.history.append(query)

            # reasoning loop
            if self.reasoning_engine:
                await self.reasoning_engine.run(self)
            else:
                use_tool_call, response = await self._default_step()
                if use_tool_call:
                    return await self.step(Messages.User(""))

        except Exception as e:
            self.interface.print_system_message(
                f"Error in agent step: {e}. Traceback: {traceback.format_exc()}",
                type="error",
            )
            return str(e)

    def update_config(self, config_str: str) -> None:
        """Update the agent's configuration."""
        parameters = config_str.split(" ")
        if len(parameters) == 2:
            key, value = parameters

            setattr(self.reasoning_engine, key, value)
            self.interface.print_system_message(
                f"Updated {key} to {value}", type="info"
            )

    async def run(self) -> None:
        """Run the agent in an interactive loop."""
        try:
            await self.initialize_conversation()

            self.interface.print_system_message(
                "Type exit or quit to end the conversation."
            )

            update_config_prefix = "--"

            while True:
                query = self.interface.input("[User]: ")

                while query.startswith(update_config_prefix):
                    query = query[len(update_config_prefix) :]
                    self.update_config(query)

                    query = self.interface.input("[User]: ")

                if query.lower() in ["exit", "quit"]:
                    break

                query = Messages.User(query)
                await self.step(query)

        except Exception as e:
            self.interface.print_system_message(
                f"Error in run: {e}. Traceback: {traceback.format_exc()}", type="error"
            )
            raise e
        finally:
            self.interface.print_history(self.history)

            if self.memory_manager:
                if len(self.history) >= 4:
                    # save notes
                    save_notes(self.agent_id, self.note_taking_toolkit.notes)

                    user_feedback = self.interface.input(
                        f"Please provide your feedback for the conversation with {self.agent_id}: "
                    )
                    await self.memory_manager.reflection_conversation(user_feedback)

    def norm_message_type(
        self,
        message: (
            BaseMessageParam | Messages.Type | str | dict | gemini.GeminiMessageParam
        ),
        role: str,
    ) -> BaseMessageParam:

        # if isinstance(message, BaseMessageParam):
        #     return message

        # if isinstance(message, dict):
        #     if role.lower() != "tool":
        #         return BaseMessageParam(
        #             role=role,
        #             content=str(message["content"]),
        #         )
        #     else:
        #         return BaseMessageParam(
        #             role=role,
        #             content=str(message["tool_calls"]),
        #         )

        content = getattr(message, "content", None)
        if content is None:
            content = getattr(message, "parts", None)

        if content is None:
            content = str(message)

        return BaseMessageParam(role=role, content=str(content))

    async def store_turn_message(
        self,
        message: Messages.Type | str | dict,
        role: str = "assistant",
        message_type: MessageType = MessageType.TEXT,
    ) -> None:
        """Store a single turn message in database."""
        if self.memory_manager:
            try:
                message = self.norm_message_type(message, role)
            except Exception as e:
                self.interface.print_system_message(
                    f"Error converting message to base message param: {e}. Message: {message}",
                    type="error",
                )
                return

            if not message.content or len(message.content) == 0:
                return

            await self.memory_manager.store_conversation(
                sender=message.role,
                message_content=str(message.content),
                message_type=message_type,
                conversation_id=self.conversation_id,
            )

    async def _store_tool_interactions(self, messages: List) -> None:
        """Store tool interactions in database."""
        if self.memory_manager:

            for msg in messages:
                try:
                    msg = self.norm_message_type(msg, "tool")
                    await self.memory_manager.store_conversation(
                        sender="tool",
                        message_content=str(msg.content),
                        message_type=MessageType.TOOL,
                        conversation_id=self.conversation_id,
                    )
                except Exception as e:
                    self.interface.print_system_message(
                        f"Error storing tool interaction: {e}", type="error"
                    )
