# Standard library imports
from dataclasses import dataclass, field
import hashlib
import inspect
import os
import traceback
import uuid
from typing import Any, Callable, List, Optional, Type, Union, Dict, Tuple

# Third-party imports
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
from pydantic import ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors

# Local imports
from src.memory.memory_manager import MemoryManager
from src.memory.models import ConversationSummary, MessageType, ContextMemory
from src.memory.memory_toolkit.long_term import get_memory_toolkit
from src.memory.memory_toolkit.short_term import ShortTermMemoryToolKit
from src.memory.memory_toolkit.static_flow.init_agent import AgentInitializer
from src.util.rotating_list import RotatingList
from src.core.prompt.tool_prompt import (
    build_prompt_from_list_tools,
    get_list_tools_name,
)
from src.core.prompt.error_prompt import format_error_message
from src.core.prompt.system import (
    build_system_prompt,
    build_context_memory_prompt,
    build_recent_conversation_context,
)
from src.interface import ConsoleInterface, BaseInterface
from src.core.reasoning.base import BaseReasoningEngine
from src.core.tools.state_mangement import StateManagementToolKit, AgentStateManager

# Constants
GEMINI_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

UPDATE_CONFIG_PREFIX = "--"
EXIT_COMMANDS = ["exit", "quit"]


def generate_agent_id() -> str:
    """Generate a unique 8-character agent identifier."""
    random_id = str(uuid.uuid4())
    return hashlib.sha256(random_id.encode()).hexdigest()[:8]


@dataclass
class BaseAgent:
    """Base class for all agents with memory integration.

    This class provides core functionality for:
    1. Agent initialization and configuration
    2. Memory management and context handling
    3. Tool execution and management
    4. Conversation handling and persistence
    5. API key rotation
    """

    # Model Configuration
    default_model: str = "gemini/gemini-1.5-flash-002"
    reflection_model: str = "gemini/gemini-1.5-pro-002"
    temperature: float = 0.2
    max_retries: int = 5

    # Agent Identity
    agent_id: str = field(default_factory=generate_agent_id)
    system_prompt: str = "You are an AI agent."

    # Memory Components
    memory_manager: Optional[MemoryManager] = None
    context_memory: Optional[ContextMemory] = None
    short_term_memory: Optional[ShortTermMemoryToolKit] = None
    recent_conversations: List[ConversationSummary] = field(default_factory=list)
    num_recent_conversations: int = 4

    # Tool Management
    tools: List[Union[Type[BaseTool], Callable]] = field(default_factory=list)

    # API Configuration
    api_keys: Optional[List[str]] = None
    rotating_api_keys: Optional[RotatingList] = None
    api_key_env_var: Optional[str] = None

    # Conversation State
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    max_history: Optional[int] = None
    history: List[Union[BaseMessageParam, Messages.Type, gemini.GeminiMessageParam]] = (
        field(default_factory=list)
    )

    # Interface
    interface: BaseInterface = field(default_factory=lambda: ConsoleInterface())

    # State Management
    state_manager: AgentStateManager = field(default_factory=AgentStateManager)

    def __post_init__(self) -> None:
        """Initialize the agent after dataclass initialization."""
        self._initialize_agent()
        self._setup_api_keys()
        self._initialize_tools()

    # Initialization Methods
    def _initialize_agent(self) -> None:
        """Initialize agent configuration and print initial setup information."""
        self._print_initialization_info()
        self._initialize_short_term_memory()

    def _print_initialization_info(self) -> None:
        """Print agent configuration details."""
        config_info = {
            "Agent ID": self.agent_id,
            "Default Model": self.default_model,
            "Slow Model": self.reflection_model,
            "Temperature": self.temperature,
            "List of Chat Engines": self.state_manager.chat_config.registry.list_engines(),
        }

        for key, value in config_info.items():
            self.interface.print_system_message(f"{key}: {value}")

    def _initialize_short_term_memory(self) -> None:
        """Initialize the short-term memory component."""
        self.short_term_memory = ShortTermMemoryToolKit()
        self.short_term_memory.set_agent(self)
        self.short_term_memory._initialize()

    def _setup_api_keys(self) -> None:
        """Configure API key rotation if multiple keys are provided."""
        if not self.api_keys:
            return

        if not self.api_key_env_var:
            raise ValueError("API key environment variable not set")

        self.interface.print_system_message(
            f"Rotating through {len(self.api_keys)} api keys"
        )
        self.interface.print_system_message(f"Api env var: {self.api_key_env_var}")
        self.rotating_api_keys = RotatingList(self.api_keys)

    def _initialize_tools(self) -> None:
        """Initialize memory toolkit and available tools."""
        state_toolkit = StateManagementToolKit(state_manager=self.state_manager)
        self.add_tools(state_toolkit.create_tools())

        if self.memory_manager:
            dynamic_toolkit = get_memory_toolkit(self.memory_manager)
            self.add_tools(dynamic_toolkit.create_tools())
            self.memory_manager.set_agent(self)

        if self.short_term_memory:
            self.add_tools(self.short_term_memory.create_tools())

    # Tool Management Methods
    def add_tools(self, tools: List[Union[Type[BaseTool], Callable]]) -> None:
        """Add new tools to the agent's toolkit."""
        self.tools.extend(tools)

    def get_tools(self) -> List[Union[Type[BaseTool], Callable]]:
        """Get the list of tools available to this agent."""
        return self.tools

    def rotate_api_key(self) -> None:
        """Rotate to the next available API key if multiple keys are configured."""
        if self.rotating_api_keys and self.api_key_env_var:
            api_key = self.rotating_api_keys.rotate()
            os.environ[self.api_key_env_var] = str(api_key)

    # Prompt Building and Message Handling
    async def initialize_conversation(self) -> None:
        """Initialize conversation by loading best prompt and short-term memory."""
        self.interface.print_system_message("Initializing conversation...")

        if not self.memory_manager:
            self.interface.print_system_message(
                "No memory manager available - using default initialization"
            )
            return

        # Check if this is first interaction (no context memory)
        context_memory = await self.memory_manager.get_context_memory()
        if not context_memory:
            await AgentInitializer.initialize_agent(self)

        # Continue with existing initialization...
        reflection_performed = await self.memory_manager.check_and_perform_reflection()

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
        self.context_memory = await self.memory_manager.get_context_memory()

        self.interface.print_system_message(
            f"Prompt after initializing:\n{self._build_prompt()[0].content}"
        )

    @prompt_template(
        """
        SYSTEM: 
        
        # INSTRUCTIONS
        
        {system_prompt}
        
        # SHORT-TERM MEMORY
        
        {memories_prompt}
        
        # CONTEXT MEMORY 
        
        {context_memory_prompt}
        
        ## RECENT CONVERSATION SUMMARY
        
        {recent_conversation_context}
        
        # CHAT MODE STATE
        
        {state_prompt}
        
        # AVAILABLE TOOLS: 
        
        {tools_prompt}
        
        MESSAGES: {history}
        """
    )
    def _build_prompt(
        self,
        include_history: bool = True,
        include_context_memory: bool = True,
        include_system_prompt: bool = True,
        include_tools_prompt: bool = True,
        include_memories_prompt: bool = True,
        include_recent_conversation_context: bool = True,
        include_state_prompt: bool = True,
    ) -> BaseDynamicConfig:
        """Build the prompt for the agent using the current state."""
        computed_fields = {
            "system_prompt": (
                build_system_prompt(self, self.recent_conversations)
                if include_system_prompt
                else ""
            ),
            "context_memory_prompt": (
                build_context_memory_prompt(self) if include_context_memory else ""
            ),
            "history": self.history if include_history else [],
            "tools_prompt": (
                build_prompt_from_list_tools(self.get_tools())
                if include_tools_prompt and self.get_tools()
                else ""
            ),
            "memories_prompt": (
                self.short_term_memory.format_memories()
                if include_memories_prompt and self.short_term_memory
                else ""
            ),
            "recent_conversation_context": (
                build_recent_conversation_context(self.recent_conversations)
                if include_recent_conversation_context
                else ""
            ),
            "state_prompt": (
                self.state_manager.format_state() if include_state_prompt else ""
            ),
        }

        return {"computed_fields": computed_fields}

    async def _assitant_turn_message(self, message: BaseMessageParam) -> None:
        """Process and store an assistant's message in the conversation history.

        Args:
            message: The message from the assistant to process
        """
        self.history.append(message)
        await self.store_turn_message(message, "assistant")

    async def _default_call(
        self, query: Optional[BaseMessageParam] = None, include_tools: bool = True
    ):
        """Make a default call to the agent with the current state and context history.

        Args:
            query: Optional query to include in the call
            include_tools: Whether to include available tools

        Returns:
            BaseDynamicConfig: The configuration for the call
        """

        messages = self._build_prompt()
        tools = self.get_tools() if include_tools else []
        self.rotate_api_key()

        if query:
            messages.append(query)

        @litellm.call(
            model=self.default_model, call_params={"temperature": self.temperature}
        )
        async def call(
            *, errors: list[ValidationError] | None = None
        ) -> BaseDynamicConfig:
            self.rotate_api_key()

            return {
                "messages": messages,
                "tools": tools,
            }

        return await call()

    async def _process_tools(
        self, tools: List[BaseTool], response: OpenAICallResponse
    ) -> List:
        """Process and execute tools called by the agent.

        Args:
            tools: List of tools to execute
            response: The response from the agent

        Returns:
            List: Messages from tool executions
        """
        tools_and_outputs = []
        self.interface.print_system_message(
            f"Agent is executing tools: {get_list_tools_name(tools)}"
        )

        for tool in tools:
            output = await self._execute_tool(tool)
            tools_and_outputs.append((tool, output))

        tool_messages = response.tool_message_params(tools_and_outputs)

        if self.memory_manager:
            await self._store_tool_interactions(tool_messages)

        return tool_messages

    async def _execute_tool(self, tool: BaseTool) -> Any:
        """Execute a single tool and handle its output.

        Args:
            tool: The tool to execute

        Returns:
            Any: The output from the tool execution

        Raises:
            ValidationError: If tool input validation fails
            Exception: If tool execution fails
        """
        self.interface.print_system_message(
            f"Calling Tool '{tool._name()}' with args {tool.args}"
        )
        try:
            output = (
                await tool.call()
                if inspect.iscoroutinefunction(tool.call)
                else tool.call()
            )
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
        self,
        include_tools: bool = True,
        *,
        errors: Optional[List[ValidationError]] = None,
    ) -> Tuple[bool, OpenAICallResponse]:
        """Execute the default step of the agent with current state and context history.

        Args:
            include_tools: Whether to include tools in the step
            errors: Optional list of validation errors from previous attempts

        Returns:
            Tuple[bool, OpenAICallResponse]: Whether tools were used and the response
        """
        if errors:
            error_message = Messages.User(
                inspect.cleandoc(
                    f"""
                    system-automessage@noreply: ERROR:
                    There are some errors in the last step when indicating that you passed wrong parameters when using tools. 
                    Error: <> {format_error_message(errors)} </>
                    Correct the parameters indicated in the error message and re-execute the action immediately without further discussion.
                    """
                )
            )
            self.history.append(error_message)
            self.interface.print_user_message(error_message.content)
            await self.store_turn_message(error_message, "user")

        response = await self._default_call(include_tools=include_tools)
        self.interface.print_agent_message(response.content)
        await self._assitant_turn_message(response.message_param)

        if response.tools:
            tool_output_messages = await self._process_tools(response.tools, response)
            self.history.extend(tool_output_messages)
            return True, response

        return False, response

    async def step(self, query: Messages.Type | None = None) -> Optional[str]:
        """Execute one step of the agent's reasoning process."""
        try:
            if query:
                await self.store_turn_message(query, "user")
                self.history.append(query)

            current_engine = self.state_manager.chat_config.current_engine

            if current_engine:
                await current_engine.run(self)
            else:
                # chat mode
                use_tool_call, response = await self._default_step()
                if use_tool_call:
                    return await self.step(Messages.User(""))

        except Exception as e:
            error_msg = f"Error in agent step: {e}. Traceback: {traceback.format_exc()}"
            self.interface.print_system_message(error_msg, type="error")
            return str(e)

        return None

    async def run(self) -> None:
        """Run the agent in an interactive loop."""
        try:
            await self.initialize_conversation()
            self.interface.print_system_message(
                "Type exit or quit to end the conversation."
            )

            while True:
                query = self.interface.input("[User]: ")

                # if query.startswith(UPDATE_CONFIG_PREFIX):
                #     self.update_config(query[len(UPDATE_CONFIG_PREFIX) :])
                #     continue

                if query.lower() in EXIT_COMMANDS:
                    break

                await self.step(Messages.User(query))

        except Exception as e:
            error_msg = f"Error in run: {e}. Traceback: {traceback.format_exc()}"
            self.interface.print_system_message(error_msg, type="error")
            raise
        finally:
            await self._handle_conversation_end()

    async def _handle_conversation_end(self) -> None:
        """Handle tasks that need to be performed when a conversation ends."""
        self.interface.print_history(self.history)

        if self.memory_manager and len(self.history) >= 6:
            task = None
            # save short term memories
            if self.short_term_memory:
                task = self.short_term_memory.save_memories()

            user_feedback = self.interface.input(
                f"Please provide your feedback for the conversation with {self.agent_id}: "
            )
            await self.memory_manager.reflection_conversation(user_feedback)

            if task:
                await task

    def norm_message_type(
        self,
        message: Union[
            BaseMessageParam, Messages.Type, str, dict, gemini.GeminiMessageParam
        ],
        role: str,
    ) -> BaseMessageParam:
        """Normalize different message types into a BaseMessageParam.

        Args:
            message: The message to normalize
            role: The role of the message sender

        Returns:
            BaseMessageParam: The normalized message
        """
        content = getattr(message, "content", None)
        if content is None:
            content = getattr(message, "parts", None)

        if content is None and isinstance(message, dict):
            content = message.get("content", None)

        if content is None:
            content = str(message)

        return BaseMessageParam(role=role, content=str(content))

    async def store_turn_message(
        self,
        message: Union[Messages.Type, str, dict],
        role: str = "assistant",
        message_type: MessageType = MessageType.TEXT,
    ) -> None:
        """Store a single turn message in the database.

        Args:
            message: The message to store
            role: The role of the message sender
            message_type: The type of the message
        """
        if not self.memory_manager:
            return

        try:
            message = self.norm_message_type(message, role)
            if not message.content:
                return

            await self.memory_manager.store_conversation(
                sender=message.role,
                message_content=str(message.content),
                message_type=message_type,
                conversation_id=self.conversation_id,
            )
        except Exception as e:
            self.interface.print_system_message(
                f"Error storing message: {e}. Message: {message}", type="error"
            )

    async def _store_tool_interactions(self, messages: List) -> None:
        """Store tool interactions in the database.

        Args:
            messages: List of tool interaction messages to store
        """
        if not self.memory_manager:
            return

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

    # def _update_reasoning_engine(self) -> None:
    #     """Update the reasoning engine based on current state."""
    #     engine = self.state_manager.chat_config.registry.get_engine(
    #         self.state_manager.chat_config.current_engine.name
    #     )
    #     if engine:
    #         self.reasoning_engine = engine
