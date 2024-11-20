from dataclasses import dataclass, field
from typing import Dict, List, Type, Optional
from mirascope.core import BaseTool, BaseToolKit, toolkit_tool
from pydantic import Field
from loguru import logger

from src.core.reasoning.base import BaseReasoningEngine
from src.core.reasoning.react import ReActEngine
from src.core.reasoning.chat import ChatEngine


@dataclass
class ReasoningEngineRegistry:
    """Registry for managing available reasoning engines."""

    engines: Dict[str, BaseReasoningEngine] = field(default_factory=dict)

    def register(self, engine: BaseReasoningEngine) -> None:
        """Register a new reasoning engine."""
        self.engines[engine.name] = engine

    def get_engine(self, name: str) -> BaseReasoningEngine | None:
        """Get a reasoning engine by name."""
        return self.engines.get(name)

    def list_engines(self) -> list[str]:
        """List all registered engine names."""
        return list(self.engines.keys())


@dataclass
class AgentChatModeConfig:
    """Configuration for agent mode settings."""

    registry: ReasoningEngineRegistry = field(default_factory=ReasoningEngineRegistry)
    current_engine: BaseReasoningEngine = field(default_factory=lambda: ChatEngine())

    def __post_init__(self):
        """Initialize registry with default engines."""
        self.registry.register(self.current_engine)


@dataclass
class AgentStateManager:
    """Manages the state of the agent."""

    chat_config: AgentChatModeConfig = field(default_factory=AgentChatModeConfig)

    def format_state(self) -> str:
        """Format the current state for inclusion in system prompt."""
        engines = self.chat_config.registry.list_engines()
        current_engine = self.chat_config.current_engine

        formatted = ""
        if current_engine:
            formatted += f"- Current Chat Engine: {current_engine}\n"
            if current_engine.state_prompt:
                formatted += f"\n{current_engine.state_prompt}\n"
        formatted += "- Available Chat Engines (engine_name: description):\n"
        for engine_name in engines:
            engine = self.chat_config.registry.get_engine(engine_name)
            if engine:
                formatted += f"  * {engine_name}: {engine.description}\n"

        formatted += "\n\nRemember to use the state_manager.switch_chat_mode(engine_name) to change the chat mode when necessary."

        return formatted

    def register_list_engines(self, engines: List[BaseReasoningEngine]) -> None:
        """Register a list of reasoning engines."""
        logger.critical(f"Registering list of engines: {engines}")
        for engine in engines:
            self.chat_config.registry.register(engine)


class StateManagementToolKit(BaseToolKit):
    """Toolkit for managing agent state and reasoning engines."""

    __namespace__ = "state_management"

    state_manager: AgentStateManager = field(default_factory=AgentStateManager)

    @toolkit_tool
    def switch_chat_mode(self, engine_name: str) -> str:
        """Switch to a different chat mode.

        Args:
            engine: The name of the chat engine to use.
                   Available engines can be checked in the system prompt.
        """
        available_engines = self.state_manager.chat_config.registry.list_engines()

        if engine_name not in available_engines:
            return f"Error: Engine '{engine_name}' not found. Available engines: {', '.join(available_engines)}"

        engine = self.state_manager.chat_config.registry.get_engine(engine_name)
        if not engine:
            return f"Error: Could not initialize engine '{engine_name}'"

        previous_engine = self.state_manager.chat_config.current_engine
        self.state_manager.chat_config.current_engine = engine

        return (
            f"Reasoning engine changed from '{previous_engine}' to '{engine}'. "
            f"The agent will use this engine for complex reasoning tasks when reasoning_mode is enabled."
        )
