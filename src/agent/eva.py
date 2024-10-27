from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Callable

from mirascope.core import BaseDynamicConfig, BaseTool, Messages, litellm

from src.agent.base_agent import BaseAgent
from src.memory import MemoryManager

from rich import print
from src.config import GOOGLE_API_KEY_LIST

print(GOOGLE_API_KEY_LIST)


@dataclass
class Eva(BaseAgent):
    """An agent that helps users find and recommend books."""

    system_prompt: str = (
        "You are a helpful agent. Assist the user with their questions and needs."
    )
    model_name: str = "gemini/gemini-1.5-flash-002"
    agent_id: str = "Eva - 9000"
    api_key_env_var: str = "GEMINI_API_KEY"
    api_keys: list[str] = field(default_factory=lambda: GOOGLE_API_KEY_LIST)

    def __post_init__(self):
        self.memory_manager = MemoryManager(db_uri=self.agent_id)
        super().__post_init__()
