from dataclasses import dataclass, field
from typing import Any, Dict, List, Type, Callable

from mirascope.core import BaseDynamicConfig, BaseTool, Messages, litellm

from src.agent.base_agent import BaseAgent
from src.memory import MemoryManager

from rich import print


def ask_user_for_help(question: str) -> str:
    """Asks user if needed."""
    print("[Assistant Needs Help]")
    print(f"[QUESTION]: {question}")
    answer = input("[ANSWER]: ")
    print("[End Help]")
    return answer


@dataclass
class Eva(BaseAgent):
    """An agent that helps users find and recommend books."""

    system_prompt: str = (
        "You are a helpful agent. Assist the user with their questions and needs."
    )
    model_name: str = "gemini/gemini-1.5-flash-002"
    agent_id: str = "Eva - 9000"

    def __post_init__(self):
        self.memory_manager = MemoryManager(db_uri=self.agent_id)
        super().__post_init__()
