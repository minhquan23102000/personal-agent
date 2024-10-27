from typing import Any, Dict, List, Type, Callable

from mirascope.core import BaseDynamicConfig, BaseTool, Messages, litellm

from src.agent.base_agent import BaseAgent
from src.memory import MemoryManager
from dataclasses import dataclass
from src.agent.memory_toolkit.dynamic_tools import DynamicMemoryToolKit
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

    system_prompt: str = "You are a helpful assistant."
    model_name: str = "gemini/gemini-1.5-flash-002"
    agent_id: str = "Eva - 9000"
    llm_call: Callable = litellm.call

    def __post_init__(self):

        self.memory_manager = MemoryManager(db_name=self.agent_id)
        super().__post_init__()

    def get_tools(self) -> List[Type[BaseTool] | Callable]:
        return [
            ask_user_for_help,
            *DynamicMemoryToolKit(memory_manager=self.memory_manager).create_tools(),
        ]

    def build_prompt(self, query: str) -> Messages.Type:
        # Customize prompt building if needed
        return super().build_prompt(query)

    @litellm.call("gemini/gemini-1.5-flash-002", call_params={"temperature": 0.5})
    async def _call(self, query: str) -> BaseDynamicConfig | Dict[str, Any]:
        return self._build_config(query)
