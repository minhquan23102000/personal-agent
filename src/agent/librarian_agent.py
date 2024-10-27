from typing import Any, Dict, List, Type

from mirascope.core import BaseDynamicConfig, BaseTool, Messages, litellm

from src.agent.base_agent import BaseAgent
from src.memory import MemoryManager
from dataclasses import dataclass


class SearchLibrary(BaseTool):
    query: str

    async def call(self) -> str:
        """Simulate searching a library database."""
        # In a real implementation, this would query a database
        return "Found: The Name of the Wind by Patrick Rothfuss"


@dataclass
class LibrarianAgent(BaseAgent):
    """An agent that helps users find and recommend books."""

    system_prompt: str = "You are a helpful librarian assistant."
    model_name: str = "gemini/gemini-1.5-flash-002"
    agent_id: str = "librarian_9000"

    def __post_init__(self):
        super().__post_init__()
        self.memory_manager = MemoryManager(db_uri=self.agent_id)

    def get_tools(self) -> List[Type[BaseTool]]:
        return [SearchLibrary]

    def build_prompt(self, query: str) -> Messages.Type:
        # Customize prompt building if needed
        return super().build_prompt(query)

    @litellm.call("gemini/gemini-1.5-flash-002", call_params={"temperature": 0.5})
    async def _defalt_llm_call(self, query: str) -> BaseDynamicConfig | Dict[str, Any]:
        return self._build_call_config(query)


# Usage
async def main():
    agent = LibrarianAgent(max_history=10)
    await agent.run()
