from typing import Any, Dict, List, Type

from mirascope.core import BaseDynamicConfig, BaseTool, Messages, litellm

from src.agent.base_agent import BaseAgent


class SearchLibrary(BaseTool):
    query: str

    async def call(self) -> str:
        """Simulate searching a library database."""
        # In a real implementation, this would query a database
        return "Found: The Name of the Wind by Patrick Rothfuss"


class LibrarianAgent(BaseAgent):
    """An agent that helps users find and recommend books."""

    system_prompt: str = "You are a helpful librarian assistant."

    def get_tools(self) -> List[Type[BaseTool]]:
        return [SearchLibrary]

    def build_prompt(self, query: str) -> Messages.Type:
        # Customize prompt building if needed
        return super().build_prompt(query)

    @litellm.call("gemini/gemini-1.5-flash-002", call_params={"temperature": 0.5})
    async def _call(self, query: str) -> BaseDynamicConfig | Dict[str, Any]:
        return self._build_config(query)


# Usage
async def main():
    agent = LibrarianAgent(max_history=10)
    await agent.run()
