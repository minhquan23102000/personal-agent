from dataclasses import dataclass, field

from src.agent.base_agent import BaseAgent
from src.memory import MemoryManager

from rich import print
from src.config import GOOGLE_API_KEY_LIST
from src.agent.tools.search import (
    WebReaderTool,
    WikipediaSearchContentTool,
    WikipediaSearchRelatedArticleTool,
    DuckDuckGoSearchTool,
)
from pathlib import Path

SYSTEM_PROMPT_PATH = Path(__file__).parent / "inital_system_prompt.md"


@dataclass
class DebugAgent(BaseAgent):
    """An agent that helps users find and recommend books."""

    system_prompt: str = SYSTEM_PROMPT_PATH.read_text()
    action_model_name: str = "gemini-1.5-flash-002"
    reasoning_model_name: str = "gemini-1.5-flash-002"
    agent_id: str = "Debug Agent 1088"
    api_key_env_var: str = "GEMINI_API_KEY"
    api_keys: list[str] = field(default_factory=lambda: GOOGLE_API_KEY_LIST)

    def __post_init__(self):
        self.memory_manager = MemoryManager(db_uri=self.agent_id)
        super().__post_init__()
        self.add_tools(
            [
                WebReaderTool,
                WikipediaSearchContentTool,
                WikipediaSearchRelatedArticleTool,
                DuckDuckGoSearchTool,
            ]
        )
