import asyncio
from dataclasses import dataclass, field

from pathlib import Path
from pyrootutils import setup_root


setup_root(".", dotenv=True, pythonpath=True, cwd=True)

print(Path.cwd())


# ------------------------------------------------------------

from src.agent.base_agent import BaseAgent
from src.memory import MemoryManager

from src.config import GOOGLE_API_KEY_LIST
from src.core.tools.search import (
    WikipediaSearchContentTool,
    WikipediaSearchRelatedArticleTool,
    DuckDuckGoSearchTool,
    WebReaderTool,
)
from src.core.reasoning.react import ReactEngine

SYSTEM_PROMPT_PATH = Path(__file__).parent / "inital_system_prompt.md"


@dataclass
class Delta3000(BaseAgent):
    """An agent that helps users find and recommend books."""

    system_prompt: str = SYSTEM_PROMPT_PATH.read_text()
    default_model: str = "gemini/gemini-1.5-flash-002"
    reflection_model: str = "gemini/gemini-1.5-pro-002"
    agent_id: str = "Delta 3000"
    api_key_env_var: str = "GEMINI_API_KEY"
    api_keys: list[str] = field(default_factory=lambda: GOOGLE_API_KEY_LIST)

    def __post_init__(self):
        self.memory_manager = MemoryManager(db_uri=self.agent_id)
        self.reasoning_engine = ReactEngine(
            max_retries=self.max_retries,
            model_name=self.default_model,
            max_deep=5,
        )
        super().__post_init__()
        self.add_tools(
            [
                DuckDuckGoSearchTool,
                WebReaderTool,
                WikipediaSearchRelatedArticleTool,
                WikipediaSearchContentTool,
            ]
        )


if __name__ == "__main__":

    agent = Delta3000()
    asyncio.run(agent.run())
