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
from src.core.tools.files.file_manager_toolkit import FileManagerToolkit
from src.core.reasoning.react import ReActEngine
from src.core.reasoning.chat import ChatEngine


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
    temperature: float = 0.5

    def __post_init__(self):
        self.memory_manager = MemoryManager(
            db_uri=self.agent_id
        )  # override defaulut before super init
        super().__post_init__()

        chat_engine = ChatEngine(
            model_name=self.reflection_model, temperature=self.temperature
        )
        reasoning_engine = ReActEngine(
            model_name=self.reflection_model, max_retries=self.max_retries, max_deep=12
        )

        # register engines
        self.state_manager.register_list_engines([chat_engine, reasoning_engine])

        # set defauult chat engine
        self.state_manager.chat_config.current_engine = chat_engine

        self.add_tools(
            [
                DuckDuckGoSearchTool,
                WebReaderTool,
            ]
        )
        self.add_tools(FileManagerToolkit().create_tools())


if __name__ == "__main__":

    agent = Delta3000()
    asyncio.run(agent.run())
