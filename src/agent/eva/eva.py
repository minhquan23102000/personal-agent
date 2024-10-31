import asyncio
from dataclasses import dataclass, field

from src.agent.base_agent import BaseAgent
from src.memory import MemoryManager

from src.config import GOOGLE_API_KEY_LIST

from pathlib import Path

from pyrootutils import setup_root

setup_root(".", dotenv=True, pythonpath=True, cwd=True)

SYSTEM_PROMPT_PATH = Path(__file__).parent / "inital_system_prompt.md"


@dataclass
class Eva(BaseAgent):
    """An agent that helps users find and recommend books."""

    system_prompt: str = SYSTEM_PROMPT_PATH.read_text()
    default_model_name: str = "gemini-1.5-flash-002"
    slow_model_name: str = "gemini-1.5-pro-002"
    agent_id: str = "Eva 8080"
    api_key_env_var: str = "GEMINI_API_KEY"
    api_keys: list[str] = field(default_factory=lambda: GOOGLE_API_KEY_LIST)

    def __post_init__(self):
        self.memory_manager = MemoryManager(db_uri=self.agent_id)
        super().__post_init__()


if __name__ == "__main__":
    asyncio.run(Eva().run())
