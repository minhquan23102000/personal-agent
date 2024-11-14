from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Dict
from mirascope.core import BaseToolKit, toolkit_tool
from pydantic import Field, ValidationError
from loguru import logger
from src.config import DATA_DIR

import json
from datetime import datetime
import os
from pathlib import Path

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


@dataclass
class Note:
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert note to dictionary for serialization."""
        return {"content": self.content, "timestamp": self.timestamp.isoformat()}

    @classmethod
    def from_dict(cls, data: Dict) -> "Note":
        """Create note from dictionary."""
        return cls(
            content=data["content"], timestamp=datetime.fromisoformat(data["timestamp"])
        )


class ShortTermMemoryToolKit(BaseToolKit):
    """Simple memory toolkit for storing and retrieving information."""

    __namespace__ = "memory_short_term"
    memories: dict[str, Note] = field(default_factory=dict)
    agent: "BaseAgent"

    def __init__(self, agent: "BaseAgent"):
        super().__init__()
        self.agent = agent
        self.memory_file = (
            Path(DATA_DIR) / self.agent.agent_id / "short_term_memory.json"
        )
        self.load_memories()  # Load memories on initialization

    @toolkit_tool
    async def remember(self, key: str, content: str) -> str:
        """Store or update a piece of information in memory.

        Args:
            self: self
            key: A descriptive label for this memory (e.g., 'user_name', 'task_{name}_goal')
            content: The information to remember
        """
        self.memories[key] = Note(content=content)
        self.save_memories()  # Auto-save after each new memory
        return f"Remembered: {key}"

    def save_memories(self) -> None:
        """Save memories to disk."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        memory_data = {key: note.to_dict() for key, note in self.memories.items()}

        with open(self.memory_file, "w") as f:
            json.dump(memory_data, f, indent=2)

    def load_memories(self) -> None:
        """Load memories from disk."""
        if not self.memory_file.exists():
            self.memories = {}
            return

        try:
            with open(self.memory_file, "r") as f:
                memory_data = json.load(f)

            self.memories = {
                key: Note.from_dict(data) for key, data in memory_data.items()
            }
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error loading memories: {e}")
            self.memories = {}

    def format_memories(self) -> str:
        """Format all memories for inclusion in system prompt."""
        if not self.memories:
            return "No memories stored."

        formatted = "Current memories:\n"
        for key, note in self.memories.items():
            formatted += f"- {key}: {note.content}\n"
        return formatted
