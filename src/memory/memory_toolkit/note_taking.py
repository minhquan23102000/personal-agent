from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from mirascope.core import BaseToolKit, toolkit_tool
from pydantic import Field, ValidationError
from loguru import logger
from src.config import DATA_DIR
import tiktoken
import json
from datetime import datetime, timedelta
import os
from pathlib import Path


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
    """Simple memory toolkit for storing and retrieving information with decay and size limits."""

    __namespace__ = "memory_short_term"

    save_path: str
    memories: dict[str, Note] = Field(default_factory=dict)
    memory_file: Path = Field(default=None)
    tokenizer: Any = Field(default=None)

    # Configuration for memory management
    max_token_size: int = 25_000  # Maximum total tokens in short-term memory
    memory_decay_hours: int = 24 * 7  # Memories older than this will be removed
    encoding_model: str = "cl100k_base"  # OpenAI's encoding model

    def _initialize(self):
        if self.memory_file is None:
            self.memory_file = (
                Path(DATA_DIR) / self.save_path / "short_term_memory.json"
            )
        if self.tokenizer is None:
            self.tokenizer = tiktoken.get_encoding(self.encoding_model)
        self.load_memories()  # Load memories on initialization

    def _get_total_tokens(self) -> int:
        """Calculate total tokens in all memories."""
        total_tokens = 0
        for note in self.memories.values():
            total_tokens += len(self.tokenizer.encode(note.content))
        return total_tokens

    @toolkit_tool
    async def remember(self, key: str, content: str) -> str:
        """Store or update a piece of information in memory.

        Args:
            self: self.
            key: A descriptive label for this memory (e.g., 'user_name', 'task_{name}_goal').
            content: The information to remember.
        """
        self.memories[key] = Note(content=content)
        self.save_memories()

        return f"Remembered: {key}"

    def save_memories(self) -> None:
        """Save memories to disk."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)

        self._apply_memory_management()

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

    def _apply_memory_management(self):
        """Apply memory decay and token size limits."""
        current_time = datetime.now()
        decay_threshold = current_time - timedelta(hours=self.memory_decay_hours)

        # Remove old memories
        self.memories = {
            key: note
            for key, note in self.memories.items()
            if note.timestamp > decay_threshold
        }

        # If still over token limit, remove oldest memories until under limit
        while self._get_total_tokens() > self.max_token_size and self.memories:
            oldest_key = min(
                self.memories.keys(), key=lambda k: self.memories[k].timestamp
            )
            del self.memories[oldest_key]

    def format_memories(self) -> str:
        """Format all memories for inclusion in system prompt."""
        if not self.memories:
            return "No memories stored."

        formatted = "Current memories:\n"
        for key, note in self.memories.items():
            formatted += f"- {key}: {note.content}\n"
        return formatted
