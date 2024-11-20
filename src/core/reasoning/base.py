from typing import TYPE_CHECKING, Optional, Protocol
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


class BaseReasoningEngine(ABC):
    """Base class for all reasoning engines."""

    name: str
    description: str
    state_prompt: Optional[str] = None

    @abstractmethod
    async def run(self, agent: "BaseAgent") -> None: ...

    def __str__(self) -> str:
        """String representation of the engine."""
        return f"{self.name} ({self.description})"
