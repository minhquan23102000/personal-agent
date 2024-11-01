from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


class BaseReasoningEngine(Protocol):
    async def run(self, agent: "BaseAgent"): ...
