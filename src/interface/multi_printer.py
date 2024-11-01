from typing import List, Literal, Any
from src.interface.base import BaseInterface
from dataclasses import dataclass


@dataclass
class MultiPrinter:
    """When you want to use multiple interfaces at the same time."""

    interface_input: BaseInterface
    printers: List[BaseInterface]

    def print_system_message(
        self, message: str, type: Literal["info", "warning", "error"] = "info"
    ) -> None:
        for printer in self.printers:
            printer.print_system_message(message, type)

    def print_user_message(self, message: Any) -> None:
        for printer in self.printers:
            printer.print_user_message(message)

    def print_agent_message(self, message: Any) -> None:
        for printer in self.printers:
            printer.print_agent_message(message)

    def print_history(self, history: list) -> None:
        for printer in self.printers:
            printer.print_history(history)

    def input(self, message: str) -> str:
        return self.interface_input.input(message)
