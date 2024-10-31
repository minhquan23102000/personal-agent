from typing import List, Literal, Any
from src.log.base import BasePrinter


class MultiPrinter(BasePrinter):
    """A printer that prints to multiple printers."""

    def __init__(self, printers: List[BasePrinter]):
        self.printers = printers

    def print_system_message(
        self, message: Any, type: Literal["info", "warning", "error"] = "info"
    ) -> None:
        for printer in self.printers:
            printer.print_system_message(message, type)

    def print_user_message(self, message: Any) -> None:
        for printer in self.printers:
            printer.print_user_message(message)

    def print_agent_message(self, message: Any) -> None:
        for printer in self.printers:
            printer.print_agent_message(message)
