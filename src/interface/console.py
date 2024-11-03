from dataclasses import dataclass, field
from typing import Any
from typing import Literal
import pyperclip
import rich
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rpds import List
import json


@dataclass
class ConsoleInterface:
    console: Console = field(default_factory=Console)

    def print_box(self, message: Any, title=None, style="bold blue"):
        # self.console.print(f"[{title}]: ", style=style, justify="left")

        if isinstance(message, list) or isinstance(message, dict):
            try:
                message = json.dumps(message, indent=2)
                message = f"""```json\n{message}\n```"""
            except Exception as e:
                with self.console.capture() as capture:
                    self.console.print(message, style=style, justify="left")
                message = capture.get()

        content = Markdown(message)
        panel = Panel(content, title=title, style=style)
        self.console.print(panel)

    def print_user_message(self, message: Any) -> None:
        self.print_box(message, title="User", style="bold blue")

    def print_agent_message(self, message: Any) -> None:
        self.print_box(message, title="Agent", style="bold green")

    def print_system_message(
        self, message: str, type: Literal["info", "warning", "error"] = "info"
    ) -> None:
        color = {
            "info": "yellow",
            "warning": "orange3",
            "error": "red",
        }[type]

        self.print_box(message, title=f"System {type.upper()}", style=f"bold {color}")

    def print_history(self, history: list) -> None:
        self.console.print(history, style="bold green")

    def input(self, message: str) -> str:
        msg = self.console.input(message)

        if msg == "pcb":
            msg = pyperclip.paste()

        return msg


if __name__ == "__main__":
    printer = ConsoleInterface()
    printer.print_user_message("Hello, world!")
    printer.print_agent_message("Hello, world!")
    printer.print_system_message("Hello, world!", "info")
    printer.print_system_message("Hello, world!", "warning")
    printer.print_system_message("Hello, world!", "error")

    printer.print_agent_message({"role": "assistant", "content": "Hello, world!"})

    example_list = ["apple", "banana", "cherry"]
    printer.print_user_message(example_list)
