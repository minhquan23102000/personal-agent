from dataclasses import dataclass, field
from typing import Any
from typing import Literal
import rich
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rpds import List


@dataclass
class ConsolePrinter:
    console: Console = field(default_factory=Console)

    def print_box(self, message: Any, title=None, style="bold blue"):
        with self.console.capture() as capture:
            self.console.print(message)
        str_output = capture.get()
        text = Text.from_markup(str_output)
        panel = Panel(text, title=title, padding=2, border_style=style)
        self.console.print(panel)

    def print_user_message(self, message: Any) -> None:
        # self.console.print("[User]", style="bold blue", justify="left")
        # self.console.print(
        #     message,
        #     style="bold green",
        #     justify="left",
        # )
        self.print_box(message, title="User", style="bold green")

    def print_agent_message(self, message: Any) -> None:
        # self.console.print("[Agent]", style="bold blue", justify="left")
        # self.console.print(message, style="bold blue", justify="left")
        self.print_box(message, title="Agent", style="bold blue")

    def print_system_message(
        self, message: Any, type: Literal["info", "warning", "error"] = "info"
    ) -> None:
        color = {
            "info": "yellow",
            "warning": "orange3",
            "error": "red",
        }[type]

        # self.console.print(
        #     f"[System {type.upper()}]", style=f"bold {color}", justify="left"
        # )
        # self.console.print(message, style=f"bold {color}", justify="left")
        self.print_box(message, title=f"System {type.upper()}", style=f"bold {color}")

    def print_history(self, history: list) -> None:
        self.console.print(history, style="bold green")


if __name__ == "__main__":
    printer = ConsolePrinter()
    printer.print_user_message("Hello, world!")
    printer.print_agent_message("Hello, world!")
    printer.print_system_message("Hello, world!", "info")
    printer.print_system_message("Hello, world!", "warning")
    printer.print_system_message("Hello, world!", "error")

    printer.print_agent_message({"role": "assistant", "content": "Hello, world!"})

    example_list = ["apple", "banana", "cherry"]
    printer.print_user_message(example_list)
