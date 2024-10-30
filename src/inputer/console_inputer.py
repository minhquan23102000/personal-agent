from dataclasses import dataclass

from rich.prompt import Prompt


@dataclass
class ConsoleInputer:
    def input(self, prompt: str, default: str = "") -> str:
        return Prompt.ask(prompt, default=default)
