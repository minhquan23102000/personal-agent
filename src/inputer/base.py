from typing import Protocol


class BaseInputer(Protocol):
    def input(self, prompt: str, default: str = "") -> str: ...
