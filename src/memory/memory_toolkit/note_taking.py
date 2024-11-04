from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
from mirascope.core import BaseToolKit, toolkit_tool
from pydantic import Field, ValidationError
from loguru import logger
from src.config import DATA_DIR

import json


@dataclass
class NotePage:
    page_topic: str
    paragraphs: List[str] = field(default_factory=list)

    def format(self) -> str:
        """Format the note into a clean, readable format."""
        formatted_note = f"### Note page topic: {self.page_topic}\n\n"
        for idx, paragraph in enumerate(self.paragraphs, 1):
            formatted_note += f"Line {idx}: {paragraph}\n\n"
        return formatted_note


def save_notes(agent_id: str, notes: dict[str, NotePage]) -> None:
    """Save all notes to a JSON file."""
    with open(DATA_DIR / f"{agent_id}_notes.json", "w") as f:
        json.dump({topic: note.__dict__ for topic, note in notes.items()}, f, indent=4)


def load_notes(agent_id: str) -> dict[str, NotePage]:
    """Load all notes from a JSON file."""
    notes_file_path = DATA_DIR / f"{agent_id}_notes.json"
    if notes_file_path.exists():
        with open(notes_file_path, "r") as f:
            return {topic: NotePage(**note) for topic, note in json.load(f).items()}
    return {}


def format_notes(notes: dict[str, NotePage]) -> str:
    """Format all notes into a clean format suitable for system prompts.

    Returns:
        Formatted string containing all notes
    """
    if not notes:
        return "Empty"

    formatted_notes = ""
    for note in notes.values():
        formatted_notes += f"{note.format()}\n"
    return formatted_notes.strip()


class NoteTakingToolkit(BaseToolKit):
    """This is agent's note taking. Use this tool to save important information, knowledge, facts, ideas, plans, etc."""

    __namespace__ = "note_taking"

    notes: dict[str, NotePage] = field(default_factory=dict)

    @toolkit_tool
    async def add_note(self, topic: str, paragraphs: List[str]) -> str:
        """Add a new note with the given topic and paragraphs. Use this tool to save important information, knowledge, facts, ideas, plans, etc.



        Args:
            self: self.
            topic: The topic/title of the note.
            paragraphs: List of paragraph contents.
        """

        if topic in self.notes:
            self.notes[topic].paragraphs.extend(paragraphs)
        else:
            self.notes[topic] = NotePage(page_topic=topic, paragraphs=paragraphs)

        logger.debug(format_notes(self.notes))

        return f"Successfully added note: {topic}"

    @toolkit_tool
    async def update_note(self, topic: str, line_number: int, new_content: str) -> str:
        """Update a specific paragraph in an existing note. Use this tool to update the outdated content of a specific paragraph in an existing note.

        Args:
            self: self.
            topic: The topic of the note to update.
            line_number: The 1-based index of the paragraph to update.
            new_content: The new content for the paragraph.
        """
        if topic not in self.notes:
            return f"Note with topic '{topic}' not found"

        note = self.notes[topic]
        if not (1 <= line_number <= len(note.paragraphs)):
            return f"Invalid line number. Note has {len(note.paragraphs)} paragraphs"

        note.paragraphs[line_number - 1] = new_content
        logger.debug(format_notes(self.notes))

        return f"Successfully updated paragraph {line_number} in note: {topic}"

    @toolkit_tool
    async def delete_note_topic(self, topic: str) -> str:
        """Delete an entire note with all its paragraphs. Use this tool to remove a complete note topic.

        Args:
            topic: The topic of the note to delete.
        """
        if topic not in self.notes:
            return f"Note with topic '{topic}' not found"

        del self.notes[topic]
        logger.debug(format_notes(self.notes))
        return f"Successfully deleted note: {topic}"

    @toolkit_tool
    async def delete_note_lines(self, topic: str, line_numbers: List[int]) -> str:
        """Delete specific paragraphs from a note. Use this tool to remove one or more paragraphs from an existing note.

        Args:
            self: self.
            topic: The topic of the note.
            line_numbers: List of 1-based line numbers to delete.
        """
        if topic not in self.notes:
            return f"Note with topic '{topic}' not found"

        note = self.notes[topic]
        max_lines = len(note.paragraphs)

        # Validate line numbers
        invalid_lines = [ln for ln in line_numbers if not (1 <= ln <= max_lines)]
        if invalid_lines:
            return (
                f"Invalid line numbers {invalid_lines}. Note has {max_lines} paragraphs"
            )

        # Sort in reverse order to avoid index shifting
        for line_num in sorted(line_numbers, reverse=True):
            note.paragraphs.pop(line_num - 1)

        logger.debug(format_notes(self.notes))
        return f"Successfully deleted paragraphs {line_numbers} from note: {topic}"


async def main():
    note_taking_toolkit = NoteTakingToolkit()
    print(await note_taking_toolkit.add_note("test", ["test1", "test2"]))
    await note_taking_toolkit.add_note("test2", ["test3", "test4"])

    await note_taking_toolkit.update_note("test", 1, "test5")
    print(await note_taking_toolkit.delete_note_lines("test", [1]))
    print(await note_taking_toolkit.delete_note_topic("test2"))

    # save notes to file
    save_notes("test note", note_taking_toolkit.notes)

    # load notes
    print(load_notes("test note"))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
