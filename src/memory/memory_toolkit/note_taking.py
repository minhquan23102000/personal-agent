from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional
from mirascope.core import BaseToolKit, toolkit_tool
from pydantic import Field, ValidationError
from loguru import logger


@dataclass
class NotePage:
    page_topic: str
    paragraphs: List[str] = field(default_factory=list)

    def format(self) -> str:
        """Format the note into a clean, readable format."""
        formatted_note = f"### Note page topic: {self.page_topic}\n\n"
        for idx, paragraph in enumerate(self.paragraphs, 1):
            formatted_note += f"Line {idx}: {paragraph}\n"
        return formatted_note


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
    def add_note(self, topic: str, paragraphs: List[str]) -> str:
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
    def update_note(self, topic: str, line_number: int, new_content: str) -> str:
        """Update a specific paragraph in an existing note. Use this tool to update the content of a specific paragraph in an existing note.

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
    def delete_note_topic(self, topic: str) -> str:
        """Delete an entire note with all its paragraphs. Use this tool to remove a complete note topic.

        Args:
            topic: The topic of the note to delete.

        Returns:
            str: Success or failure message.
        """
        if topic not in self.notes:
            return f"Note with topic '{topic}' not found"

        del self.notes[topic]
        logger.debug(format_notes(self.notes))
        return f"Successfully deleted note: {topic}"

    @toolkit_tool
    def delete_note_lines(self, topic: str, line_numbers: List[int]) -> str:
        """Delete specific paragraphs from a note. Use this tool to remove one or more paragraphs from an existing note.

        Args:
            self: self.
            topic: The topic of the note.
            line_numbers: List of 1-based line numbers to delete.

        Returns:
            str: Success or failure message.
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
