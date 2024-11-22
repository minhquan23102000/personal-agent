from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union, Any
import json
from PIL import Image, UnidentifiedImageError
import os
from mirascope.core import BaseToolKit, toolkit_tool
from docling.document_converter import DocumentConverter, InputFormat
from docling.datamodel.document import ConversionResult
import fnmatch
import subprocess


@dataclass
class GitignoreParser:
    """Parser for .gitignore patterns"""

    patterns: List[str] = field(default_factory=list)

    def load_gitignore(self, dir_path: Path) -> None:
        """Load patterns from .gitignore file if it exists"""
        gitignore_path = dir_path / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path, "r") as f:
                self.patterns = [
                    line.strip()
                    for line in f.readlines()
                    if line.strip() and not line.startswith("#")
                ]

    def is_ignored(self, path: Path, relative_to: Path) -> bool:
        """Check if a path matches any gitignore pattern"""
        rel_path = str(path.relative_to(relative_to))
        for pattern in self.patterns:
            if fnmatch.fnmatch(rel_path, pattern):
                return True
            if fnmatch.fnmatch(path.name, pattern):
                return True
        return False


class FileManagerToolkit(BaseToolKit):
    """A toolkit for managing file operations including reading, writing, and navigation.

    This toolkit provides a comprehensive suite of tools for file management operations,
    supporting various file formats including text, JSON, PDF, Office documents, and images.
    """

    __namespace__ = "file_system"

    tree_command: str = "C:/cygwin64/bin/tree.exe"
    doc_converter: DocumentConverter = DocumentConverter(
        allowed_formats=[
            InputFormat.PDF,
            InputFormat.DOCX,
            InputFormat.HTML,
            InputFormat.PPTX,
            InputFormat.MD,
            InputFormat.ASCIIDOC,
            InputFormat.IMAGE,
        ]
    )

    @toolkit_tool
    def read_file(self, file_path: str) -> str:
        """Read and return the contents of a file as a string.

        Args:
            file_path: Relative path to the text file from base_path
        """
        full_path = Path(file_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    @toolkit_tool
    def write_file(self, file_path: str, content: str) -> str:
        """Write text content to a file.

        Args:
            self: self.
            file_path: Relative path where the file should be written
            content: The text content to write
        """
        full_path = Path(file_path)

        try:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            # Convert all newlines to system's native newline character
            normalized_content = os.linesep.join(content.splitlines())
            with open(full_path, "w", encoding="utf-8") as f:
                f.write(normalized_content)
                # Ensure file ends with a newline
                if normalized_content and not normalized_content.endswith(os.linesep):
                    f.write(os.linesep)
            return f"Successfully wrote content to {file_path}"
        except Exception as e:
            return f"Error writing file: {str(e)}"

    @toolkit_tool
    def read_document(self, file_path: str) -> str:
        """Read and extract text from various document formats (PDF, DOCX, HTML, etc.).

        Args:
            file_path: Relative local path or url to the file document
        """
        full_path = file_path
        try:
            result: ConversionResult = self.doc_converter.convert(str(full_path))
            return result.document.export_to_markdown()
        except Exception as e:
            return f"Error processing document: {str(e)}"

    @toolkit_tool
    def print_folder_structure(self, dir_path: str) -> str:
        """See files structure of a directory"""
        full_path = Path(dir_path)
        gitignore_parser = GitignoreParser()
        gitignore_parser.load_gitignore(full_path)

        def build_tree(path: Path, prefix: str = "") -> str:
            """Recursively build a tree structure as a string."""
            if path.name == ".git":
                return ""  # Skip the .git directory

            tree_str = (
                f"{prefix}{path.name}/\n" if path.is_dir() else f"{prefix}{path.name}\n"
            )
            if path.is_dir():
                children = [
                    build_tree(child, prefix + "    ")
                    for child in sorted(path.iterdir())
                    if not gitignore_parser.is_ignored(child, full_path)
                ]
                tree_str += "".join(children)
            return tree_str

        try:
            tree_structure = build_tree(full_path)
            return tree_structure if tree_structure else "Empty directory"
        except Exception as e:
            return f"Error listing directory: {str(e)}"

    @toolkit_tool
    def create_directory(self, dir_path: str) -> str:
        """Create a new directory.

        Args:
            dir_path: Relative path for the new directory

        Returns:
            Success message or error description
        """
        full_path = Path(dir_path)
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            return f"Successfully created directory: {dir_path}"
        except Exception as e:
            return f"Error creating directory: {str(e)}"

    def read_image(self, file_path: str) -> Image.Image | str:
        """Read an Image file

        Args:
            file_path: Relative path to the image file
        """
        full_path = file_path
        try:
            return Image.open(full_path)
        except (FileNotFoundError, UnidentifiedImageError, Exception) as e:
            return f"Error reading image: {str(e)}"

    @toolkit_tool
    def change_working_directory(self, dir_path: str) -> str:
        """Change the current working directory of the toolkit.

        Args:
            dir_path: Path to set as the new working directory. Can be absolute or relative to current base_path
        """
        try:
            new_path = Path(dir_path)
            if not new_path.is_absolute():
                new_path = new_path.resolve()

            if not new_path.exists():
                return f"Error: Directory '{new_path}' does not exist"
            if not new_path.is_dir():
                return f"Error: '{new_path}' is not a directory"

            os.chdir(new_path)

            return f"Successfully changed working directory to: {new_path}"
        except Exception as e:
            return f"Error changing directory: {str(e)}"
