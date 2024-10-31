from dataclasses import dataclass
from typing import Optional, List
import wikipedia
from pydantic import Field

from mirascope.core import BaseTool


class WikipediaSearchTool(BaseTool):
    """
    A tool for searching Wikipedia articles (a big library of knowledge about the world) and retrieving content.
    This tool can search for articles, get summaries, and retrieve full content.

    Examples:
        - Searching for "Albert Einstein" to get information about the physicist
        - Looking up "Python (programming language)" for programming-related content
        - Finding information about "Machine Learning" concepts
    """

    query: str = Field(
        ...,
        description="The search query or article title to look up on Wikipedia",
        examples=[
            "Albert Einstein",
            "Python (programming language)",
            "Machine Learning",
        ],
    )

    sentences: Optional[int] = Field(
        default=5,
        description="Number of sentences to return in the summary (default: 5)",
        examples=[3, 5, 10],
    )

    full_content: bool = Field(
        default=False,
        description="Whether to return the full article content instead of just a summary",
        examples=[True, False],
    )

    def call(self) -> str:
        try:
            # Search for the page
            search_results = wikipedia.search(self.query)

            if not search_results:
                return f"No Wikipedia articles found for '{self.query}'"

            # Get the most relevant page
            try:
                page = wikipedia.page(search_results[0], auto_suggest=False)
            except wikipedia.DisambiguationError as e:
                # If we hit a disambiguation page, take the first option
                page = wikipedia.page(e.options[0], auto_suggest=False)

            if self.full_content:
                return page.content

            # Get the summary with specified number of sentences
            return wikipedia.summary(
                search_results[0], sentences=self.sentences, auto_suggest=False
            )

        except wikipedia.exceptions.PageError:
            return f"Could not find a Wikipedia page for '{self.query}'"
        except Exception as e:
            return f"An error occurred while searching Wikipedia: {str(e)}"


class WikipediaRelatedTool(BaseTool):
    """
    A tool for finding related Wikipedia articles based on a search query.
    Returns a list of related article titles that can be used for further research.

    Examples:
        - Finding articles related to "Quantum Physics"
        - Discovering topics related to "Artificial Intelligence"
        - Exploring subjects connected to "World War II"
    """

    query: str = Field(
        ...,
        description="The search query to find related Wikipedia articles",
        examples=["Quantum Physics", "Artificial Intelligence", "World War II"],
    )

    max_results: int = Field(
        default=5,
        description="Maximum number of related articles to return",
        examples=[5, 10, 15],
    )

    def call(self) -> str:
        try:
            # Search for related articles
            related_articles = wikipedia.search(self.query, results=self.max_results)

            if not related_articles:
                return f"No related Wikipedia articles found for '{self.query}'"

            # Format the results
            result = f"Related articles to '{self.query}':\n"
            for i, article in enumerate(related_articles, 1):
                result += f"{i}. {article}\n"

            return result.strip()

        except Exception as e:
            return f"An error occurred while searching for related articles: {str(e)}"
