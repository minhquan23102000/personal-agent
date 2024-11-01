from dataclasses import dataclass
from typing import Optional, List
import wikipedia
from pydantic import Field, ConfigDict

from mirascope.core import BaseTool


class WikipediaSearchContentTool(BaseTool):
    """
    A tool for searching Wikipedia articles (a big library of knowledge about the world) and retrieving content. Only useful if you want to find facts, knowledges, or historical events.
    """

    query: str = Field(
        ...,
        description="The search query or article title to look up on Wikipedia",
    )

    async def call(self) -> str:
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

            # return page.content

            # Get the summary with specified number of sentences
            return wikipedia.summary(
                search_results[0], sentences=30, auto_suggest=False
            )

        except wikipedia.exceptions.PageError:
            return f"Could not find a Wikipedia page for '{self.query}'"
        except Exception as e:
            return f"An error occurred while searching Wikipedia: {str(e)}"


class WikipediaSearchRelatedArticleTool(BaseTool):
    """
    A tool for finding related Wikipedia articles based on a search query.
    Returns a list of related article titles that can be used for further research.
    Only useful if you want to find facts, knowledges, or historical events.
    """

    query: str = Field(
        ...,
        description="The search query to find related Wikipedia articles",
    )

    async def call(self) -> str:
        try:
            # Search for related articles
            related_articles = wikipedia.search(self.query, results=10)

            if not related_articles:
                return f"No related Wikipedia articles found for '{self.query}'"

            # Format the results
            result = f"Related articles to '{self.query}':\n"
            for i, article in enumerate(related_articles, 1):
                result += f"{i}. {article}\n"

            return result.strip()

        except Exception as e:
            return f"An error occurred while searching for related articles: {str(e)}"
