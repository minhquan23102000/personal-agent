from dataclasses import dataclass
from typing import List, Optional
from pydantic import Field, field_validator

from duckduckgo_search import DDGS

from mirascope.core import BaseTool
from loguru import logger


@dataclass
class SearchResult:
    """Data class to store search result information"""

    title: str
    link: str
    snippet: str


class DuckDuckGoSearchTool(BaseTool):
    """
    A tool for performing web searches using DuckDuckGo search engine.
    This tool is useful for finding information on the internet.
    """

    query: str = Field(
        ...,
        description="The search query to look up (ensure it same language as the region you want to search in)",
    )

    # safesearch: str = Field(
    #     default="moderate", description="SafeSearch setting: 'on', 'moderate', or 'off'"
    # )

    # @field_validator("safesearch")
    # def validate_safesearch(cls, v):
    #     allowed = ["on", "moderate", "off"]
    #     if v.lower() not in allowed:
    #         raise ValueError(f"safesearch must be one of {allowed}")
    #     return v.lower()

    def format_search_results(self, search_results: List[SearchResult]) -> str:
        """Formats the search results into a readable string."""
        formatted_results = []
        for result in search_results:
            formatted_results.append(
                f"Title: {result.title}\nLink: {result.link}\nSnippet: {result.snippet}\n"
            )
        return (
            "\n---\n".join(formatted_results)
            + "\n---\nUse the WebReaderTool to access and read the full content of the most relevant search results if needed for more comprehensive and relevant information."
        )

    async def call(self) -> str:
        """
        Executes the DuckDuckGo search with the specified parameters.

        Returns:
            str: A string of search results containing title, link, and snippet
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    keywords=self.query,
                    region="wt-wt",
                    # safesearch=self.safesearch,
                    max_results=5,
                )

                search_results = []
                for r in results:
                    result = SearchResult(
                        title=r["title"], link=r["href"], snippet=r["body"]
                    )
                    search_results.append(result)

                logger.debug(
                    f"Found {len(search_results)} results for query: {self.query}"
                )
                return self.format_search_results(search_results)

        except Exception as e:
            logger.error(f"Error performing DuckDuckGo search: {str(e)}")
            return f"Error performing DuckDuckGo search: {str(e)}"

    def __str__(self) -> str:
        return f"DuckDuckGoSearchTool(query='{self.query}')"
