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
    You can use this tools to research topics, find answers to questions, or get information on specific people, companies, events, etc.
    """

    query: str = Field(
        ...,
        description="The search query to look up (ensure it same language as the region you want to search in)",
        examples=[
            "Latest developments in AI 2024",
            "Python programming best practices",
            "Twitter: Elon Musk",
            "Tokyo weather",
        ],
    )

    region: str = Field(
        default="wt-wt",
        description="Region for search results (e.g., 'us-en', 'uk-en', 'jp-jp', 'vn-vn', 'wt-wt' for worldwide)",
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

    async def call(self) -> List[SearchResult]:
        """
        Executes the DuckDuckGo search with the specified parameters.

        Returns:
            List[SearchResult]: A list of search results containing title, link, and snippet
        """
        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    keywords=self.query,
                    region=self.region,
                    # safesearch=self.safesearch,
                    max_results=10,
                )

                search_results = []
                for r in results:
                    result = SearchResult(
                        title=r["title"], link=r["link"], snippet=r["body"]
                    )
                    search_results.append(result)

                logger.debug(
                    f"Found {len(search_results)} results for query: {self.query}"
                )
                return search_results

        except Exception as e:
            logger.error(f"Error performing DuckDuckGo search: {str(e)}")
            return []

    def __str__(self) -> str:
        return f"DuckDuckGoSearchTool(query='{self.query}', region='{self.region}')"
