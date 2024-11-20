import asyncio
from typing import List, Optional, Annotated
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
from mirascope.core import BaseTool
from pydantic import Field, ConfigDict, model_validator, AfterValidator


def validate_urls_count(urls: List[str]) -> List[str]:
    if not 1 <= len(urls) <= 5:
        raise ValueError("Number of URLs must be between 1 and 5")
    return urls


class WebReaderTool(BaseTool):
    """This tool allows you to read and extract content from multiple web pages. After conducting a web search, you can use it to review the details of the pages you find."""

    urls: List[str] = Field(
        ...,
        description="List of URLs to read content from. Minimum is 1 URL, Maximum is 5 URLs.",
    )

    def __clean_text(self, text: str) -> str:
        """Clean and format extracted text"""
        return " ".join(text.split())

    def __extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from webpage while filtering out navigation, ads, etc"""
        for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
            tag.decompose()
        return self.__clean_text(soup.get_text(separator=" ", strip=True))

    def __validate_url(self, url: str) -> bool:
        """Validate if string is a proper URL"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False

    async def __read_single_url(self, url: str) -> str:
        """Process a single URL and return formatted content"""
        if not self.__validate_url(url):
            return f"URL: {url}\nError: Invalid URL format"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.title.string if soup.title else ""
            content = self.__extract_main_content(soup)

            return f"URL: {url}\nTitle: {title}\nContent: {content}"

        except requests.RequestException as e:
            return f"URL: {url}\nError: Failed to fetch URL: {str(e)}"

    async def call(self) -> str:
        """Process multiple URLs and return their contents in a formatted string"""
        results = await asyncio.gather(
            *[self.__read_single_url(url) for url in self.urls]
        )
        return "\n\n---\n\n".join(results)
