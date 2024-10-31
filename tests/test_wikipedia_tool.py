import pytest
import wikipedia

from src.agent.tools.search.wikipedia import (
    WikipediaSearchContentTool,
    WikipediaSearchRelatedArticleTool,
)


class TestWikipediaSearchTool:
    @pytest.fixture
    def wiki_tool(self):
        return WikipediaSearchContentTool(query="Albert Einstein", full_content=False)

    @pytest.mark.asyncio
    async def test_successful_search(self, wiki_tool):
        result = await wiki_tool.call()
        assert "Einstein" in result
        assert len(result.split(".")) >= 5  # At least 5 sentences

    @pytest.mark.asyncio
    async def test_full_content_retrieval(self):
        tool = WikipediaSearchContentTool(query="Albert Einstein", full_content=True)
        result = await tool.call()
        assert len(result) > 1000  # Full content should be substantial
        assert "Einstein" in result
        assert "physics" in result.lower()

    @pytest.mark.asyncio
    async def test_custom_sentence_length(self):
        tool = WikipediaSearchContentTool(query="Albert Einstein", full_content=False)
        result = await tool.call()
        assert len(result.split(".")) <= 3  # 2 sentences + possible partial

    @pytest.mark.asyncio
    async def test_nonexistent_page(self):
        tool = WikipediaSearchContentTool(
            query="ThisPageDefinitelyDoesNotExistOnWikipedia12345", full_content=False
        )
        result = await tool.call()
        assert "Could not find" in result or "No Wikipedia articles found" in result

    @pytest.mark.asyncio
    async def test_disambiguation_handling(self):
        # "Python" typically hits a disambiguation page
        tool = WikipediaSearchContentTool(query="Python", full_content=False)
        result = await tool.call()
        assert len(result) > 0
        assert isinstance(result, str)


class TestWikipediaRelatedTool:
    @pytest.fixture
    def related_tool(self):
        return WikipediaSearchRelatedArticleTool(query="Quantum Physics")

    @pytest.mark.asyncio
    async def test_successful_related_search(self, related_tool):
        result = await related_tool.call()
        assert "Related articles to 'Quantum Physics'" in result
        assert "1." in result  # Should have numbered results
        assert len(result.split("\n")) > 1  # Should have multiple results

    @pytest.mark.asyncio
    async def test_custom_max_results(self):
        tool = WikipediaSearchRelatedArticleTool(query="Quantum Physics")
        result = await tool.call()
        lines = result.split("\n")
        assert len(lines) == 4  # Title + 3 results

    @pytest.mark.asyncio
    async def test_nonexistent_topic(self):
        tool = WikipediaSearchRelatedArticleTool(
            query="ThisTopicDefinitelyDoesNotExistOnWikipedia12345"
        )
        result = await tool.call()
        assert "No related Wikipedia articles found" in result

    def test_input_validation(self):

        # Test empty query
        with pytest.raises(Exception):
            WikipediaSearchRelatedArticleTool(query="")
