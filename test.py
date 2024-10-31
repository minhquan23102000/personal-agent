from src.agent.tools.search.duckduckgo_search import DuckDuckGoSearchTool
import asyncio

tool = DuckDuckGoSearchTool(query="Albert Einstein")
print(asyncio.run(tool.call()))
