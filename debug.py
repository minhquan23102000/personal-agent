from src.agent.debug_agent import DebugAgent
import asyncio

from src.memory.memory_toolkit.static_flow.conversation_summary import (
    generate_conversation_summary,
)
from rich import print
from src.core.tools.files.file_manager_toolkit import FileManagerToolkit


async def test_file_manager_toolkit():
    toolkit = FileManagerToolkit()
    print(toolkit.list_directory("."))
    print(toolkit.read_file("README.md"))


async def main():
    agent = DebugAgent()
    await agent.run()


if __name__ == "__main__":

    # asyncio.run(agent.run())
    # asyncio.run(main())
    asyncio.run(test_file_manager_toolkit())
