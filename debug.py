from src.agent.debug_agent import DebugAgent
import asyncio

from src.memory.memory_toolkit.static_flow.conversation_summary import (
    generate_conversation_summary,
)
from rich import print


async def main():
    agent = DebugAgent()
    await agent.run()


if __name__ == "__main__":

    # asyncio.run(agent.run())
    asyncio.run(main())
