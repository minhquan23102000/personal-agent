from src.agent.debug_agent import DebugAgent
import asyncio
from typing import List


from mirascope.core import BaseDynamicConfig, litellm, Messages
from pydantic import BaseModel

from src.agent.eva import Eva
from src.memory.memory_toolkit.static_flow.conversation_summary import (
    generate_conversation_summary,
)
from rich import print


async def main():
    agent = DebugAgent()
    print(agent.get_tools())
    await agent.run()
    # agent.history = history_sample
    # summary = await generate_conversation_summary(agent)
    # print(summary)


if __name__ == "__main__":

    # asyncio.run(agent.run())
    asyncio.run(main())
