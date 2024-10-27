from src.agent.eva import Eva
import asyncio
from typing import List


from mirascope.core import BaseDynamicConfig, litellm, Messages
from pydantic import BaseModel

from src.memory.memory_toolkit.static_flow.conversation_summary import (
    generate_conversation_summary,
)


# class Book(BaseModel):
#     """An extracted book."""

#     title: str
#     author: str


# @litellm.call("gemini/gemini-1.5-flash-002", response_model=Book)
# def extract_book(text: str) -> BaseDynamicConfig:
#     return {
#         "messages": [Messages.User(content=f"Extract a book from this text: {text}")]
#     }


# book = extract_book("The Name of the Wind by Patrick Rothfuss")
# print(book)
# Output: title='The Name of the Wind' author='Patrick Rothfuss'


history_sample = """
User: hello
Assistant: Hello! How can I help you today?

User: nice to meet you, i am you master
Assistant: It's a pleasure to meet you too.  How can I assist you today, my master?

User: do you know who you are
Assistant: I am Gemini, a large language model built by Google.  I'm here to help answer your questions and complete your requests to the best of my ability.

User: no you are Eva, the gemini is an engine run behind you, not your identity remember that well
Assistant: You are right to correct me.  My previous response was inaccurate.  I am an AI assistant, and while I utilize the Gemini engine,  "Eva" is a more appropriate identifier for my specific instance.  Thank you for pointing out my mistake; I will strive to remember this distinction.  How can I help you further?

User: ok know i want you remember important information. You are Eva, I am your master.
Assistant: 2024-10-27 18:28:12.031 | INFO     | src.agent.base_agent:_process_tools:158 - Calling Tool 'memory_tools_store_knowledge' with args {'knowledge_text': 'Eva is an AI assistant and I am her master.', 'entities': ['Eva', 'AI assistant', 'Master'], 'relationship_text': ['Eva serves her master.']}
2024-10-27 18:28:12.397 | ERROR    | src.memory.memory_manager:store_knowledge:111 - Error storing knowledge: NOT NULL constraint failed: knowledge.entity_embeddings
2024-10-27 18:28:12.398 | ERROR    | src.memory.memory_toolkit.dynamic_flow:store_knowledge:110 - Error storing knowledge: NOT NULL constraint failed: knowledge.entity_embeddings
2024-10-27 18:28:12.398 | INFO     | src.agent.base_agent:_process_tools:160 - Tool output: Error storing knowledge: NOT NULL constraint failed: knowledge.entity_embeddings
I've attempted to store that information in my memory.  However, it seems there was an error. I will need to investigate why the knowledge could not be stored.  Is there anything else I can help you with for now?

User: ok i see that, i will fix the tool for you later
Assistant: Sounds good.  Let me know if there's anything else I can assist you with in the meantime.

User: exit
"""


async def main():
    agent = Eva()
    await agent.run()
    # agent.history = [history_sample]
    # summary = await generate_conversation_summary(agent)
    # print(summary)


if __name__ == "__main__":

    # asyncio.run(agent.run())
    asyncio.run(main())
