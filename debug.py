from src.agent.eva import Eva
import asyncio
from typing import List


from mirascope.core import BaseDynamicConfig, litellm, Messages
from pydantic import BaseModel


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

if __name__ == "__main__":
    pass
    agent = Eva()
    asyncio.run(agent.run())
