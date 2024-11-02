# Mirascrope Tools Createline Guide

## What is a tool?

Tools are user-defined functions that an LLM (Large Language Model) can ask the user to invoke on its behalf. This greatly enhances the capabilities of LLMs by enabling them to perform specific tasks, access external data, interact with other systems, and more.

Mirascope enables defining tools in a provider-agnostic way, which can be used across all supported LLM providers without modification.

## How the tool call?

When an LLM decides to use a tool, it indicates the tool name and argument values in its response. It's important to note that the LLM doesn't actually execute the function; instead, you are responsible for calling the tool and (optionally) providing the output back to the LLM in a subsequent interaction. 

## Basic Usage and Syntax 

There are two ways of defining tools in Mirascope: BaseTool and functions.

You can consider the functional definitions a shorthand form of writing the BaseTool version of the same tool. Under the hood, tools defined as functions will get converted automatically into their corresponding BaseTool.

Let's take a look at a basic example of each using Mirascope vs. official provider SDKs:
```python
from mirascope.core import BaseTool, litellm, prompt_template
from pydantic import Field


class GetBookAuthor(BaseTool):
    """Returns the author of the book with the given title."""

    title: str = Field(..., description="The title of the book.")

    def call(self) -> str:
        if self.title == "The Name of the Wind":
            return "Patrick Rothfuss"
        elif self.title == "Mistborn: The Final Empire":
            return "Brandon Sanderson"
        else:
            return "Unknown"
```

## Validation and Error Handling
Since BaseTool is a subclass of Pydantic's BaseModel, they are validated on construction, so it's important that you handle potential ValidationError's for building more robust applications:
```python
from typing import Annotated

from mirascope.core import BaseTool, litellm, prompt_template
from pydantic import AfterValidator, Field, ValidationError


def is_upper(v: str) -> str:
    assert v.isupper(), "Must be uppercase"
    return v


class GetBookAuthor(BaseTool):
    """Returns the author of the book with the given title."""

    title: Annotated[str, AfterValidator(is_upper)] = Field(
        ..., description="The title of the book."
    )

    def call(self) -> str:
        if self.title == "THE NAME OF THE WIND":
            return "Patrick Rothfuss"
        elif self.title == "MISTBORN: THE FINAL EMPIRE":
            return "Brandon Sanderson"
        else:
            return "Unknown"


@litellm.call("gpt-4o-mini", tools=[GetBookAuthor])
@prompt_template("Who wrote {book}?")
def identify_author(book: str): ...


response = identify_author("The Name of the Wind")
try:
    if tool := response.tool:
        print(tool.call())
    else:
        print(response.content)
except ValidationError as e:
    print(e)
    # > 1 validation error for GetBookAuthor
    #   title
    #     Assertion failed, Must be uppercase [type=assertion_error, input_value='The Name of the Wind', input_type=str]
    #       For further information visit https://errors.pydantic.dev/2.8/v/assertion_error
```

In this example we've added additional validation, but it's important that you still handle ValidationError's even with standard tools since they are still BaseModel instances and will validate the field types regardless.


## Toolkits

Toolkits are collections of tools that can be used to group related tools together. They are useful for organizing tools into logical groups, making it easier to manage and reuse them in different parts of the application.

The BaseToolKit class enables:

- Organiziation of a group of tools under a single namespace. This can be useful for making it clear to the LLM when to use certain tools over others. For example, you could namespace a set of tools under "file_system" to indicate that those tools are specifically for interacting with the file system.
- Dynamic tool definitions. This can be useful for generating tool definitions that are dependent on some input or state. For example, you may want to update the description of tools based on an argument of the call being made.

```python
from mirascope.core import (
    BaseDynamicConfig,
    BaseToolKit,
    litellm,
    prompt_template,
    toolkit_tool,
)


class BookTools(BaseToolKit):
    __namespace__ = "book_tools"

    reading_level: str

    @toolkit_tool
    def suggest_author(self, author: str) -> str:
        """Suggests an author for the user to read based on their reading level.

        User reading level: {self.reading_level}
        """
        return f"I would suggest you read some books by {author}"


@litellm.call("gemini/gemini-1.5-flash-002")
@prompt_template("What {genre} author should I read?")
def recommend_author(genre: str, reading_level: str) -> BaseDynamicConfig:
    toolkit = BookTools(reading_level=reading_level)
    return {"tools": toolkit.create_tools()}


response = recommend_author("fantasy", "beginner")
if tool := response.tool:
    print(tool.call())
    # Output: I would suggest you read some books by J.K. Rowling

response = recommend_author("fantasy", "advanced")
if tool := response.tool:
    print(tool.call())
    # Output: I would suggest you read some books by Brandon Sanderson

```
## Documentation Guidelines

When documenting tools, it's important to provide clear and concise descriptions that help the LLM understand how and when to use the tool. Think about the use case of the tool and how it can be used to solve a problem. Include many examples to help the LLM understand how to use the tool. Include scenarios to help the LLM understand how the tool can be used in real-world when they are on action.


## Simplexity

Tools should be designed for simplicity. Each tool should focus on a single task and perform it effectively. Avoid unnecessary complexity; instead, ensure that the agent can easily understand how to use the tool without confusion or ambiguity.


## Error Handling

Tools should be designed to handle errors gracefully. If a tool fails, return the error message back to the LLM so that the LLM can understand what went wrong and take appropriate action.
