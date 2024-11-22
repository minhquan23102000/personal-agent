from mirascope.core import (
    openai,
    BaseTool,
    prompt_template,
    Messages,
    BaseMessageParam,
    BaseDynamicConfig,
    litellm,
)
from pydantic import BaseModel, Field
from typing import cast


class ReAct(BaseModel):

    thought: str
    action: str
    have_final_answer: bool


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


class AuthorProfile(BaseTool):
    """Returns the profile of the author with the given name."""

    name: str = Field(..., description="The name of the author.")

    def call(self) -> str:
        return f"Author {self.name} has written many books. He was born in 1977."


@litellm.call(
    "gemini/gemini-1.5-flash-002",
    response_model=ReAct,
    tools=[GetBookAuthor, AuthorProfile],
    json_mode=True,
)
@prompt_template(
    """
    SYSTEM: You are a helpful assistant that can answer questions about books.
    Provide a thought based on the observation.
    Determine the most optimal action to take. 
    If you have the final answer in this step, set have_final_answer to True.
    
    Available tools: 
    - GetBookAuthor: Returns the author of the book with the given title.
    - AuthorProfile: Returns the profile of the author with the given name.
    
    MESSAGES: {history}
    """
)
def reasoning_call(history: list[BaseMessageParam]) -> BaseDynamicConfig:

    return {"computed_fields": {"history": history}}


@litellm.call(
    "gemini/gemini-1.5-flash-002",
)
@prompt_template(
    """
    SYSTEM: You are a helpful assistant that can answer questions about books.
    
    MESSAGES: {history}
    """
)
def final_call(history: list[BaseMessageParam]): ...


text = "Can you tell me more about the author of the book 'The Name of the Wind'?"
history = []

history.append(Messages.User(content=text))
while True:

    ai_response = reasoning_call(history)

    history.append(Messages.Assistant(content=str(ai_response)))

    print(ai_response)

    response = cast(litellm.LiteLLMCallResponse, ai_response._response)

    if tool := response.tool:
        tools_and_outputs = [(tool, tool.call())]
        history += response.tool_message_params(tools_and_outputs)

        print(tools_and_outputs)

    if ai_response.have_final_answer:
        break


# final call
print(final_call(history))
