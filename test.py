from rich import print

from mirascope.core import BaseMessageParam, gemini, prompt_template

# UPDATED
history: list[gemini.GeminiMessageParam] = [
    BaseMessageParam(role="user", content="Hello, AI"),
    BaseMessageParam(role="assistant", content="Hello, user!"),
    BaseMessageParam(role="user", content="What is the weather in Tokyo"),
]

# INCORRECT (Gemini needs to have alternating user / assistant message chain, with an optional initial (pseudo) system message)
#  history: list[BaseMessageParam | gemini.GeminiCallResponse] = [
#      BaseMessageParam(role="user", content="Hello, AI!"),
#      BaseMessageParam(role="agent 001", content="Hello, user!"),
#      BaseMessageParam(role="user", content="What is the weather in Tokyo?"),
#      BaseMessageParam(
#          role="agent 001", content="User asked about the weather in Tokyo."
#      ),
#  ]


def get_weather_in_tokyo(city: str) -> str:
    """Get the weather in a city"""
    return f"The weather in {city} is 20 degrees Celsius."


@litellm.call(model="gemini-1.5-flash-002", tools=[get_weather_in_tokyo])
@prompt_template(
    """
    MESSAGES: {history}
    """
)
def test(history): ...


response = test(history)
history.append(response.message_param)

if tool := response.tool:
    # UPDATED
    tool_response = response.tool_message_params([(tool, tool.call())])
    history += tool_response
    # INCORRECT (`tool_message_params` returns a list)
    # history.append(response.tool_message_params([(response.tool, tool_output)]))


new_response = test(history)
print(new_response.content)
history.append(new_response.message_param)

print(history)

print(tool_response)
print(tool_response.content)
