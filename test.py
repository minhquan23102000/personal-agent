from mirascope.core import litellm, BaseMessageParam, prompt_template, gemini
from rich import print

history: list[BaseMessageParam | gemini.GeminiCallResponse] = [
    BaseMessageParam(role="user", content="Hello, AI!"),
    BaseMessageParam(role="agent 001", content="Hello, user!"),
    BaseMessageParam(role="user", content="What is the weather in Tokyo?"),
    BaseMessageParam(
        role="agent 001", content="User asked about the weather in Tokyo."
    ),
]


def get_weather_in_tokyo(city: str) -> str:
    """Get the weather in a city"""
    return f"The weather in {city} is 20 degrees Celsius."


@gemini.call(model="gemini-1.5-flash-002", tools=[get_weather_in_tokyo])
@prompt_template(
    """
    MESSAGES: {history}
    """
)
def test(history): ...


response = test(history)

print(response.content)

history.append(response.message_param)

if response.tool:
    tool_output = response.tool.call()
    history.append(response.tool_message_params([(response.tool, tool_output)]))

    print(history)


new_response = test(history)
print(new_response.content)
