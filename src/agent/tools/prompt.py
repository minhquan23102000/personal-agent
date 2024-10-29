from mirascope.core import BaseTool
from typing import Callable


def parse_docstring_tool_func(func: Callable) -> dict:
    """Parse the docstring of a given function and return its name.

    Args:
        func (Callable): The function whose docstring is to be parsed.

    Returns:
        dict: A dictionary with the function name as the key and the docstring as the value.
    """
    function_name = func.__name__
    if func.__doc__:
        return {function_name: func.__doc__.strip()}
    return {function_name: "No instruction present."}


def parse_docstring_mirascope_tool(tool: BaseTool) -> dict:

    return {tool._name(): tool._description()}


PROMPT_TOOL_TEMPLATE = """
Tool: {tool_name}
Tool Description: 
{tool_description}
""".strip()


def build_prompt_from_tool(tool: BaseTool | Callable) -> str:
    if isinstance(tool, BaseTool):
        return PROMPT_TOOL_TEMPLATE.format(**parse_docstring_mirascope_tool(tool))
    return PROMPT_TOOL_TEMPLATE.format(**parse_docstring_tool_func(tool))


def get_list_tools_name(tools: list[BaseTool | Callable]) -> list[str]:
    rs = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            rs.append(tool._name())
        else:
            rs.append(tool.__name__)
    return rs


def build_prompt_from_list_tools(tools: list[BaseTool | Callable]) -> str:
    return "\n".join([build_prompt_from_tool(tool) for tool in tools])
