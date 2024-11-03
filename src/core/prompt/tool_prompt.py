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
        return {"tool_name": function_name, "tool_description": func.__doc__.strip()}
    return {"tool_name": function_name, "tool_description": "No instruction present."}


def parse_docstring_mirascope_tool(tool: BaseTool) -> dict:

    return {"tool_name": tool._name(), "tool_description": tool._description()}


PROMPT_TOOL_TEMPLATE = """
Tool: {tool_name}

Description: 
{tool_description}  
"""


def clean_args_in_tool_description(tool_description: str) -> str:
    # return tool_description.split("Args:")[0].strip()

    return tool_description.replace("self: self.", "").strip()


def build_prompt_from_tool(tool: BaseTool | Callable) -> str:
    if isinstance(tool, BaseTool):
        tool_info = parse_docstring_mirascope_tool(tool)
    else:
        tool_info = parse_docstring_tool_func(tool)

    tool_info["tool_description"] = clean_args_in_tool_description(
        tool_info["tool_description"]
    )
    return PROMPT_TOOL_TEMPLATE.format(**tool_info)


def get_list_tools_name(tools: list[BaseTool | Callable]) -> list[str]:
    rs = []
    for tool in tools:
        if isinstance(tool, BaseTool):
            rs.append(tool._name())
        else:
            rs.append(tool.__name__)
    return rs


def build_prompt_from_list_tools(tools: list[BaseTool | Callable]) -> str:
    return "\n---\n".join([build_prompt_from_tool(tool) for tool in tools])
