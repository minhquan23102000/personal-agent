from dataclasses import dataclass
from typing import TYPE_CHECKING
import inspect
import pydantic
from mirascope.core import (
    BaseDynamicConfig,
    BaseMessageParam,
    Messages,
    prompt_template,
    litellm,
    gemini,
)
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import ValidationError

from src.core.prompt.error_prompt import format_error_message
from src.core.prompt.tool_prompt import get_list_tools_name
from loguru import logger
import json

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


class ReasoningAction(pydantic.BaseModel):
    """Thought and action reasoning."""

    feeling: str = pydantic.Field(
        description="Describe your feelings using short keywords, format in uppercase. If none, leave it blank."
    )
    thought: str = pydantic.Field(
        description="Provide your analysis and thought of the current situation, highlighting key observations and insights that reflect the context and implications."
    )
    goal_completed: bool = pydantic.Field(
        description="True if you have completed the final goal or a milestone goal.",
    )
    got_final_answer: bool = pydantic.Field(
        description="True if you have got the answer for the user's question.",
    )
    talk_to_user: bool = pydantic.Field(
        description="Talk to user in this action? True if you do.",
    )
    action: str = pydantic.Field(
        description="Determine the most significant action you can take immediately based on your current thoughts and feelings. Identify any tools or resources required for this action. Include only one action at a time to ensure clarity and effectiveness. Do not add plan here."
    )
    talk_to_user_flag_2: bool = pydantic.Field(
        description="Talk to user in this action? True if you do.",
    )


def parse_reasoning_response(
    response: gemini.GeminiCallResponse,
) -> ReasoningAction | str:
    json_start = response.content.index("{")
    json_end = response.content.rfind("}")
    json_content = response.content[json_start : json_end + 1]

    json_response = json.loads(json_content)

    try:
        return ReasoningAction(**json_response)
    except ValidationError:
        return response.content


@dataclass
class ReactEngineV2:
    """Handles the react loop logic for agents."""

    model_name: str = "gemini-1.5-flash-002"
    max_retries: int = 3
    max_deep: int = 3

    def format_reasoning_response(
        self, agent: "BaseAgent", response: ReasoningAction
    ) -> str:
        """Format the reasoning response."""
        how_to_address_user = (
            agent.context_memory.how_to_address_user if agent.context_memory else "user"
        )

        return inspect.cleandoc(
            f"""
            * Feeling: {response.feeling}
            * Thought: {response.thought}
            * Action: {response.action}
            """
        )

    @prompt_template(
        """
        MESSAGES: 
        {messages}
        
        USER:
        Available tools names:
        {tools_names:list}
        
        Reflect on the previous message or output, noting thoughts and feelings regarding the current situation and observations. Subsequently, determine the appropriate actions to take. This is your internal thought, not for user, so write it as if you are taking note for yourself, ensure the note content are concise and clear as possible.
        
        Ensure response is in JSON format with following fields:
        - feeling: Describe your feelings using short keywords, format in uppercase. If none, leave it blank.
        - thought: Provide your analysis and thought of the current situation, highlighting key observations and insights that reflect the context and implications.
        - goal_completed: True if you have completed the final goal or a milestone goal.
        - got_final_answer: True if you have got the answer for the user's question.
        - talk_to_user: True if you need to talk to user in this action.
        - action: Determine the most significant action you can take immediately based on your current thoughts and feelings. Identify any tools or resources required for this action. Include only one action at a time to ensure clarity and effectiveness. 
        - talk_to_user_flag_2: True if you need to talk to user in this action.
        """
    )
    def _prompt_template(self, messages: list[BaseMessageParam], tools: list): ...

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        after=collect_errors(ValidationError),
    )
    @litellm.call(model=model_name, json_mode=True)
    async def _call(
        self,
        agent: "BaseAgent",
    ):
        """Reasoning about the action to take."""
        return {
            "messages": agent._build_prompt(),
            "tools": agent.get_tools(),
            "computed_fields": {"tools_names": get_list_tools_name(agent.get_tools())},
        }

    # @retry(
    #     stop=stop_after_attempt(max_retries),
    #     wait=wait_exponential(multiplier=1, min=4, max=30),
    #     after=collect_errors(ValidationError),
    # )
    async def _step(
        self,
        agent: "BaseAgent",
        *,
        errors: list[ValidationError] | None = None,
    ) -> dict:
        """Execute the action."""
        # reasoning step
        reasoning_response_original = await self._call(agent)  # type: ignore
        agent.history.append(reasoning_response_original.message_param)

        reasoning_response = parse_reasoning_response(reasoning_response_original)
        # reasoning_response = reasoning_response_original

        if isinstance(reasoning_response, str):
            reasoning_message = reasoning_response
            agent.interface.print_agent_message(reasoning_message)
            return {
                "break": True,
                "reasoning_response": reasoning_response,
                "use_tool_call": False,
            }

        reasoning_message = self.format_reasoning_response(agent, reasoning_response)
        agent.interface.print_agent_message(reasoning_message)

        # tool call
        if reasoning_response_original.tools:
            tool_output_messages = await agent._process_tools(
                reasoning_response_original.tools, reasoning_response_original
            )  # type: ignore
            # store message to history before tool, make sure the tool call is successful before add to history
            agent.history.extend(tool_output_messages)  # type: ignore
            use_tool_call = True

        # condtion to break the loop
        if (
            reasoning_response.goal_completed
            or reasoning_response.talk_to_user
            or reasoning_response.talk_to_user_flag_2
            or reasoning_response.got_final_answer
        ):
            return {
                "break": True,
                "reasoning_response": reasoning_response,
                "use_tool_call": use_tool_call,
            }

        return {
            "break": False,
            "reasoning_response": reasoning_response,
            "use_tool_call": use_tool_call,
        }

    async def run(self, agent: "BaseAgent"):
        """The main loop of the react engine."""
        i = 0
        while i < self.max_deep:
            step_result = await self._step(agent)

            if step_result["break"]:
                break
            i += 1

        # final step
        use_tool_call = step_result["use_tool_call"]
        while use_tool_call and i < self.max_deep + 3:
            # append empty message to history

            # tool call step
            use_tool_call, response = await agent._default_step()
            i += 1
