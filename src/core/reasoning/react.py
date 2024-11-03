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


@dataclass
class ReactEngine:
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
        {tools:list}
        
        Reflect on the previous message or output, noting thoughts and feelings regarding the current situation and observations. Subsequently, determine the appropriate actions to take. This is your internal thought, not for user, so write it as if you are taking note for yourself, ensure the note content are concise and clear as possible.
        """
    )
    def _prompt_template(self, messages: list[BaseMessageParam], tools: list): ...

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=30),
        after=collect_errors(ValidationError),
    )
    @gemini.call(
        model=model_name,
        response_model=ReasoningAction,
        json_mode=True,
    )
    async def _reasoning_step(
        self,
        messages: list[BaseMessageParam],
        tools: list,
    ):
        """Reasoning about the action to take."""
        prompt = self._prompt_template(messages, tools)
        return prompt

    async def _action_step(
        self,
        agent: "BaseAgent",
        reasoning_response: ReasoningAction,
    ) -> dict:
        """Execute the action."""
        # agent.history.append(
        #     Messages.Assistant(f"Let's execute the step by step of the action plan")
        # )
        use_tool_call, response = await agent._default_step()

        # condtion to break the loop
        if (
            reasoning_response.goal_completed
            or reasoning_response.talk_to_user
            or reasoning_response.talk_to_user_flag_2
            or reasoning_response.got_final_answer
        ):
            return {"break": True, "response": response, "use_tool_call": use_tool_call}

        return {"break": False, "response": response, "use_tool_call": use_tool_call}

    async def run(self, agent: "BaseAgent"):
        """The main loop of the react engine."""
        i = 0
        while i < self.max_deep:
            # reasoning step
            reasoning_response = await self._reasoning_step(
                agent._build_prompt(), get_list_tools_name(agent.get_tools())
            )  # type: ignore
            reasoning_message = self.format_reasoning_response(
                agent, reasoning_response
            )
            agent.interface.print_agent_message(reasoning_message)

            agent_reasoning_message = Messages.Assistant(reasoning_message)
            agent.history.append(agent_reasoning_message)

            # logger.debug(f"reasoning_response: {reasoning_response}")

            # action step
            action_result = await self._action_step(agent, reasoning_response)

            if action_result["break"]:
                break
            i += 1

        # if agent use tool call, continue to step for agent handle tool output
        use_tool_call = action_result["use_tool_call"]
        while use_tool_call and i < self.max_deep:
            # append empty message to history

            # tool call step
            use_tool_call, response = await agent._default_step()
            i += 1
