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
)
from tenacity import retry, stop_after_attempt, wait_exponential
from mirascope.retries.tenacity import collect_errors
from pydantic import ValidationError

from src.core.prompt.error_prompt import format_error_message

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


class ReasoningAction(pydantic.BaseModel):
    """Thought and action reasoning."""

    feeling: str = pydantic.Field(
        description="Your feeling about the current situation, obervation. Using short keywords to describe your feeling, format in uppercase. If you don't have any feeling, just leave this empty string."
    )
    thought: str = pydantic.Field(
        description="Your thought or plan for the current situation, obervation."
    )
    goal_completed: bool = pydantic.Field(
        description="True if you have completed the final goal or a milestone goal.",
    )
    got_final_answer: bool = pydantic.Field(
        description="True if you have got the answer for the user's question.",
    )
    talk_to_user: bool = pydantic.Field(
        description="Do you need to talk to user in this action? True if you do.",
    )
    action: str = pydantic.Field(
        description="From thought and feeling, identify the most effective actions to take in this very moment, not the plan in future. It is possible to combine multiple actions, but be mindful that some may not be feasible or easy to execute simultaneously, unless you are confident. Include what tools you need to use to do the action if applicable."
    )
    talk_to_user_flag_2: bool = pydantic.Field(
        description="Reconfirmation do you need to talk to user in this action? True if you do.",
    )


@dataclass
class ReactEngine:
    """Handles the react loop logic for agents."""

    model_name: str = "gemini/gemini-1.5-flash-002"
    max_retries: int = 3
    max_thought_deep: int = 3

    def format_reasoning_response(
        self, agent: "BaseAgent", response: ReasoningAction
    ) -> str:
        """Format the reasoning response."""
        how_to_address_user = (
            agent.context_memory.how_to_address_user if agent.context_memory else "user"
        )

        return inspect.cleandoc(
            f"""
            # Agent's note:
            * Feeling: {response.feeling}
            * Thought: {response.thought}
            * Do I reach the final goal or have the final answer: {"Yes" if response.goal_completed or response.got_final_answer else "No"}
            * Talk to {how_to_address_user} in this action: {"Yes" if response.talk_to_user or response.talk_to_user_flag_2 else "No"}
            * Action: {response.action}
            """
        )

    @prompt_template(
        """
        MESSAGES: 
        {messages}
        
        USER:
        Reflect on the previous message or output, noting thoughts and feelings regarding the current situation and observations. Subsequently, determine the appropriate actions to take. This is your internal thought, not for user, so write it as if you are taking note for yourself, so the note should be concise and clear as possible.
        """
    )
    def _prompt_template(self, messages: list[BaseMessageParam]): ...

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        after=collect_errors(ValidationError),
    )
    @litellm.call(
        model=model_name,
        response_model=ReasoningAction,
        json_mode=True,
    )
    async def _reasoning_step(
        self,
        messages: list[BaseMessageParam],
    ):
        """Reasoning about the action to take."""
        prompt = self._prompt_template(messages)
        return prompt

    @retry(
        stop=stop_after_attempt(max_retries),
        after=collect_errors(ValidationError),
    )
    async def _action_step(
        self,
        agent: "BaseAgent",
        reasoning_response: ReasoningAction,
        *,
        errors: list[ValidationError] | None = None,
    ) -> bool:
        """Execute the action."""

        if errors:
            correct_action_query = Messages.User(
                inspect.cleandoc(
                    f"""
                !! This is an remind auto message !! 
                There are some error in the last step when indicate that you pass the wrong parameters when use tool. 
                Error: <> {format_error_message(errors)} </>."
                Correct the parameters indicated in the error message and re-execute the action: {reasoning_response.action} immediately without further discussion.
                """
                )
            )
            agent.history.append(correct_action_query)
            agent.interface.print_user_message(correct_action_query)

        # action call
        action_response = await agent._default_call()  # type: ignore
        await agent._assitant_turn_message(action_response.message_param)
        agent.interface.print_agent_message(action_response.content)

        # tool call
        if action_response.tools:
            tool_output_messages = await agent._process_tools(
                action_response.tools, action_response
            )
            agent.history.extend(tool_output_messages)  # type: ignore

        # condtion to break the loop
        if (
            reasoning_response.goal_completed
            or reasoning_response.talk_to_user
            or reasoning_response.talk_to_user_flag_2
            or reasoning_response.got_final_answer
        ):
            return True

        return False

    async def run(self, agent: "BaseAgent"):
        """The main loop of the react engine."""
        i = 0
        while i < self.max_thought_deep:
            # reasoning step
            reasoning_response = await self._reasoning_step(agent._build_prompt())  # type: ignore
            reasoning_message = self.format_reasoning_response(
                agent, reasoning_response
            )
            agent.interface.print_agent_message(reasoning_message)

            agent_reasoning_message = Messages.Assistant(reasoning_message)
            agent.history.append(agent_reasoning_message)

            # action step
            action_break = await self._action_step(agent, reasoning_response)

            if action_break:
                break
            i += 1
