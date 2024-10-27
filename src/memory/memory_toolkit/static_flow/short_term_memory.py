from mirascope.core import prompt_template
from pydantic import BaseModel, Field

from src.memory.memory_toolkit.static_flow.conversation_summary import (
    BaseConversationSummary,
)

from loguru import logger
from src.agent.base_agent import BaseAgent


class BaseShortTermMemoryUpdate(BaseModel):
    """Model for short-term memory updates."""

    user_info: str = Field(description="Updated user information")
    recent_goal_and_status: str = Field(description="Current goals and their status")
    important_context: str = Field(description="Key contextual information")
    agent_beliefs: str = Field(description="Updated agent beliefs")
    agent_info: str = Field(description="Agent's current information and identity")


@prompt_template(
    """
    Please analyze the conversation history and extract key information to update the agent's short-term memory.
    
    Current Short-Term Memory:
    User Info: {current_memory.user_info}
    Recent Goals: {current_memory.recent_goal_and_status}
    Important Context: {current_memory.important_context}
    Agent Beliefs: {current_memory.agent_beliefs}
    
    Conversation History:
    {self.history}
    
    Conversation Summary:
    {summary}
    
    Please provide updates for the short-term memory in the following areas:
    1. User Information: Any new information about the user's preferences, background, or needs
    2. Recent Goals: Current goals and their status
    3. Important Context: Key contextual information that should be remembered
    4. Agent Beliefs: Updated understanding about the world and user intentions
    5. Agent Info: Agent's current information and identity
    
    Focus on:
    - Most recent and relevant information
    - Clear status of ongoing tasks
    - Important context for future interactions
    - Updated beliefs based on the conversation
    - Agent's current information (personality, name, gender, language, style, age, etc.)
    """
)
def short_term_memory_prompt(history, summary, current_memory): ...


async def generate_updated_short_term_memory(
    agent: BaseAgent, summary: BaseConversationSummary
) -> BaseShortTermMemoryUpdate:
    """Get updated short-term memory state after conversation."""
    try:
        prompt = short_term_memory_prompt(
            history=agent.history,
            summary=summary,
            current_memory=agent.short_term_memory,
        )
        response = await agent._custom_llm_call(
            query=prompt,
            response_model=BaseShortTermMemoryUpdate,
            json_mode=True,
        )
        return response
    except Exception as e:
        logger.error(f"Error generating updated short-term memory: {e}")
        raise

    # Update end_conversation to include short-term memory update
    async def memory_conversation(self) -> None:
        """Handle all end of conversation tasks."""
        logger.info("Starting end of conversation memory...")
        try:
            if not self.memory_manager:
                logger.warning(
                    "No memory manager available - skipping conversation end tasks"
                )
                return

            # 1. Generate conversation summary and reflection
            summary_response = await self.generate_conversation_summary()
            reflection_response = await self.perform_self_reflection()

            self.current_conversation_summary = summary_response

            # 2. Store conversation summary and improvements
            await self.memory_manager.store_conversation_summary(
                conversation_id=self.conversation_id,
                prompt=self.system_prompt,
                conversation_summary=str(summary_response),
                improve_prompt=reflection_response.improved_prompt,
                reward_score=reflection_response.reward_score,
                feedback_text="\n".join(
                    [
                        "Strengths:",
                        *[f"- {s}" for s in reflection_response.strengths],
                        "\nAreas for Improvement:",
                        *[f"- {a}" for a in reflection_response.areas_for_improvement],
                        "\nSpecific Examples:",
                        *[f"- {e}" for e in reflection_response.specific_examples],
                    ]
                ),
                improvement_suggestion="\n".join(
                    reflection_response.areas_for_improvement
                ),
            )

            # 3. Update short-term memory
            memory_updates = await self.generate_updated_short_term_memory(
                summary_response
            )

            # Store updated memory
            await self.memory_manager.store_short_term_memory(
                user_info=memory_updates.user_info,
                last_conversation_summary=str(summary_response),
                recent_goal_and_status=memory_updates.recent_goal_and_status,
                important_context=memory_updates.important_context,
                agent_beliefs=memory_updates.agent_beliefs,
                agent_info=memory_updates.agent_info,
            )
        except Exception as e:
            logger.error(f"Error memory conversation: {e}")
            raise e

        logger.info("Successfully completed all conversation end tasks")
