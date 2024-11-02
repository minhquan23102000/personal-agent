import inspect
from loguru import logger
from src.memory.memory_toolkit.static_flow.conversation_summary import (
    BaseConversationSummary,
    generate_conversation_summary,
)
from src.memory.memory_toolkit.static_flow.short_term_memory import (
    BaseShortTermMemoryUpdate,
    generate_updated_short_term_memory,
)
from src.memory.memory_toolkit.static_flow.perform_relection import (
    BaseSelfReflection,
    perform_self_reflection,
)
from src.memory.memory_toolkit.static_flow.save_long_term import save_long_term_memory
from typing import TYPE_CHECKING
from rich import print

if TYPE_CHECKING:
    from src.memory.memory_manager import MemoryManager


def format_summary(summary: BaseConversationSummary) -> str:
    return inspect.cleandoc(
        f"""    
Summary: 
{summary.summary}
Key Points: 
{"\n- ".join(summary.key_points)}
Outcomes: 
{"\n- ".join(summary.outcomes)}
"""
    )


# Update end_conversation to include short-term memory update
async def reflection_conversation(
    memory_manager: "MemoryManager", user_feedback: str = "No user feedback provided."
) -> None:
    """Handle all end of conversation tasks."""
    print("Starting end of conversation memory...")

    try:

        # 1. Generate conversation summary and reflection
        summary_response = await generate_conversation_summary(memory_manager.agent)
        summary_str = format_summary(summary_response)
        memory_manager.agent.rotate_api_key()

        # 2. Save important information to long-term memory
        await save_long_term_memory(memory_manager.agent)
        memory_manager.agent.rotate_api_key()

        # 3. Perform self reflection
        reflection_response = await perform_self_reflection(
            memory_manager.agent, user_feedback
        )
        memory_manager.agent.rotate_api_key()

        print(f"Summary: {summary_str}\n\n")
        print(f"Reflection: {reflection_response}\n\n")

        # 4. Store reflection  and improvements
        await memory_manager.store_conversation_summary(
            conversation_id=memory_manager.agent.conversation_id,
            prompt=memory_manager.agent.system_prompt,
            conversation_summary=summary_str,
            improve_prompt=reflection_response.improved_prompt,
            reward_score=reflection_response.reward_score,
            feedback_text="\n".join(
                [
                    reflection_response.critique,
                    "Strengths:",
                    *[f"- {s}" for s in reflection_response.strengths],
                    "\nAreas for Improvement:",
                    *[f"- {a}" for a in reflection_response.areas_for_improvement],
                    "\nSpecific Examples:",
                    *[f"- {e}" for e in reflection_response.specific_examples],
                ]
            ),
            improvement_suggestion="\n".join(reflection_response.areas_for_improvement),
        )

        # 5. Update short-term memory
        memory_updates = await generate_updated_short_term_memory(
            summary=summary_str,
            agent=memory_manager.agent,
            user_feedback=user_feedback,
        )
        memory_manager.agent.rotate_api_key()

        print(f"Memory Updates: {memory_updates}")

        # Store updated memory
        await memory_manager.store_short_term_memory(
            conversation_id=memory_manager.agent.conversation_id,
            user_info=memory_updates.user_info,
            last_conversation_summary=summary_str,
            recent_goal_and_status=memory_updates.recent_goal_and_status,
            important_context=memory_updates.important_context,
            agent_beliefs=memory_updates.agent_beliefs,
            agent_info=memory_updates.agent_info,
            environment_info=memory_updates.environment_info,
            how_to_address_user=memory_updates.how_to_address_user,
        )
    except Exception as e:
        logger.error(f"Error memory conversation: {e}")
        raise e

    print("Successfully completed all conversation end tasks")
