from typing import TYPE_CHECKING, List
import inspect
import datetime

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent
    from src.memory.models import ConversationSummary


def build_recent_conversation_context(summaries: List["ConversationSummary"]) -> str:
    """Format conversation summaries into readable context"""
    if not summaries:
        return ""

    conversation_context = "\n\n".join(
        f"[{i+1}] Conversation {summary.conversation_id} at {summary.timestamp.strftime('%Y-%m-%d %H:%M:%S')}:\n"
        f"    {summary.conversation_summary}"
        for i, summary in enumerate(summaries)
    )

    return inspect.cleandoc(
        f"""
# RECENT CONVERSATIONS SUMMARIES:
{conversation_context}
        """
    ).strip()


def build_system_prompt(
    agent: "BaseAgent", recent_conversations: List["ConversationSummary"]
) -> str:
    """Build system prompt with recent conversation context"""

    return inspect.cleandoc(
        f"""
# AGENT ID: {agent.agent_id}
CURRENT TIME: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

# SYSTEM INSTRUCTIONS:
{agent.system_prompt}
        """
    ).strip()


def build_short_term_memory_prompt(agent: "BaseAgent") -> str:
    if agent.context_memory:
        return inspect.cleandoc(
            f"""
# CONTEXT MEMORY (VERY IMPORTANT INFORMATION AND CONTEXT FOR THE AGENT)

## AGENT INFORMATION & IDENTITY: 
{agent.context_memory.agent_info}

## AGENT BELIEFS: 
{agent.context_memory.agent_beliefs}

## USER INFORMATION: 
{agent.context_memory.user_info}

## HOW TO ADDRESS USER: 
{agent.context_memory.how_to_address_user}

## LAST CONVERSATION SUMMARY on {agent.context_memory.timestamp.strftime("%Y-%m-%d %H:%M:%S")}:
{agent.context_memory.last_conversation_summary}

## RECENT GOALS AND STATUS: 
{agent.context_memory.recent_goal_and_status}

## IMPORTANT CONTEXT: 
{agent.context_memory.important_context}

## ENVIRONMENT INFORMATION: 
{agent.context_memory.environment_info}
        """
        ).strip()
    else:
        return ""
