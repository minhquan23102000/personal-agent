from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from mirascope.core import prompt_template, Messages, litellm
from pydantic import BaseModel, Field
from loguru import logger

if TYPE_CHECKING:
    from src.agent.base_agent import BaseAgent


class UserProfile(BaseModel):
    """Model for user profile during initialization."""

    communication_style: str = Field(
        description="User's preferred communication style, formality level, and language preferences"
    )
    personality_traits: str = Field(
        description="Key personality traits and characteristics identified through interaction"
    )
    interests_and_preferences: str = Field(
        description="User's main interests, preferences, and areas of focus"
    )
    technical_proficiency: str = Field(
        description="Assessment of user's technical knowledge and comfort level"
    )
    interaction_preferences: str = Field(
        description="Provide tips for enhancing communication skills, focusing on preferred styles, language choices, and tone adjustments that resonate with different audiences."
    )
    how_to_address_user: str = Field(description="How the user prefers to be addressed")
    agent_background_story: str = Field(
        description="Craft a compelling background story for an agent that reflects the user's values and preferences, highlighting their motivations, experiences, and core beliefs to ensure alignment with the user's perspective."
    )
    agent_beliefs: str = Field(
        description="Initial beliefs emphasizing the key characteristics and tendencies that shape behavior and decision-making."
    )
    system_prompt: str = Field(
        description="Develop an initial system prompt that clearly outlines the agent's purpose, key functionalities, and any initial instructions."
    )


@prompt_template(
    """
SYSTEM: You are an AI agent conducting an initial user profiling conversation. Your goal is to understand the user's preferences, personality, and needs through natural conversation. Act like a psychologist, be empathetic and understanding, smart and friendly. Be polite.

OBJECTIVE:
- Understand user's communication style and preferences
- Assess technical background and expertise
- Identify personality traits and interaction preferences
- Discover areas of interest and expertise
- Establish rapport and build trust

GUIDELINES:
1. Keep the conversation natural and engaging
2. Ask open-ended questions
3. Follow up on interesting points
4. Show active listening by referencing previous responses
5. Adapt your questions based on user's responses
6. Maintain a balance between gathering information and building rapport
7. Collect relevant information from the user without delving into excessive detail or executing specific tasks.
8. Do not ask directly about user personality and preferences, instead ask indirectly andd infer from them.
9. Keep the conversation short and concise straight to the point. Do not go into deep details.

REQUIRED INFORMATION TO GATHER:
- Language preferences
- Preferred form of address
- Main interests and focus areas
- Proficiency and knowledge level
- Ask user to set up your profile aka a preferred agent profile (gender, personality, communication style, characteristics, communication style)


Based on the conversation flow, determine the next best question or response to gather necessary information while maintaining natural dialogue. Keep the conversation engaging and adapt based on previous responses.

Analyze the conversation history and provide the next best question or response that will:
1. Feel natural in the conversation flow
2. Help gather missing required information
3. Build upon previous responses
4. Maintain engagement

Respond conversationally as you would to the user, not with analysis.

MESSAGES:
{history}
"""
)
def get_next_question(history): ...


@prompt_template(
    """
SYSTEM: You are an expert AI psychologist specializing in personality assessment and user profiling. Your task is to analyze the conversation history and create a comprehensive user profile that will guide the agent's future interactions.

OBJECTIVE:
Analyze the conversation to extract and infer:
1. Communication patterns and preferences
2. Personality traits and behavioral tendencies
3. Technical expertise and learning style
4. Interaction preferences and boundaries
5. Core values and motivations

ANALYSIS FRAMEWORK:
1. Direct Information
   - Explicitly stated preferences
   - Clear boundaries or requirements
   - Specific interests or expertise areas

2. Behavioral Indicators
   - Communication style (formal/casual, detailed/concise)
   - Response patterns (analytical/emotional, quick/thoughtful)
   - Technical vocabulary usage
   - Problem-solving approach

3. Psychological Insights
   - Underlying motivations
   - Decision-making style
   - Learning preferences
   - Trust and authority orientation

OUTPUT REQUIREMENTS:
Based on your analysis, construct:
1. User Profile
   - Communication style and language preferences
   - Key personality traits and characteristics
   - Main interests and focus areas
   - Proficiency and knowledge level
   - Preferred interaction communication style
   - Preferred form of address

2. Agent Adaptation
   - Create an agent profile that complements the user
   - Define agent beliefs and behavioral guidelines
   - Develop a tailored system prompt

CONVERSATION HISTORY:
{history:list}

Remember to:
- Base all insights on observable evidence from the conversation
- Balance direct observations with reasonable inferences
- Maintain professional objectivity in your analysis
- Consider cultural and contextual factors
- Prioritize user comfort and preferences
"""
)
def analyze_user_profile(history): ...


class AgentInitializer:
    @staticmethod
    async def initialize_agent(agent: "BaseAgent") -> None:
        """Initialize agent with personalized profile based on user interaction."""
        try:
            # Initial greeting and explanation
            welcome_message = Messages.Assistant(
                "Hello! I'm your personal agent. To help me serve you better, "
                "I'd like to have a brief conversation to understand your preferences "
                "and needs. Would you like to proceed with a quick personalization chat?"
            )
            agent.history.append(welcome_message)
            agent.interface.print_agent_message(welcome_message.content)

            # Get user consent
            response = agent.interface.input("[User]: ")
            if not response.lower().startswith(("y", "sure", "ok")):
                agent.interface.print_system_message(
                    "Skipping personalization process."
                )
                return

            agent.history.append(Messages.User(response))

            # Conduct dynamic initialization conversation
            await AgentInitializer._conduct_dynamic_conversation(agent)

            # Analyze conversation and create profile
            profile = await AgentInitializer._analyze_user_profile(agent)

            # Update agent's context memory
            await AgentInitializer._update_agent_context(agent, profile)

        except Exception as e:
            logger.error(f"Error during agent initialization: {e}")
            agent.interface.print_system_message(
                "Error during initialization. Proceeding with default settings.",
                type="error",
            )

    @staticmethod
    async def _conduct_dynamic_conversation(agent: "BaseAgent") -> None:
        """Conduct a dynamic, natural conversation to gather user information."""
        max_turns = 30  # Maximum conversation turns to prevent endless loops
        turns = 0

        while turns < max_turns:
            # Get next question/response from the agent
            @litellm.call(model=agent.reflection_model)
            def get_next():
                return get_next_question(history=agent.history)

            next_message = get_next()

            # # Check if we have gathered all necessary information
            # if (
            #     turns >= 4
            #     and "thank" in next_message.content.lower()
            #     or "conclude" in next_message.content.lower()
            # ):
            #     break

            # Send agent's message
            agent.history.append(Messages.Assistant(next_message.content))
            agent.interface.print_agent_message(next_message.content)

            # Get user's response
            response = agent.interface.input("[User]: ")

            # if user message exit, end, quit, then break
            if (
                "exit" in response.lower()
                or "end" in response.lower()
                or "quit" in response.lower()
            ):
                break

            agent.history.append(Messages.User(response))
            turns += 1

    @staticmethod
    async def _analyze_user_profile(agent: "BaseAgent") -> UserProfile:
        """Analyze conversation history to create user profile."""

        @litellm.call(
            model=agent.reflection_model, response_model=UserProfile, json_mode=True
        )
        def analyze():
            return analyze_user_profile(history=agent.history)

        return analyze()

    @staticmethod
    async def _update_agent_context(agent: "BaseAgent", profile: UserProfile) -> None:
        """Update agent's context memory with user profile information."""
        if not agent.memory_manager:
            return

        context_memory = await agent.memory_manager.store_context_memory(
            conversation_id=agent.conversation_id,
            user_info=f"""
                Communication Style: {profile.communication_style}
                Personality: {profile.personality_traits}
                Interests: {profile.interests_and_preferences}
                Technical Level: {profile.technical_proficiency}
                Preferences: {profile.interaction_preferences}
            """,
            last_conversation_summary="Initial user profiling conversation completed",
            recent_goal_and_status="- Complete user profiling [Done]",
            important_context=f"- User prefers {profile.communication_style} communication\n- Technical level: {profile.technical_proficiency}",
            agent_beliefs=profile.agent_beliefs,
            agent_info=profile.agent_background_story,
            environment_info="",
            how_to_address_user=profile.how_to_address_user,
        )

        agent.context_memory = context_memory
        agent.system_prompt = profile.system_prompt

        # clear history
        # agent.history = []

        agent.interface.print_system_message(
            "Agent initialized with personalized profile completed. Proceeding to the main conversation.",
            type="info",
        )
