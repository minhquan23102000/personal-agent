from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel


class MessageType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    TOOL = "tool"


class ShortTermMemory(BaseModel):
    conversation_id: str
    user_info: str
    last_conversation_summary: str
    recent_goal_and_status: str
    important_context: str
    agent_beliefs: str  # Store the agent's current beliefs about the world and the user's intentions.
    agent_info: (
        str  # Store the agent's basic information (name, role, personality, etc)
    )
    environment_info: str
    how_to_address_user: str
    timestamp: datetime


class ConversationMemory(BaseModel):
    conversation_id: str
    turn_id: int
    timestamp: datetime
    sender: str
    message_content: str
    message_type: MessageType


class Knowledge(BaseModel):
    knowledge_id: int
    text: str
    entities: List[str]
    entity_embeddings: List[float] | None = None
    text_embedding: List[float] | None = None
    keywords: List[str] | None = None


class EntityRelationship(BaseModel):
    relationship_id: int
    relationship_text: str
    embedding: List[float] | None = None


class ConversationSummary(BaseModel):
    conversation_id: str
    prompt: str
    feedback_text: Optional[str]
    example: Optional[str]
    improvement_suggestion: Optional[str]
    improve_prompt: str
    reward_score: float
    conversation_summary: str
    timestamp: datetime
