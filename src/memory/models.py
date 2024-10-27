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
    user_info: str
    last_conversation_summary: str
    recent_goal_and_status: str
    important_context: str
    agent_beliefs: str  # Store the agent's current beliefs about the world and the user's intentions.
    agent_info: (
        str  # Store the agent's basic information (name, role, personality, etc)
    )


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
    entity_embeddings: bytes
    text_embedding: bytes
    keywords: List[str]


class EntityRelationship(BaseModel):
    relationship_id: int
    relationship_text: str
    embedding: bytes


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
