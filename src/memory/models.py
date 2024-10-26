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


class ShortTermMemory(BaseModel):
    conversation_id: int
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
    conversation_id: int
    prompt: str
    feedback_text: Optional[str]
    example: Optional[str]
    improvement_suggestion: Optional[str]
    prompt_version: str
    reward_score: float
    conversation_summary: str
    timestamp: datetime
