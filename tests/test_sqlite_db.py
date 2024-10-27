import uuid
import pytest
import asyncio
from datetime import datetime, UTC
from typing import List
import json
import os
from pathlib import Path
from src.memory.database.sqlite import SQLiteDatabase
from src.memory.models import (
    ConversationMemory,
    Knowledge,
    EntityRelationship,
    ConversationSummary,
    MessageType,
    ShortTermMemory,
)

DB_URI = "test_memory"


@pytest.fixture()
def db():

    return SQLiteDatabase(db_uri=f"{DB_URI}_{uuid.uuid4}", embedding_size=364)


@pytest.fixture(autouse=True)
async def pytest_teardowns(db):
    # Setup
    await db._setup_connection()
    yield
    # Teardown
    await db.close()

    os.remove(db.db_uri)


@pytest.mark.asyncio
async def test_store_conversation(db):
    conversation = await db.store_conversation(
        sender="user",
        message_content="Hello, AI!",
        message_type=MessageType.TEXT,
        conversation_id="test_conv_1",
    )
    assert isinstance(conversation, ConversationMemory)
    assert conversation.conversation_id == "test_conv_1"
    assert conversation.sender == "user"
    assert conversation.message_content == "Hello, AI!"
    assert conversation.message_type == MessageType.TEXT


@pytest.mark.asyncio
async def test_store_knowledge(db):
    knowledge = await db.store_knowledge(
        text="The sky is blue.",
        entities=["sky"],
        keywords=["blue", "color"],
        text_embedding=[0.1] * 364,
        entity_embeddings=[0.2] * 364,
    )
    assert isinstance(knowledge, Knowledge)
    assert knowledge.text == "The sky is blue."
    assert knowledge.entities == ["sky"]
    assert knowledge.keywords == ["blue", "color"]


@pytest.mark.asyncio
async def test_store_entity_relationship(db):
    relationship = await db.store_entity_relationship(
        relationship_text="Sky has color blue", embedding=[0.3] * 364
    )
    assert isinstance(relationship, EntityRelationship)
    assert relationship.relationship_text == "Sky has color blue"


@pytest.mark.asyncio
async def test_get_conversation_context(db):
    # Store some conversations first
    for i in range(5):
        await db.store_conversation(
            sender="user",
            message_content=f"Message {i}",
            message_type=MessageType.TEXT,
            conversation_id="test_conv_2",
        )

    context = await db.get_conversation_context("test_conv_2", limit=3)
    assert len(context) == 3
    assert all(isinstance(c, ConversationMemory) for c in context)
    assert [c.message_content for c in context] == [
        "Message 4",
        "Message 3",
        "Message 2",
    ]


@pytest.mark.asyncio
async def test_search_similar_knowledge(db):
    # Store some knowledge first
    for i in range(5):
        await db.store_knowledge(
            text=f"Knowledge {i}",
            entities=["entity"],
            keywords=["keyword"],
            text_embedding=[0.1 * i] * 364,
            entity_embeddings=[0.2 * i] * 364,
        )

    results = await db.search_similar_knowledge([0.3] * 364, limit=3)
    assert len(results) == 3
    assert all(isinstance(k, Knowledge) for k in results)


@pytest.mark.asyncio
async def test_search_similar_entities(db):
    # Store some entity relationships first
    for i in range(5):
        await db.store_entity_relationship(
            relationship_text=f"Relationship {i}", embedding=[0.1 * i] * 364
        )

    results = await db.search_similar_entities([0.3] * 364, limit=3)
    assert len(results) == 3
    assert all(isinstance(e, EntityRelationship) for e in results)


@pytest.mark.asyncio
async def test_store_and_get_conversation_summary(db):
    summary = await db.store_conversation_summary(
        conversation_id="test_conv_3",
        prompt="Test prompt",
        conversation_summary="Test summary",
        improve_prompt="Improved prompt",
        reward_score=0.8,
        feedback_text="Good job",
        example="Example",
        improvement_suggestion="Suggestion",
    )
    assert isinstance(summary, ConversationSummary)

    retrieved_summary = await db.get_conversation_summary("test_conv_3")
    assert retrieved_summary.conversation_id == summary.conversation_id
    assert retrieved_summary.prompt == summary.prompt
    assert retrieved_summary.conversation_summary == summary.conversation_summary
    assert retrieved_summary.improve_prompt == summary.improve_prompt
    assert retrieved_summary.reward_score == summary.reward_score
    assert retrieved_summary.feedback_text == summary.feedback_text
    assert retrieved_summary.example == summary.example
    assert retrieved_summary.improvement_suggestion == summary.improvement_suggestion


@pytest.mark.asyncio
async def test_update_conversation_feedback(db):
    # First store a summary
    await db.store_conversation_summary(
        conversation_id="test_conv_4",
        prompt="Test prompt",
        conversation_summary="Test summary",
        improve_prompt="Improved prompt",
        reward_score=0.8,
    )

    updated_summary = await db.update_conversation_feedback(
        conversation_id="test_conv_4",
        feedback_text="Updated feedback",
        reward_score=0.9,
        improvement_suggestion="New suggestion",
        example="New example",
    )
    assert isinstance(updated_summary, ConversationSummary)
    assert updated_summary.feedback_text == "Updated feedback"
    assert updated_summary.reward_score == 0.9


@pytest.mark.asyncio
async def test_get_best_performing_prompts(db):
    # Store some summaries with different scores
    for i in range(5):
        await db.store_conversation_summary(
            conversation_id=f"test_conv_{i}_best_performing",
            prompt=f"Prompt {i}",
            conversation_summary=f"Summary {i}",
            improve_prompt=f"Improved {i}",
            reward_score=i,
        )

    best_prompts = await db.get_best_performing_prompts(limit=3)
    assert len(best_prompts) == 3
    assert all(isinstance(p, ConversationSummary) for p in best_prompts)
    assert [p.reward_score for p in best_prompts] == [4, 3, 2]


@pytest.mark.asyncio
async def test_store_and_get_short_term_memory(db):
    stored_memory = await db.store_short_term_memory(
        user_info="Test TEXT",
        last_conversation_summary="Last summary",
        recent_goal_and_status="Recent goal",
        important_context="Important context",
        agent_beliefs="Agent beliefs",
        agent_info="Agent info",
    )
    assert isinstance(stored_memory, ShortTermMemory)

    retrieved_memory = await db.get_short_term_memory()
    assert retrieved_memory == stored_memory
