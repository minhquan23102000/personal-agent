from typing import AsyncGenerator, Literal, Optional, List
import sqlite3
import sqlite_vec
import json
from datetime import datetime, UTC
import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from loguru import logger

from src.memory.database.base import BaseDatabase
from src.memory.models import (
    ConversationMemory,
    Knowledge,
    EntityRelationship,
    ConversationSummary,
    MessageType,
    ShortTermMemory,
)


class SQLiteDatabase(BaseDatabase):
    def __init__(self, db_path: str = "memory.db", embedding_size: int = 364):
        super().__init__({"db_path": db_path, "embedding_size": embedding_size})

    def _validate_config(self) -> None:
        db_path = self.connection_config["db_path"]
        if not isinstance(db_path, str):
            raise ValueError("db_path must be a string")

        embedding_size = self.connection_config["embedding_size"]
        if not isinstance(embedding_size, int) or embedding_size <= 0:
            raise ValueError("embedding_size must be a positive integer")

    def _setup_connection(self) -> None:
        """Initialize SQLite and load vector extension"""
        try:
            db_path = self.connection_config["db_path"]
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(db_path) as conn:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)

                # Verify installation
                (vec_version,) = conn.execute("select vec_version()").fetchone()
                logger.info(f"sqlite-vec version: {vec_version}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite database: {str(e)}")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[sqlite3.Connection, None]:
        conn = sqlite3.connect(self.connection_config["db_path"])
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            yield conn
        finally:
            conn.close()

    async def initialize(self) -> None:
        """Initialize database schema"""
        async with self.get_connection() as conn:
            # Add short term memory state table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_term_memory_state (
                    id INTEGER PRIMARY KEY,
                    user_info TEXT NOT NULL,
                    last_conversation_summary TEXT NOT NULL,
                    recent_goal_and_status TEXT NOT NULL,
                    important_context TEXT NOT NULL,
                    agent_beliefs TEXT NOT NULL,
                    agent_info TEXT NOT NULL,
                    timestamp DATETIME NOT NULL
                )
                """
            )

            # Create short term memory table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS short_term_memory (
                    conversation_id INTEGER PRIMARY KEY,
                    turn_id INTEGER NOT NULL,
                    timestamp DATETIME NOT NULL,
                    sender TEXT NOT NULL,
                    message_content TEXT NOT NULL,
                    message_type TEXT NOT NULL
                )
            """
            )

            # Create knowledge table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge (
                    knowledge_id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    entities TEXT NOT NULL,
                    entity_embeddings BLOB NOT NULL,
                    text_embedding BLOB NOT NULL,
                    keywords TEXT NOT NULL
                )
            """
            )

            # Create vector index for knowledge
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0(
                    text_embedding({self.connection_config['embedding_size']})
                )
                
            """
            )

            # Create entities table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    relationship_id INTEGER PRIMARY KEY,
                    relationship_text TEXT NOT NULL,
                    embedding BLOB NOT NULL
                )
            """
            )

            # Create vector index for entities
            conn.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS entities_vec USING vec0(
                    embedding({self.connection_config['embedding_size']})
                )
            """
            )

            # Create conversation summary table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summary (
                    conversation_id INTEGER PRIMARY KEY,
                    prompt TEXT NOT NULL,
                    feedback_text TEXT,
                    example TEXT,
                    improvement_suggestion TEXT,
                    improve_prompt TEXT NOT NULL,
                    reward_score REAL NOT NULL,
                    conversation_summary TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES short_term_memory(conversation_id)
                )
            """
            )

            conn.commit()

    async def store_conversation(
        self,
        sender: str,
        message_content: str,
        message_type: MessageType,
        conversation_id: Optional[int] = None,
    ) -> ConversationMemory:
        """SQLite implementation of conversation storage"""
        async with self.get_connection() as conn:
            if conversation_id is None:
                cursor = conn.execute(
                    """
                    INSERT INTO short_term_memory (turn_id, timestamp, sender, message_content, message_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (1, datetime.now(UTC), sender, message_content, message_type.value),
                )
                conversation_id = cursor.lastrowid or 0  # Ensure it's not None
                turn_id = 1
            else:
                cursor = conn.execute(
                    "SELECT MAX(turn_id) FROM short_term_memory WHERE conversation_id = ?",
                    (conversation_id,),
                )
                max_turn = cursor.fetchone()
                turn_id = (max_turn[0] or 0) + 1

                conn.execute(
                    """
                    INSERT INTO short_term_memory (conversation_id, turn_id, timestamp, sender, message_content, message_type)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conversation_id,
                        turn_id,
                        datetime.now(UTC),
                        sender,
                        message_content,
                        message_type.value,
                    ),
                )

            conn.commit()

            return ConversationMemory(
                conversation_id=conversation_id,
                turn_id=turn_id,
                timestamp=datetime.now(UTC),
                sender=sender,
                message_content=message_content,
                message_type=message_type,
            )

    async def store_knowledge(
        self,
        text: str,
        entities: List[str],
        keywords: List[str],
        text_embedding: np.ndarray,
        entity_embeddings: np.ndarray,
    ) -> Knowledge:
        """Store knowledge with embeddings"""
        async with self.get_connection() as conn:
            text_embedding_blob = sqlite_vec.serialize_float32(text_embedding.tolist())
            entity_embeddings_blob = sqlite_vec.serialize_float32(
                entity_embeddings.tolist()
            )

            cursor = conn.execute(
                """
                INSERT INTO knowledge (text, entities, entity_embeddings, text_embedding, keywords)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    text,
                    json.dumps(entities),
                    entity_embeddings_blob,
                    text_embedding_blob,
                    json.dumps(keywords),
                ),
            )
            knowledge_id = cursor.lastrowid or 0

            # Add to vector index
            conn.execute(
                "INSERT INTO knowledge_vec(rowid, text_embedding) VALUES (?, ?)",
                (knowledge_id, text_embedding_blob),
            )

            conn.commit()

            return Knowledge(
                knowledge_id=knowledge_id,
                text=text,
                entities=entities,
                entity_embeddings=entity_embeddings_blob,
                text_embedding=text_embedding_blob,
                keywords=keywords,
            )

    async def store_entity_relationship(
        self,
        relationship_text: str,
        embedding: np.ndarray,
    ) -> EntityRelationship:
        """Store entity relationship with embedding"""
        async with self.get_connection() as conn:
            embedding_blob = sqlite_vec.serialize_float32(embedding.tolist())

            cursor = conn.execute(
                """
                INSERT INTO entities (relationship_text, embedding)
                VALUES (?, ?)
                """,
                (relationship_text, embedding_blob),
            )
            relationship_id = cursor.lastrowid or 0

            # Add to vector index
            conn.execute(
                "INSERT INTO entities_vec(rowid, embedding) VALUES (?, ?)",
                (relationship_id, embedding_blob),
            )

            conn.commit()

            return EntityRelationship(
                relationship_id=relationship_id,
                relationship_text=relationship_text,
                embedding=embedding_blob,
            )

    async def get_conversation_context(
        self, conversation_id: int, limit: int = 10
    ) -> List[ConversationMemory]:
        """Retrieve conversation context"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT conversation_id, turn_id, timestamp, sender, message_content, message_type
                FROM short_term_memory
                WHERE conversation_id = ?
                ORDER BY turn_id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            )

            rows = cursor.fetchall()
            return [
                ConversationMemory(
                    conversation_id=row[0],
                    turn_id=row[1],
                    timestamp=datetime.fromisoformat(row[2]),
                    sender=row[3],
                    message_content=row[4],
                    message_type=MessageType(row[5]),
                )
                for row in rows
            ]

    async def search_similar_knowledge(
        self,
        query_embedding: np.ndarray,
        vector_column: Literal[
            "text_embedding", "entity_embeddings"
        ] = "text_embedding",
        limit: int = 5,
    ) -> List[Knowledge]:
        """Search for similar knowledge using vector similarity"""
        async with self.get_connection() as conn:
            query_blob = sqlite_vec.serialize_float32(query_embedding.tolist())
            cursor = conn.execute(
                f"""
                SELECT k.*, vec_cosine_similarity(k_vec.{vector_column}, ?) as similarity
                FROM knowledge k
                JOIN knowledge_vec k_vec ON k.knowledge_id = k_vec.rowid
                ORDER BY similarity DESC
                LIMIT ?
                """,
                (query_blob, limit),
            )

            rows = cursor.fetchall()
            return [
                Knowledge(
                    knowledge_id=row[0],
                    text=row[1],
                    entities=json.loads(row[2]),
                    entity_embeddings=row[3],
                    text_embedding=row[4],
                    keywords=json.loads(row[5]),
                )
                for row in rows
            ]

    async def search_similar_entities(
        self, query_embedding: np.ndarray, limit: int = 5
    ) -> List[EntityRelationship]:
        """Search for similar entities using vector similarity"""
        async with self.get_connection() as conn:
            query_blob = sqlite_vec.serialize_float32(query_embedding.tolist())
            cursor = conn.execute(
                """
                SELECT e.*, vec_cosine_similarity(e_vec.embedding, ?) as similarity
                FROM entities e
                JOIN entities_vec e_vec ON e.relationship_id = e_vec.rowid
                ORDER BY similarity DESC
                LIMIT ?
                """,
                (query_blob, limit),
            )

            rows = cursor.fetchall()
            return [
                EntityRelationship(
                    relationship_id=row[0],
                    relationship_text=row[1],
                    embedding=row[2],
                )
                for row in rows
            ]

    async def store_conversation_summary(
        self,
        conversation_id: int,
        prompt: str,
        conversation_summary: str,
        improve_prompt: str,
        reward_score: float,
        feedback_text: Optional[str] = None,
        example: Optional[str] = None,
        improvement_suggestion: Optional[str] = None,
    ) -> ConversationSummary:
        """Store conversation summary"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO conversation_summary (
                    conversation_id, prompt, feedback_text, example,
                    improvement_suggestion, improve_prompt, reward_score,
                    conversation_summary, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    prompt,
                    feedback_text,
                    example,
                    improvement_suggestion,
                    improve_prompt,
                    reward_score,
                    conversation_summary,
                    datetime.now(UTC),
                ),
            )

            conn.commit()

            return ConversationSummary(
                conversation_id=conversation_id,
                prompt=prompt,
                feedback_text=feedback_text,
                example=example,
                improvement_suggestion=improvement_suggestion,
                improve_prompt=improve_prompt,
                reward_score=reward_score,
                conversation_summary=conversation_summary,
                timestamp=datetime.now(UTC),
            )

    async def get_conversation_summary(
        self, conversation_id: int
    ) -> Optional[ConversationSummary]:
        """Get conversation summary"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM conversation_summary
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            return ConversationSummary(
                conversation_id=row[0],
                prompt=row[1],
                feedback_text=row[2],
                example=row[3],
                improvement_suggestion=row[4],
                improve_prompt=row[5],
                reward_score=row[6],
                conversation_summary=row[7],
                timestamp=datetime.fromisoformat(row[8]),
            )

    async def update_conversation_feedback(
        self,
        conversation_id: int,
        feedback_text: str,
        reward_score: float,
        improvement_suggestion: Optional[str] = None,
        example: Optional[str] = None,
    ) -> Optional[ConversationSummary]:
        """Update conversation feedback"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE conversation_summary
                SET feedback_text = ?,
                    reward_score = ?,
                    improvement_suggestion = ?,
                    example = ?
                WHERE conversation_id = ?
                RETURNING *
                """,
                (
                    feedback_text,
                    reward_score,
                    improvement_suggestion,
                    example,
                    conversation_id,
                ),
            )

            row = cursor.fetchone()
            if not row:
                return None

            conn.commit()

            return ConversationSummary(
                conversation_id=row[0],
                prompt=row[1],
                feedback_text=row[2],
                example=row[3],
                improvement_suggestion=row[4],
                improve_prompt=row[5],
                reward_score=row[6],
                conversation_summary=row[7],
                timestamp=datetime.fromisoformat(row[8]),
            )

    async def get_best_performing_prompts(
        self, limit: int = 5
    ) -> List[ConversationSummary]:
        """Get best performing prompts"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM conversation_summary
                ORDER BY reward_score DESC
                LIMIT ?
                """,
                (limit,),
            )

            rows = cursor.fetchall()
            return [
                ConversationSummary(
                    conversation_id=row[0],
                    prompt=row[1],
                    feedback_text=row[2],
                    example=row[3],
                    improvement_suggestion=row[4],
                    improve_prompt=row[5],
                    reward_score=row[6],
                    conversation_summary=row[7],
                    timestamp=datetime.fromisoformat(row[8]),
                )
                for row in rows
            ]

    async def close(self) -> None:
        """No persistent connection to close in SQLite"""
        pass

    async def store_short_term_memory(
        self,
        user_info: str,
        last_conversation_summary: str,
        recent_goal_and_status: str,
        important_context: str,
        agent_beliefs: str,
        agent_info: str,
    ) -> ShortTermMemory:
        """Store short-term memory state"""
        async with self.get_connection() as conn:
            # First clear any existing state since we only keep current state
            conn.execute("DELETE FROM short_term_memory_state")

            cursor = conn.execute(
                """
                INSERT INTO short_term_memory_state (
                    user_info, last_conversation_summary, recent_goal_and_status,
                    important_context, agent_beliefs, agent_info, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ? )
                """,
                (
                    user_info,
                    last_conversation_summary,
                    recent_goal_and_status,
                    important_context,
                    agent_beliefs,
                    agent_info,
                    datetime.now(UTC),
                ),
            )

            conn.commit()

            return ShortTermMemory(
                user_info=user_info,
                last_conversation_summary=last_conversation_summary,
                recent_goal_and_status=recent_goal_and_status,
                important_context=important_context,
                agent_beliefs=agent_beliefs,
                agent_info=agent_info,
            )

    async def get_short_term_memory(self) -> Optional[ShortTermMemory]:
        """Retrieve current short-term memory state"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT user_info, last_conversation_summary, recent_goal_and_status,
                       important_context, agent_beliefs, agent_info
                FROM short_term_memory_state
                ORDER BY timestamp DESC
                LIMIT 1
                """
            )

            row = cursor.fetchone()
            if not row:
                return None

            return ShortTermMemory(
                user_info=row[0],
                last_conversation_summary=row[1],
                recent_goal_and_status=row[2],
                important_context=row[3],
                agent_beliefs=row[4],
                agent_info=row[5],
            )
