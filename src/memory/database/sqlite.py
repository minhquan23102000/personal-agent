from typing import AsyncGenerator, Literal, Optional, List
import sqlite3
import sqlite_vec
import json
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field

from src.memory.database.base import BaseDatabase
from src.memory.models import (
    ConversationMemory,
    Knowledge,
    EntityRelationship,
    ConversationSummary,
    MessageType,
    ShortTermMemory,
)
import re
from src.config import DATA_DIR


@dataclass
class SQLiteDatabase(BaseDatabase):
    db_uri: str
    embedding_size: int
    similarity_threshold: float = 0.5

    def __post_init__(self):
        # Clean up db_uri
        self.db_uri = re.sub(r"[^\w]", "_", self.db_uri)
        self.db_uri = re.sub(r"_{2,}", "_", self.db_uri)
        self.db_uri = self.db_uri.strip("_")

        if not self.db_uri.endswith(".db"):
            self.db_uri = f"{DATA_DIR}/{self.db_uri}.db"

        # reverse similarity threshold
        self.similarity_threshold = 1 - self.similarity_threshold

        super().__post_init__()

    def _setup_connection(self) -> None:
        """Initialize SQLite and load vector extension"""
        try:
            db_uri = self.db_uri
            Path(db_uri).parent.mkdir(parents=True, exist_ok=True)

            with sqlite3.connect(db_uri) as conn:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)

                # Verify installation
                (vec_version,) = conn.execute("select vec_version()").fetchone()
                logger.info(f"sqlite-vec version: {vec_version}")
                logger.info("Initializing database schema")
                self.initialize(conn)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize SQLite database: {str(e)}")

    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[sqlite3.Connection, None]:
        conn = sqlite3.connect(self.db_uri)
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            yield conn
        finally:
            conn.close()

    def initialize(self, conn: sqlite3.Connection) -> None:
        """Initialize database schema"""
        # Add short term memory state table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS short_term_memory_state (
                id INTEGER PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                user_info TEXT NOT NULL,
                last_conversation_summary TEXT NOT NULL,
                recent_goal_and_status TEXT NOT NULL,
                important_context TEXT NOT NULL,
                agent_beliefs TEXT NOT NULL,
                agent_info TEXT NOT NULL,
                environment_info TEXT NOT NULL,
                how_to_address_user TEXT DEFAULT "",
                timestamp DATETIME NOT NULL
            )
            """
        )

        # Create vector index for short term memory
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS short_term_memory_vec USING vec0(
                id integer primary key,
                summary_embedding float[{self.embedding_size}]
            )
            """
        )

        # Create short term memory table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT,
                turn_id INTEGER NOT NULL,
                timestamp DATETIME NOT NULL,
                sender TEXT NOT NULL,
                message_content TEXT NOT NULL,
                message_type TEXT NOT NULL,
                PRIMARY KEY (conversation_id, turn_id)
            )
        """
        )

        # Create knowledge table (without embedding columns)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge (
                knowledge_id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                entities TEXT NOT NULL,
                keywords TEXT NOT NULL
            )
        """
        )

        # Create vector index for knowledge
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_vec USING vec0(
                knowledge_id integer primary key,
                text_embedding float[{self.embedding_size}],
                entity_embeddings float[{self.embedding_size}]
            )
            """
        )

        # Create entities table (without embedding column)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                relationship_id INTEGER PRIMARY KEY,
                relationship_text TEXT NOT NULL
            )
        """
        )

        # Create vector index for entities
        conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS entities_vec USING vec0(
                relationship_id integer primary key,
                embedding float[{self.embedding_size}]
            )
            """
        )

        # Create conversation summary table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_summary (
                conversation_id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                feedback_text TEXT,
                example TEXT,
                improvement_suggestion TEXT,
                improve_prompt TEXT NOT NULL,
                reward_score REAL NOT NULL,
                conversation_summary TEXT NOT NULL,
                timestamp DATETIME NOT NULL
            )
        """
        )

        conn.commit()

    async def store_conversation(
        self,
        sender: str,
        message_content: str,
        message_type: MessageType,
        conversation_id: str,
    ) -> ConversationMemory:
        """SQLite implementation of conversation storage"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT MAX(turn_id) FROM conversations WHERE conversation_id = ?",
                (conversation_id,),
            )
            max_turn = cursor.fetchone()
            turn_id = (max_turn[0] or 0) + 1

            conn.execute(
                """
                INSERT INTO conversations (conversation_id, turn_id, timestamp, sender, message_content, message_type)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    turn_id,
                    datetime.now(),
                    sender,
                    message_content,
                    message_type.value,
                ),
            )

            conn.commit()

            return ConversationMemory(
                conversation_id=conversation_id,
                turn_id=turn_id,
                timestamp=datetime.now(),
                sender=sender,
                message_content=message_content,
                message_type=message_type,
            )

    async def store_knowledge(
        self,
        text: str,
        entities: List[str],
        keywords: List[str],
        text_embedding: List[float],
        entity_embeddings: List[float] | None,
    ) -> Knowledge:
        """Store knowledge with embeddings"""
        async with self.get_connection() as conn:
            # Store in regular table
            cursor = conn.execute(
                """
                INSERT INTO knowledge (text, entities, keywords)
                VALUES (?, ?, ?)
                """,
                (
                    text,
                    json.dumps(entities),
                    json.dumps(keywords),
                ),
            )
            knowledge_id = cursor.lastrowid or 0

            # Store in vector table
            conn.execute(
                """
                INSERT INTO knowledge_vec(
                    knowledge_id, 
                    text_embedding,
                    entity_embeddings
                ) VALUES (?, ?, ?)
                """,
                (
                    knowledge_id,
                    json.dumps(text_embedding),
                    (
                        json.dumps(entity_embeddings)
                        if entity_embeddings is not None
                        else json.dumps([0] * self.embedding_size)
                    ),
                ),
            )

            conn.commit()

            return Knowledge(
                knowledge_id=knowledge_id,
                text=text,
                entities=entities,
                entity_embeddings=entity_embeddings,
                text_embedding=text_embedding,
                keywords=keywords,
            )

    async def store_entity_relationship(
        self,
        relationship_text: str,
        embedding: List[float],
    ) -> EntityRelationship:
        """Store entity relationship with embedding"""
        async with self.get_connection() as conn:
            # Store in regular table
            cursor = conn.execute(
                """
                INSERT INTO entities (relationship_text)
                VALUES (?)
                """,
                (relationship_text,),
            )
            relationship_id = cursor.lastrowid or 0

            # Store in vector table
            conn.execute(
                """
                INSERT INTO entities_vec(
                    relationship_id,
                    embedding
                ) VALUES (?, ?)
                """,
                (
                    relationship_id,
                    json.dumps(embedding),
                ),
            )

            conn.commit()

            return EntityRelationship(
                relationship_id=relationship_id,
                relationship_text=relationship_text,
                embedding=embedding,
            )

    async def get_conversation_context(
        self, conversation_id: str, limit: int = 10
    ) -> List[ConversationMemory]:
        """Retrieve conversation context"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT conversation_id, turn_id, timestamp, sender, message_content, message_type
                FROM conversations
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
        query_embedding: List[float],
        vector_column: Literal[
            "text_embedding", "entity_embeddings"
        ] = "text_embedding",
        limit: int = 5,
    ) -> List[Knowledge]:
        """Search for similar knowledge using cosine similarity"""
        async with self.get_connection() as conn:
            query_json = json.dumps(query_embedding)

            cursor = conn.execute(
                f"""
                WITH vector_matches AS (
                    SELECT 
                        knowledge_id,
                        {vector_column} as embedding,
                        vec_distance_cosine({vector_column}, ?) as distance
                    FROM knowledge_vec
                    WHERE vec_distance_cosine({vector_column}, ?) <= ?
                    ORDER BY distance
                    LIMIT ?
                )
                SELECT 
                    k.*,
                    vm.embedding,
                    vm.distance
                FROM knowledge k
                JOIN vector_matches vm ON k.knowledge_id = vm.knowledge_id
                ORDER BY vm.distance
                """,
                (query_json, query_json, self.similarity_threshold, limit),
            )

            rows = cursor.fetchall()
            return [
                Knowledge(
                    knowledge_id=row[0],
                    text=row[1],
                    entities=json.loads(row[2]),
                )
                for row in rows
            ]

    async def search_similar_entities(
        self, query_embedding: List[float], limit: int = 5
    ) -> List[EntityRelationship]:
        """Search for similar entities using cosine similarity"""
        async with self.get_connection() as conn:
            query_json = json.dumps(query_embedding)

            cursor = conn.execute(
                """
                WITH vector_matches AS (
                    SELECT 
                        relationship_id,
                        embedding,
                        vec_distance_cosine(embedding, ?) as distance
                    FROM entities_vec
                    WHERE vec_distance_cosine(embedding, ?) <= ?
                    ORDER BY distance
                    LIMIT ?
                )
                SELECT 
                    e.*,
                    vm.embedding,
                    vm.distance
                FROM entities e
                JOIN vector_matches vm ON e.relationship_id = vm.relationship_id
                ORDER BY vm.distance
                """,
                (query_json, query_json, self.similarity_threshold, limit),
            )

            rows = cursor.fetchall()
            return [
                EntityRelationship(
                    relationship_id=row[0],
                    relationship_text=row[1],
                )
                for row in rows
            ]

    async def store_conversation_summary(
        self,
        conversation_id: str,
        prompt: str,
        conversation_summary: str,
        improve_prompt: str,
        reward_score: float,
        feedback_text: str = "",
        example: str = "",
        improvement_suggestion: str = "",
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
                    datetime.now(),
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
                timestamp=datetime.now(),
            )

    async def get_conversation_summary(
        self, conversation_id: str
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
        conversation_id: str,
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
        conversation_id: str,
        user_info: str,
        last_conversation_summary: str,
        recent_goal_and_status: str,
        important_context: str,
        agent_beliefs: str,
        agent_info: str,
        environment_info: str,
        how_to_address_user: str,
        summary_embedding: List[float],
    ) -> ShortTermMemory:
        """Store short-term memory state with embedding"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO short_term_memory_state (
                    conversation_id, user_info, last_conversation_summary, recent_goal_and_status,
                    important_context, agent_beliefs, agent_info, environment_info, how_to_address_user, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_info,
                    last_conversation_summary,
                    recent_goal_and_status,
                    important_context,
                    agent_beliefs,
                    agent_info,
                    environment_info,
                    how_to_address_user,
                    datetime.now(),
                ),
            )

            memory_id = cursor.lastrowid

            # Store embedding in vector table
            conn.execute(
                """
                INSERT INTO short_term_memory_vec(
                    id,
                    summary_embedding
                ) VALUES (?, ?)
                """,
                (
                    memory_id,
                    json.dumps(summary_embedding),
                ),
            )

            conn.commit()

            return ShortTermMemory(
                conversation_id=conversation_id,
                user_info=user_info,
                last_conversation_summary=last_conversation_summary,
                recent_goal_and_status=recent_goal_and_status,
                important_context=important_context,
                agent_beliefs=agent_beliefs,
                agent_info=agent_info,
                environment_info=environment_info,
                how_to_address_user=how_to_address_user,
                timestamp=datetime.now(),
            )

    async def get_short_term_memory(
        self, conversation_id: Optional[str] = None
    ) -> Optional[ShortTermMemory]:
        """Retrieve current short-term memory state"""
        async with self.get_connection() as conn:
            query = """
                SELECT conversation_id, user_info, last_conversation_summary, recent_goal_and_status,
                       important_context, agent_beliefs, agent_info, environment_info, how_to_address_user, timestamp
                FROM short_term_memory_state
            """
            params = []

            if conversation_id:
                query += " WHERE conversation_id = ?"
                params.append(conversation_id)

            query += " ORDER BY timestamp DESC LIMIT 1"

            cursor = conn.execute(query, params)
            row = cursor.fetchone()
            if not row:
                return None

            return ShortTermMemory(
                conversation_id=row[0],
                user_info=row[1],
                last_conversation_summary=row[2],
                recent_goal_and_status=row[3],
                important_context=row[4],
                agent_beliefs=row[5],
                agent_info=row[6],
                environment_info=row[7],
                how_to_address_user=row[8],
                timestamp=row[9],
            )

    async def get_latest_conversation_summary(self) -> Optional[ConversationSummary]:
        """Get the most recent conversation summary"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM conversation_summary
                ORDER BY timestamp DESC
                LIMIT 1
                """
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

    async def search_similar_short_term_memories(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.1,
    ) -> List[ShortTermMemory]:
        """Search for similar short-term memories using cosine similarity"""
        async with self.get_connection() as conn:
            query_json = json.dumps(query_embedding)

            cursor = conn.execute(
                """
                WITH vector_matches AS (
                    SELECT 
                        id,
                        summary_embedding,
                        vec_distance_cosine(summary_embedding, ?) as distance
                    FROM short_term_memory_vec
                    WHERE vec_distance_cosine(summary_embedding, ?) <= ?
                    ORDER BY distance
                    LIMIT ?
                )
                SELECT 
                    s.*,
                    vm.distance
                FROM short_term_memory_state s
                JOIN vector_matches vm ON s.id = vm.id
                ORDER BY vm.distance
                """,
                (query_json, query_json, 1 - similarity_threshold, limit),
            )

            rows = cursor.fetchall()
            return [
                ShortTermMemory(
                    conversation_id=row[1],
                    user_info=row[2],
                    last_conversation_summary=row[3],
                    recent_goal_and_status=row[4],
                    important_context=row[5],
                    agent_beliefs=row[6],
                    agent_info=row[7],
                    environment_info=row[8],
                    timestamp=row[10],
                    how_to_address_user=row[9],
                )
                for row in rows
            ]

    async def get_recent_conversation_summaries(
        self, limit: int = 5
    ) -> List[ConversationSummary]:
        """Get the most recent conversation summaries ordered by timestamp"""
        async with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM conversation_summary
                ORDER BY timestamp DESC
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

    async def get_conversation_details(
        self,
        conversation_id: Optional[str] = None,
        senders: Optional[List[str]] = None,
        limit: int = 100,
    ) -> List[ConversationMemory]:
        """Retrieve conversation details filtered by conversation ID and senders"""
        async with self.get_connection() as conn:
            query = """
                SELECT conversation_id, turn_id, timestamp, sender, 
                       message_content, message_type
                FROM conversations
                WHERE 1=1
            """
            params = []

            if conversation_id:
                query += " AND conversation_id = ?"
                params.append(conversation_id)

            if senders:
                placeholders = ",".join("?" * len(senders))
                query += f" AND sender IN ({placeholders})"
                params.extend(senders)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            cursor = conn.execute(query, params)
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
