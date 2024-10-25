## LLM Agent Memory System Design with SQLite (Turso) and RAG

This document outlines the design of a memory system for an LLM agent, leveraging SQLite (Turso) for storage and a Retrieval-Augmented Generation (RAG) strategy for knowledge retrieval. The system integrates short-term and long-term memory, incorporates keywords for efficient search, and utilizes entity and relationship structures for richer knowledge representation.

### 1. Overview

The goal is to build a robust and scalable memory system that enables the LLM agent to:

* **Retain context:** Remember past interactions within a conversation (short-term memory).
* **Access knowledge:** Retrieve relevant information from a larger knowledge base (long-term memory).
* **Understand relationships:**  Comprehend connections between different concepts and entities.
* **Reason and infer:**  Draw conclusions and make inferences based on stored knowledge.

### 2. Technology Stack

* **SQLite (Turso):** A serverless, distributed SQL database that provides scalability and ease of use.
* **RAG:** A technique that retrieves relevant information from a knowledge base to augment the LLM's prompt, improving response quality and accuracy.

### 3. Data Schema

The memory system is implemented using the following tables in SQLite:

**3.1 ShortTermMemory:**

| Column            | Data Type         | Description                                            |
| ----------------- | ----------------- | ------------------------------------------------------ |
| `conversation_id` | TEXT, PRIMARY KEY | Unique identifier for each conversation.               |
| `turn_id`         | INTEGER           | Sequential number for each turn within a conversation. |
| `speaker`         | TEXT              | "user" or "agent".                                     |
| `utterance`       | TEXT              | The text of the spoken turn.                           |
| `timestamp`       | DATETIME          | Timestamp of the turn.                                 |

**3.2 LongTermMemory:**

| Column          | Data Type         | Description                                                                   |
| --------------- | ----------------- | ----------------------------------------------------------------------------- |
| `document_id`   | TEXT, PRIMARY KEY | Unique identifier for each document.                                          |
| `document_text` | TEXT              | The content of the document.                                                  |
| `embedding`     | TBD               | Vector embedding of the document text.                                        |
| `source`        | TBD               | A list of source, reference of the document (e.g., website, file). (optional) |
| `metadata`      | TEXT              | JSON string containing additional metadata (e.g., author, date).              |
| `keywords`      | TEXT              | Comma-separated list of relevant keywords for the document.                   |


**3.3 ConversationContext:**

| Column               | Data Type         | Description                                                                                               |
| -------------------- | ----------------- | --------------------------------------------------------------------------------------------------------- |
| `conversation_id`    | TEXT, PRIMARY KEY | Foreign key referencing `ShortTermMemory`.                                                                |
| `relevant_documents` | TEXT              | Comma-separated list of `document_id`s from `LongTermMemory` deemed relevant to the current conversation. |


**3.4 Entities:**

| Column        | Data Type         | Description                                                                  |
| ------------- | ----------------- | ---------------------------------------------------------------------------- |
| `entity_id`   | TEXT, PRIMARY KEY | Unique identifier for each entity.                                           |
| `entity_name` | TEXT              | The name or label of the entity.                                             |
| `entity_type` | TEXT              | Category or type of the entity (e.g., "person", "location", "organization"). |
| `description` | TEXT              | Brief description of the entity.                                             |
| embedding     | VECTOR            | embedding vector for de duplication via sematic search                       |

**3.5 Relationships:**

| Column              | Data Type         | Description                                                         |
| ------------------- | ----------------- | ------------------------------------------------------------------- |
| `relationship_id`   | TEXT, PRIMARY KEY | Unique identifier for each relationship.                            |
| `source_entity`     | TEXT              | Foreign key referencing `Entities.entity_id`.                       |
| `target_entity`     | TEXT              | Foreign key referencing `Entities.entity_id`.                       |
| `relationship_type` | TEXT              | The type of relationship (e.g., "works_at", "located_in", "knows"). |

**3.6 DocumentEntities (Join Table):**

| Column        | Data Type | Description                                           |
| ------------- | --------- | ----------------------------------------------------- |
| `document_id` | TEXT      | Foreign key referencing `LongTermMemory.document_id`. |
| `entity_id`   | TEXT      | Foreign key referencing `Entities.entity_id`.         |

### 4. Relationships between Tables

* **ShortTermMemory** and **ConversationContext:** One-to-one relationship based on `conversation_id`.
* **ConversationContext** and **LongTermMemory:** Many-to-many relationship through the `relevant_documents` field in `ConversationContext`.
* **LongTermMemory** and **Entities:** Many-to-many relationship through the `DocumentEntities` join table.
* **Entities** and **Relationships:**  One-to-many relationship where `source_entity` and `target_entity` in `Relationships` reference `entity_id` in `Entities`.

### 5. RAG Implementation

**5.1 Retrieval:**

1. When a new user utterance is received, analyze the current conversation context (from `ShortTermMemory` and `ConversationContext`) and the user's question.
2. Query `LongTermMemory` to retrieve relevant documents:
    * **Semantic Search (optional):** If embeddings are used, calculate the embedding of the user's query and perform a similarity search against the `embedding` field in `LongTermMemory`.
    * **Keyword-based Search:** Utilize the `keywords` field in `LongTermMemory` and SQLite's full-text search capabilities to find documents containing relevant keywords from the user's query.
    * **Hybrid Approach:** Combine semantic and keyword-based search for optimal retrieval.
3. Store the `document_id`s of the retrieved documents in the `relevant_documents` field of the corresponding `ConversationContext` entry.

**5.2 Augmentation:**

1. Retrieve the content of the relevant documents from `LongTermMemory` using the `document_id`s stored in `ConversationContext`.
2. Construct a new prompt for the LLM that includes the user's question, relevant conversation history (from `ShortTermMemory`), and the retrieved document content.
3. The LLM generates a response based on this augmented prompt, incorporating the retrieved knowledge.


### 6. Keyword and Entity Management

* **Keyword Extraction:**  
    * Use LLM automatically extract relevant keywords from documents added to `LongTermMemory`.
    * Store the extracted keywords in the `keywords` field.
* **Entity Recognition and Linking:**
    * Employ LLM models to identify entities in documents added to `LongTermMemory`.
    * Link identified entities to existing entries in the `Entities` table or create new entries if they don't exist.
    * Populate the `DocumentEntities` table to establish the many-to-many relationship between documents and entities.
* **Relationship Extraction:**
    * Utilize relationship extraction techniques to identify relationships between entities mentioned in documents.
    * Add identified relationships to the `Relationships` table, linking the corresponding entities.


### 7. Example Workflow

1. **User starts a new conversation:** A new entry is created in `ShortTermMemory` and `ConversationContext` with a unique `conversation_id`.
2. **User asks a question:** The question is stored as a new turn in `ShortTermMemory`.
3. **RAG Retrieval:** The agent retrieves relevant documents from `LongTermMemory` based on the question and conversation history using semantic search, keyword search, or a hybrid approach.
4. **Context Update:** The `document_id`s of the retrieved documents are stored in the `relevant_documents` field of the `ConversationContext` entry.
5. **LLM Prompt Augmentation:** The agent constructs a new prompt for the LLM, including the user's question, relevant conversation history, and the content of the retrieved documents.
6. **Response Generation:** The LLM generates a response based on the augmented prompt.
7. **Conversation Update:** The user's question and the agent's response are added as new turns in `ShortTermMemory`.

### 8. Conclusion

This document provides a comprehensive guide for designing and implementing an LLM agent memory system using SQLite (Turso) and RAG. By combining short-term and long-term memory, incorporating keywords, and utilizing entity and relationship structures, you can build powerful and knowledgeable LLM agents capable of engaging in meaningful and context-aware conversations. Remember to adapt and extend this design based on the specific needs and complexities of your agent and application. 
