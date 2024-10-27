## AI Agent Memory System Design Document

Package path (put all the implementation in this package): `src.memory`



Next Phase Implementation Plan:
Phase 5: Integration to Agent
Connect with the agent system
Implement automated improvement mechanisms


**1. Introduction**

This document outlines the design of a memory system for an AI agent, enabling it to learn from past interactions, understand complex relationships, and provide insightful responses. The system utilizes SQLite Turso as the primary storage, leverages an LLM for extraction and summarization, and incorporates a reward-based feedback mechanism for continuous improvement.

**2. System Architecture**

The memory system consists of four interconnected tables:

**2.0 Short Term Memory**

* **Purpose:** Stores the agent's short-term memory, including user information, last conversation summary, current goal, important context, and agent beliefs.
* **Storage:** SQLite Turso
* **Structure:**
    * `user_info`: Current user information (TEXT)
    * `last_conversation_summary`: Summary of the last conversation (TEXT)
    * `recent_goal_and_status`: Recent user goals and their status (TEXT)
    * `important_context`: Important contextual information (TEXT)
    * `agent_beliefs`: Agent's current beliefs about the world and the user's intentions (TEXT)
    * `agent_info`: Agent's current information (TEXT)
* **Considerations:**
    * This will be updated after each conversation.


**2.1 Conversation Store**

* **Purpose:** Stores detailed conversation data, including text, media, and files.
* **Storage:** SQLite Turso
* **Structure:**
    * `conversation_id`: Unique identifier (INTEGER PRIMARY KEY)
    * `turn_id`: the order number of the message (Integer)
    * `timestamp`: Timestamp of each interaction (DATETIME)
    * `sender`: Identifier for the sender (agent or user) (TEXT)
    * `message_content`: Text, media URL, or file path (TEXT)
    * `message_type`: Categorize the message (text, image, audio, etc.) (TEXT)
* **Considerations:**
    * Implement a mechanism to clear or archive older conversations.
    * Consider compression for large media files or external storage with references.
    * Using local file storage for media message type first (will consider to move into cloud storage)

**2.2 Knowledge**

* **Purpose:** Stores long-term knowledge extracted from various sources.
* **Storage:** SQLite Turso
* **Structure:**
    * `knowledge_id`: Unique identifier (INTEGER PRIMARY KEY)
    * `text`: The knowledge text itself (TEXT)
    * `entities`: List of relevant entities (TEXT)
    * `entity_embeddings`: Pre-computed entity embeddings (BLOB)
    * `text_embedding`: Pre-computed text embedding (BLOB)
    * `keywords`: List of extracted keywords (TEXT)
* **Considerations:**
    * Utilize the LLM for entity extraction, embedding generation, and keyword identification.
    * Consider a vector database alongside SQLite Turso for efficient similarity search. SQLite Turso also support vector search.

**2.3 Entities**

* **Purpose:** Stores relationships between entities and their embeddings.
* **Storage:** SQLite Turso
* **Structure:**
    * `relationship_id`: Unique identifier (INTEGER PRIMARY KEY)
    * `relationship_text`: Text representing the relationship (TEXT). Format "{entity_x} {relationship} {entity_y}"
    * `embedding`: Embedding representing the relationship (BLOB)
* **Considerations:**
    * Utilize the LLM for relationship extraction and embedding generation.
    * Consider a graph database for complex relationship modeling if needed.

**2.4 Conversation Summary**

* **Purpose:** Stores conversation summaries, feedback, and improvement suggestions.
* **Storage:** SQLite Turso
* **Structure:**
    * `conversation_id`: Foreign key referencing Short-Term Memory (INTEGER)
    * `prompt`: The initial prompt that started the conversation (TEXT)
    * `feedback_text`: Detailed feedback on the conversation (TEXT)
    * `example`: Example of desired agent behavior (TEXT)
    * `improvement_suggestion`: LLM-generated suggestion (TEXT)
    * `improve_prompt`: Identifier for the prompt version used (TEXT)
    * `reward_score`: Overall quality score for the conversation (REAL)
    * `conversation_summary`: Summary of the conversation (TEXT)
    * `timestamp`: Timestamp of the conversation (DATETIME)
* **Considerations:**
    * Utilize the LLM for summarization, feedback analysis, and suggestion generation.
    * Implement a mechanism for the agent to learn from feedback and adjust its behavior.
    * Define clear criteria and a method for assigning the reward score.

**3. Tech Stack Integration**

* **SQLite Turso:** Use the Turso client library for database interaction.
* **LLM:** Integrate your chosen LLM (e.g., OpenAI API) for various tasks. We use mirascope.
* **Workflow:** Design a clear workflow for data flow between the agent, LLM, and database. Consider using a message queue for asynchronous processing.

**4. Retrieval Mechanism**

* **Contextual Retrieval:** Retrieve relevant information from Knowledge and Entities tables based on user queries or conversation context. This involves understanding the user's intent and identifying relevant knowledge based on the current conversation flow.
* **Similarity Search:** Utilize embeddings and vector databases for efficient similarity-based retrieval. This involves comparing the embedding of the user query or conversation context with the embeddings stored in the Knowledge and Entities tables to find the most semantically similar entries.
* **Keyword-based Search:** Implement keyword-based search for quick retrieval of relevant knowledge entries. This involves extracting keywords from the user query and searching for knowledge entries that contain those keywords.
* 
* **Data flow examples:**
    * **User query related to some knowledge:**
        1. Perform similarity search between the user query embedding and the text embeddings of the retrieved knowledge entries. 
        2. Get entities from the user query and the top-ranked knowledge entries.
        3. Perform similarity search in the Entities table to find relationships that involve similar entities.
        4. Re-rank the knowledge entries and entity relationships based on the combined similarity scores and relevance to the user query. (Using cross-encoder re-ranker for example)
    * **User query related to some entities:**
        1. Extract entities from the user query.
        2. Perform similarity search in the Entities table to find relationships that involve similar entities.
        3. Perform similarity search search on entity embedding in the Knowledge table to retrieve knowledge entries related to the identified entities.
        4. Re-rank the entity relationships and knowledge entries based on the combined similarity scores and relevance to the user query. (Using cross-encoder re-ranker for example)



**5. Automated Improvement**

* **Prompt Refinement:** Store prompt variations, track performance, and generate new prompts based on feedback.
* **Self-Reflection:** Implement a mechanism for the agent to analyze its own performance and identify areas for self-improvement.
* **External Knowledge Integration:** Integrate external APIs or knowledge graphs to enhance the agent's knowledge base.

**6. Continuous Improvement**

* **Regularly analyze conversation summaries and feedback to identify areas for improvement.**
* **Experiment with different embedding models and techniques.**
* **Focus on user satisfaction and task completion as key metrics.**

**7. Developer Notes**

* **Database Schema:** Implement the table structures as described above in SQLite Turso.
* **LLM Integration:** Develop API calls and data processing logic for interacting with the chosen LLM.
* **Workflow Implementation:** Implement the data flow between the agent, LLM, and database using appropriate libraries and tools.
* **Retrieval Logic:** Develop efficient retrieval mechanisms for accessing relevant information from the memory system.
* **Automated Improvement Logic:** Implement algorithms and logic for prompt refinement, self-reflection, and external knowledge integration.
* **Monitoring and Evaluation:** Implement logging and monitoring tools to track agent performance and identify areas for improvement.

**8. Conclusion**

This memory system design provides a robust foundation for building an intelligent and continuously improving AI agent. By carefully implementing the components and integrating them with the chosen tech stack, developers can create an agent capable of engaging in meaningful and insightful conversations. Remember to iterate and refine the system based on real-world usage and feedback to achieve optimal performance. 
