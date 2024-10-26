# Agent-Memory Integration

Note: every agent will have a separate memory.

## 1. Dynamic Tools (Used during conversation)

**1.1 Knowledge Storage:**
* Description: Allows the agent to store new knowledge and entities encountered during a conversation into the long-term memory.
* Trigger: Agent identifies new information deemed relevant for future use (e.g., based on keywords, user feedback, or novelty detection).
* Implementation details: 
    1. Agent extracts potentially important knowledge from the current conversation turn.
    2. Agent checks if this knowledge already exists in the Knowledge table. If it exists, no action is taken.
    3. If the knowledge is new, the agent stores it in the Knowledge table.
    4. Agent extracts entities and relationships from the new knowledge and stores them in the Entity Relationship table.
* Example: User: "Did you know that the capital of France is Paris?" Agent: "I didn't know that. Thank you for the information." (Agent stores "The capital of France is Paris" in the Knowledge table and extracts "France" and "Paris" as entities with the "capital of" relationship.)

**1.2 Knowledge Search:**
* Description: Enables the agent to retrieve relevant knowledge from the long-term memory based on the current conversation context.
* Trigger: User query or agent's internal need for information (e.g., when answering a question or providing relevant context).
* Implementation details: 
    1. Agent formulates a query based on the current conversation context (e.g., using keywords, entities, or semantic embeddings).
    2. Agent performs a similarity search in the Knowledge table using the formulated query.
    3. Agent ranks the retrieved knowledge entries based on their similarity to the query and other relevance criteria (e.g., recency, user feedback).
* Example: User: "What is the capital of France?" Agent: (Performs Knowledge Search with the query "capital of France" and retrieves the knowledge entry "The capital of France is Paris.") "The capital of France is Paris."

**1.3 Entity Relationship Search:**
* Description: Allows the agent to retrieve relevant entity relationships from the long-term memory based on the current conversation context.
* Trigger: User query or agent's internal need for relationship information (e.g., when understanding connections between entities or answering questions about relationships).
* Implementation details: 
    1. Agent extracts entities from the current conversation context.
    2. Agent performs a similarity search in the Entity Relationship table using the extracted entities.
    3. Agent ranks the retrieved relationships based on their similarity to the entities and other relevance criteria (e.g., relationship type, frequency).
* Example: User: "Is Paris related to France?" Agent: (Performs Entity Relationship Search with the entities "Paris" and "France" and retrieves the relationship "Paris is the capital of France.") "Yes, Paris is the capital of France."


## 2. Static Flow 

### When conversation starts:

1. Agent will load the best system prompt from memory (conversation_summary.improve_prompt) with the highest reward score. If multiple prompts have the same highest score, the most recent one will be selected.
2. Agent will load the short term memory from memory and integrate it into the system prompt. This may involve appending relevant information from the short-term memory (e.g., user goals, salient entities) to the prompt or dynamically modifying the prompt based on the context.

### After conversation ends:


**2.1 Conversation Store:**
* Description: Stores the entire conversation history from the agent's conversation into memory.
* Trigger: End of conversation.
* Implementation details: Store message one by one by use store_conversation function in memory_manager.py.

**2.2 Conversation Analysis and Prompt Refinement:**
* Description: Summarizes the conversation, collects feedback (user-provided and/or agent-generated), performs self-reflection, identifies errors, and suggests corrections.
* Trigger: End of conversation.
* Implementation details: 
    1. Agent reflects on the entire conversation, analyzing its own actions, user responses, and the overall outcome.
    2. Agent performs self-reflection, identifying successful and unsuccessful aspects of its behavior.
    3. Agent generates feedback on its own performance, including a reward score based on predefined criteria (e.g., task completion, user satisfaction, information accuracy).
    4. Agent identifies areas for improvement and generates an improved prompt based on the analysis and feedback.
    5. Agent stores the conversation summary, feedback, reward score, and improved prompt in the conversation_summary table.

**2.3 Short Term Memory Update:**
* Description: Updates the short term memory in memory. 
* Trigger: End of conversation.
* Implementation details: 
    * **Store:** Important information to store includes:
        * Recent user goals and their status.
        * Salient entities and their relationships.
        * Important context relevant to the conversation.
        * User feedback and sentiment.
        * Agent's current beliefs about the world and the user's intentions.
        * Last conversation summary.
    * **Forget:** Information to forget includes:
        * Older conversation turns that are no longer relevant.
        * Less important context.
        * Outdated user goals.

## 3. Error Handling

* **Storage Errors:** Implement mechanisms to handle potential errors during storage operations (e.g., database connection errors, write failures). This may involve retrying the operation, logging the error, or notifying the user.
* **Retrieval Errors:** Implement mechanisms to handle potential errors during retrieval operations (e.g., database connection errors, invalid queries). This may involve returning a default value, logging the error, or notifying the user.


# Implementation Plan


# Memory Integration Plan for Base Agent

## 1. Extend BaseAgent Class

### A. Add Memory Manager
```python:src/agent/base_agent.py
class BaseAgent(ABC):
    memory_manager: MemoryManager
    conversation_id: Optional[int] = None
    agent_id: str  # Unique identifier for the agent
```

### B. Initialize Memory Components
- Add memory manager initialization in post_init method
- Set up initial short-term memory state
- Load best performing system prompt from memory

## 2. Memory Integration Points

### A. Conversation Start
1. Load best system prompt:
   - Query memory for highest reward score prompt
   - Update system_prompt attribute if exists
   
2. Load short-term memory:
   - Get latest short-term memory state
   - Integrate into system prompt building and update agent's context if exists

### B. During Conversation
1. Store each message:
   - Track conversation using conversation_id
   - Store both user and agent messages
   - Reference existing store_conversation method and reference document: ([2](https://mirascope.com/docs/learn/agents/)) 

2. Knowledge Management:
   - Add knowledge storage tool
   - Add knowledge search tool
   - Add entity relationship tools

### C. Conversation End
1. Store conversation summary:
   - Generate conversation summary
   - Perform self-reflection
   - Calculate reward score
   - Store improved prompt

2. Update short-term memory:
   - Update user information
   - Store recent goals and status
   - Update agent beliefs
   - Store important context

## 3. New Abstract Methods

```python
class BaseAgent(ABC):
    @abstractmethod
    async def generate_conversation_summary(self) -> str:
        """Generate summary of the current conversation"""
        pass
    
    @abstractmethod
    async def perform_self_reflection(self) -> Tuple[float, str, str]:
        """Return (reward_score, feedback, improved_prompt)"""
        pass
    
    @abstractmethod
    async def update_short_term_memory(self) -> None:
        """Update agent's short-term memory state"""
        pass
```

## 4. Enhanced Tool System

### A. Memory-Aware Tools
Add new base tools for memory operations:
```python
class KnowledgeStorageTool(BaseTool):
    """Store new knowledge in long-term memory."""
    text: str
    entities: List[str]
    keywords: List[str]
    relationship_text: List[str] # e.g. "Paris is the capital of France"

class KnowledgeSearchTool(BaseTool):
    """Search knowledge base"""
    query: str

class EntityRelationshipTool(BaseTool):
    """Search entity relationships"""
    entities: List[str]
```

### B. Tool Integration
Modify the existing tool handling system([2](https://mirascope.com/docs/learn/agents/)) to:
- Pass memory_manager to tools
- Track tool usage in conversation history
- Store tool outputs in knowledge base when relevant
```python
    def get_tools(self) -> List[Type[BaseTool]]:
        return [
            KnowledgeStorageTool(),
            KnowledgeSearchTool(),
            EntityRelationshipTool(),
        ]
```

## 5. Enhanced Prompt Building

Modify build_prompt method to incorporate:
- Short-term memory context
- Relevant knowledge from long-term memory
- Recent conversation history
- Current goals and status

## 6. Error Handling

Add memory-specific error handling:
- Database connection errors
- Embedding generation failures
- Storage/retrieval retries
- Fallback mechanisms
- Error as returning value for the agent. Because agent can't see the error log.

## 7. Async Support

Ensure all memory operations support async:
- Memory storage operations
- Knowledge retrieval
- Tool execution
- Summary generation

## 8. Configuration System

Add memory-specific configuration:
- Vector similarity thresholds
- Maximum history length
- Storage retention policies
- Embedding model settings

## Implementation Notes

1. The integration should maintain backward compatibility with existing agent implementations.
2. Memory operations should be non-blocking where possible.
3. Use dependency injection for memory components to allow different implementations.
4. Implement proper cleanup and resource management.
5. Add comprehensive logging for memory operations.
6. Consider implementing memory operation batching for performance.
7. Add memory state validation mechanisms.
8. Implement memory migration strategies for updates.

This plan provides a framework for integrating the memory system while maintaining the flexibility and extensibility of the base agent system. The actual implementation should be done incrementally, with thorough testing at each stage.
