## Memory Tools

The Memory toolkit provides a simple way for agents to maintain short-term memory across interactions. Think of it like taking quick notes that can be referenced later.

```python
from mirascope.core import BaseToolKit, toolkit_tool

# Example usage
memory = MemoryToolkit()

# Store important information
await memory.remember("user_name", "Alice")
await memory.remember("task_goal", "Create a weekly report")

# Recall information later
user = await memory.recall("user_name")  # Returns "Alice"

# Get all memories for system prompt
all_memories = memory.format_memories()
```

The memories can be included in the system prompt to provide context for the agent's responses. This is particularly useful for maintaining conversation context or tracking task progress.
```

This simplified version makes it easier for agents to understand and use the memory system while still maintaining the core functionality needed for short-term memory in system prompts.