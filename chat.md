# Problem
Currently after the end conversation, agent will perferom reflextion as coding the the src/memory/memory_toolkit/static_flow/end_conversation.py 

The funciton is trigger at the base agent class, when the conversation is ended. But the problem that something due to the rate limit, or other issues, the reflection is not performed. 


# Solution
So i want to have a second trigger, the second trigger is at the start of the next conversation. If the reflection is not performed, then perform the reflection at the start of the next conversation, before initializing the agent.

# Implementation

1. First check if the last conversation_id (order by timestap desc) in the conversation detailed is in the conversation_summary table, or short term memory table. If not, then perform is not performed in the last conversation. We need to trigger them.
2. To trigger, we load all the last conversation detailes to the agent history, and then perform the reflection.
3. After performing reflection, we store the reflection to the conversation summary table, and short term memory table.
4. Clear the agent history. Then re initialize the agent.

