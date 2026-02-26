import os
from llm import model
from vectorstore import vector_store
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage

# Set USER_AGENT
os.environ["USER_AGENT"] = "LangChainRAGAgent/1.0"

from prompts import get_rag_prompt
from cache import get_cache, set_cache, get_hash

# Global map for correlation (fallback for ContextVar scope issues)
CORRELATION_MAP = {}

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context and memory into state messages."""
    from memory import ChatMemoryManager
    
    # Extract session_id from request state (passed from server)
    session_id = request.state.get("session_id", "default")
    memory = ChatMemoryManager(session_id=session_id)
    
    last_msg = None
    if "messages" in request.state and request.state["messages"]:
        last_msg = request.state["messages"][-1]
    
    if not last_msg or not last_msg.content:
        return "You are a helpful assistant."
    
    last_query = last_msg.content

    from retriever import final_retriever
    from observability import retrieved_doc_ids_var
    retrieved_docs, doc_ids = final_retriever.invoke_with_metadata(last_query)
    
    # Store doc_ids in context variable
    retrieved_doc_ids_var.set(doc_ids)
    
    # Also store in global map using message ID or object hash as key
    # This helps when the context is lost
    msg_key = getattr(last_msg, "id", None) or hash(last_msg)
    CORRELATION_MAP[msg_key] = doc_ids
    
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # Memory Management
    # 1. Add current user message to history
    memory.add_message(last_msg)
    
    # 2. Check if we need to summarize
    if memory.should_summarize():
        full_history = memory.get_history()
        summary = model.summarize_conversation(full_history)
        memory.set_summary(summary)
        memory.clear_history()
        # Keep the latest message in history after clearing
        memory.add_message(last_msg)
    
    # 3. Get history and summary for the prompt
    windowed_history = memory.get_windowed_history()
    summary = memory.get_summary() or "None"
    
    history_str = "\n".join([f"{m.type}: {m.content}" for m in windowed_history])
    
    prompt = get_rag_prompt(docs_content, chat_history=history_str, summary=summary)
    
    # Note: AI messages are added to memory in the server after generation
    
    return prompt

# To properly cache the LLM response, we should wrap the agent invocation
# but since the user suggested "Final LLM response cache" in production pattern:
# Check LLM cache -> Check retrieval cache ...

# Create the agent with the middleware
# The new create_agent returns a graph that can be invoked directly
agent = create_agent(model, tools=[], middleware=[prompt_with_context])

if __name__ == "__main__":
    print("--- Middleware Chain (Graph) Ready ---")
    query = "What information do you collect?"
    print(f"Question: {query}")
    
    # Graphs returned by create_agent are usually invoked with a dict containing "messages"
    try:
        inputs = {"messages": [HumanMessage(content=query)]}
        # Using .invoke() on the CompiledStateGraph
        response = agent.invoke(inputs)
        
        print("\n--- Response ---")
        # In this framework, the response messages are often in the 'messages' key
        if "messages" in response:
            print(response["messages"][-1].content)
        else:
            print(response)
    except Exception as e:
        print(f"\n--- Error during execution ---")
        import traceback
        traceback.print_exc()