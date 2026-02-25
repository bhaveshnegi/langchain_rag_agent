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
    """Inject context into state messages."""
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

    prompt = get_rag_prompt(docs_content)
    
    # LLM Response Cache Check
    # We cache based on the full prompt hash because context can change
    # Note: This is a bit tricky here since we are in a 'dynamic_prompt' middleware
    # which is intended to return a prompt string, not the final answer.
    # However, we can use a trick to skip the LLM call if we want, 
    # but the middleware system might expect a prompt.
    
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