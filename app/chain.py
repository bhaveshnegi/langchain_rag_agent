import os
from llm import model
from vectorstore import vector_store
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_core.messages import HumanMessage

# Set USER_AGENT
os.environ["USER_AGENT"] = "LangChainRAGAgent/1.0"

from prompts import get_rag_prompt

@dynamic_prompt
def prompt_with_context(request: ModelRequest) -> str:
    """Inject context into state messages."""
    # In this middleware, request.state is typically the agent state
    # We look for the last message
    last_query = ""
    if "messages" in request.state and request.state["messages"]:
        last_query = request.state["messages"][-1].content
    
    if not last_query:
        return "You are a helpful assistant."

    retrieved_docs = vector_store.similarity_search(last_query, k=2)
    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

    prompt = get_rag_prompt(docs_content)
    # print("DEBUG PROMPT:", prompt) # Uncomment to see the full prompt
    return prompt

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