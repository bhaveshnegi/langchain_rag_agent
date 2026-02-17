import os
from llm import model
from tools import retrieve_context
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub

# Set USER_AGENT for LangChain requests
os.environ["USER_AGENT"] = "LangChainRAGAgent/1.0"

def main():
    # 1. Define the tools
    tools = [retrieve_context]

    # 2. Pull a standard ReAct prompt from the hub
    # This Pull uses the langchain_classic version which is compatible with ChatHuggingFace
    prompt = hub.pull("hwchase17/react")

    # 3. Create the ReAct agent
    agent = create_react_agent(model, tools, prompt)

    # 4. Create the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # 5. Run the agent
    print("--- RAG Agent Ready ---")
    query = "What is task decomposition in the context of LLM agents?"
    print(f"Question: {query}")
    
    try:
        response = agent_executor.invoke({"input": query})
        print("\n--- Agent Response ---")
        print(response["output"])
    except Exception as e:
        print(f"\n--- Error during execution ---")
        print(e)

if __name__ == "__main__":
    main()