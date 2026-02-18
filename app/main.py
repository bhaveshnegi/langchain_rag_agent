import os
from llm import model
from tools import retrieve_context
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub

# Set USER_AGENT for LangChain requests
os.environ["USER_AGENT"] = "LangChainRAGAgent/1.0"

def get_agent_executor():
    # 1. Define the tools
    tools = [retrieve_context]

    # 2. Define a custom ReAct prompt with grounding rules
    from prompts import AGENT_INSTRUCTIONS, AGENT_FINAL_FORMAT
    from langchain_classic.prompts import PromptTemplate

    template = (
        AGENT_INSTRUCTIONS +
        "\n\nAnswer the following questions as best you can. You have access to the following tools:\n\n"
        "{tools}\n\n"
        "Use the following format:\n\n"
        "Question: the input question you must answer\n"
        "Thought: you should always think about what to do\n"
        "Action: the action to take, should be one of [{tool_names}]\n"
        "Action Input: the input to the action\n"
        "Observation: the result of the action\n"
        "... (this Thought/Action/Action Input/Observation can repeat N times)\n"
        "Thought: I now know the final answer\n"
        "Final Answer: " + AGENT_FINAL_FORMAT + "\n"
        "\n"
        "Begin!\n\n"
        "Question: {input}\n"
        "Thought: {agent_scratchpad}"
    )

    prompt = PromptTemplate.from_template(template)

    # 3. Create the ReAct agent
    agent = create_react_agent(model, tools, prompt)

    # 4. Create the AgentExecutor
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

agent_executor = get_agent_executor()

if __name__ == "__main__":
    # 5. Run the agent
    print("--- RAG Agent Ready ---")
    query = "What information do you collect?"
    print(f"Question: {query}")
    
    try:
        response = agent_executor.invoke({"input": query})
        print("\n--- Agent Response ---")
        print(response["output"])
    except Exception as e:
        print(f"\n--- Error during execution ---")
        print(e)