import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import agent_executor

def test_agent():
    query = "What information do you collect?"
    print(f"Testing agent with query: {query}")
    try:
        response = agent_executor.invoke({"input": query})
        print("\n--- Response ---")
        print(response.get("output"))
    except Exception as e:
        print("\n--- ERROR ---")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agent()
