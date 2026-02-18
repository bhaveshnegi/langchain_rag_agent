from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys

# Add 'app' directory to sys.path to allow standalone imports (like 'import llm')
# regardless of where the server is started from.
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from main import agent_executor
from chain import agent as chain_agent
from langchain_core.messages import HumanMessage

app = FastAPI(title="LangChain RAG API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    query: str

@app.post("/chat/agent")
async def chat_agent(request: ChatRequest):
    try:
        response = agent_executor.invoke({"input": request.query})
        return {"response": response.get("output", "No response generated.")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/chain")
async def chat_chain(request: ChatRequest):
    try:
        inputs = {"messages": [HumanMessage(content=request.query)]}
        response = chain_agent.invoke(inputs)
        
        # Extract content from the last message in the graph state
        if "messages" in response:
            content = response["messages"][-1].content
        else:
            content = str(response)
            
        return {"response": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def get_frontend():
    return FileResponse("../index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
