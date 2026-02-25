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
from chain import agent as chain_agent, CORRELATION_MAP
from langchain_core.messages import HumanMessage
from observability import log_event, get_token_usage_from_metadata, retrieved_doc_ids_var
import time

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
    start_time = time.time()
    try:
        response = agent_executor.invoke({"input": request.query})
        latency = time.time() - start_time
        
        # Extract metadata and doc IDs
        metadata = response.get("response_metadata", {})
        token_usage = get_token_usage_from_metadata(metadata)
        
        # In AgentExecutor, intermediate_steps usually contains tool output
        # For simplicity, we can't easily get the doc IDs without deeper hook
        # but we can log what we have.
        
        # Get retrieved doc IDs from ContextVar (set by tool during invoke)
        retrieved_doc_ids = retrieved_doc_ids_var.get()

        log_event(
            event_type="agent_request",
            query=request.query,
            latency=latency,
            model_id="agent_executor", # Or from model.model_id if available
            retrieved_doc_ids=retrieved_doc_ids,
            token_usage=token_usage
        )
        
        return {"response": response.get("output", "No response generated.")}
    except Exception as e:
        latency = time.time() - start_time
        log_event(event_type="agent_error", query=request.query, latency=latency, model_id="unknown", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/chain")
async def chat_chain(request: ChatRequest):
    start_time = time.time()
    try:
        inputs = {"messages": [HumanMessage(content=request.query)]}
        # The chain.py middleware will populate request.state["retrieved_doc_ids"]
        # but we need to access it. LangChain's create_agent returns a CompiledStateGraph
        # we can't easily access the middleware-modified request.state from the response 
        # unless it's returned in the state.
        
        response = chain_agent.invoke(inputs)
        latency = time.time() - start_time
        
        # Extract content
        if "messages" in response:
            last_message = response["messages"][-1]
            content = last_message.content
            metadata = last_message.response_metadata
        else:
            content = str(response)
            metadata = {}

        token_usage = get_token_usage_from_metadata(metadata)
        
        # Get retrieved doc IDs using multiple methods:
        # 1. Look for correlation key in CORRELATION_MAP
        last_msg = inputs["messages"][-1]
        msg_key = getattr(last_msg, "id", None) or hash(last_msg)
        retrieved_doc_ids = CORRELATION_MAP.pop(msg_key, [])
        
        # 2. Fallback to graph state (if it starts working)
        if not retrieved_doc_ids:
            retrieved_doc_ids = response.get("retrieved_doc_ids", [])
            
        # 3. Fallback to ContextVar
        if not retrieved_doc_ids:
            retrieved_doc_ids = retrieved_doc_ids_var.get()
        
        log_event(
            event_type="chain_request",
            query=request.query,
            latency=latency,
            model_id="chain_agent",
            retrieved_doc_ids=retrieved_doc_ids,
            token_usage=token_usage
        )
            
        return {"response": content}
    except Exception as e:
        latency = time.time() - start_time
        log_event(event_type="chain_error", query=request.query, latency=latency, model_id="unknown", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def get_frontend():
    return FileResponse("../index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
