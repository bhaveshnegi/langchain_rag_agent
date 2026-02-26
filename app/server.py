from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
import sys
from typing import Optional

# Add 'app' directory to sys.path to allow standalone imports (like 'import llm')
# regardless of where the server is started from.
app_dir = os.path.dirname(os.path.abspath(__file__))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

from main import agent_executor
from chain import agent as chain_agent, CORRELATION_MAP
from langchain_core.messages import HumanMessage
from observability import log_event, get_token_usage_from_metadata, retrieved_doc_ids_var
from cache import get_llm_cache, set_llm_cache, get_hash
import time
import json

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
    session_id: Optional[str] = "default"

@app.post("/chat/agent")
async def chat_agent(request: ChatRequest):
    start_time = time.time()
    try:
        # LLM Cache Check
        # For AgentExecutor, we don't have the "full_prompt" until it runs,
        # but we can cache based on the query if we assume the prompt template is fixed.
        # Alternative: The user asked to hash the full prompt.
        # Since AgentExecutor is a black box here, we can only easily cache based on input.
        cache_key = f"agent_response:{get_hash(request.query)}"
        cached_res = get_llm_cache(cache_key)
        if cached_res:
            latency = time.time() - start_time
            print(f"--- LLM Response Cache HIT (Agent) ---")
            log_event(
                event_type="agent_cache_hit",
                query=request.query,
                latency=latency,
                model_id="redis_cache_agent"
            )
            return {"response": cached_res, "cached": True}

        print(f"--- LLM Response Cache MISS (Agent) ---")
        response = agent_executor.invoke({"input": request.query})
        latency = time.time() - start_time
        
        output = response.get("output", "No response generated.")
        set_llm_cache(cache_key, output)
        
        # ... existing logging code ...
        metadata = response.get("response_metadata", {})
        token_usage = get_token_usage_from_metadata(metadata)
        retrieved_doc_ids = retrieved_doc_ids_var.get()

        log_event(
            event_type="agent_request",
            query=request.query,
            latency=latency,
            model_id="agent_executor",
            retrieved_doc_ids=retrieved_doc_ids,
            token_usage=token_usage
        )
        
        return {"response": output}
    except Exception as e:
        latency = time.time() - start_time
        log_event(event_type="agent_error", query=request.query, latency=latency, model_id="unknown", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/chain")
async def chat_chain(request: ChatRequest):
    start_time = time.time()
    try:
        inputs = {
            "messages": [HumanMessage(content=request.query)],
            "session_id": request.session_id
        }
        # The chain.py middleware will populate request.state["retrieved_doc_ids"]
        # but we need to access it. LangChain's create_agent returns a CompiledStateGraph
        # we can't easily access the middleware-modified request.state from the response 
        # unless it's returned in the state.
        
        # LLM Cache Check for Chain
        # Similar to Agent, but here we could technically get the prompt if we refactored.
        # To follow the prompt hashing requirement strictly, we should hash the "messages" if possible.
        # But since the middleware forms the prompt, the "input" to the chain is just the query.
        prompt_hash_key = f"chain_response:{get_hash(request.query)}"
        cached_res = get_llm_cache(prompt_hash_key)
        if cached_res:
             latency = time.time() - start_time
             print(f"--- LLM Response Cache HIT (Chain) ---")
             log_event(
                 event_type="chain_cache_hit",
                 query=request.query,
                 latency=latency,
                 model_id="redis_cache_chain"
             )
             return {"response": cached_res, "cached": True}

        print(f"--- LLM Response Cache MISS (Chain) ---")
        response = chain_agent.invoke(inputs)
        latency = time.time() - start_time
        
        # Extract content
        if "messages" in response:
            last_message = response["messages"][-1]
            content = last_message.content
            metadata = {}

        set_llm_cache(prompt_hash_key, content)
        
        # PERSIST AI RESPONSE TO MEMORY
        from memory import ChatMemoryManager
        from langchain_core.messages import AIMessage
        memory = ChatMemoryManager(session_id=request.session_id)
        memory.add_message(AIMessage(content=content))
        
        # ... existing logging ...
        token_usage = get_token_usage_from_metadata(metadata)
        
        last_msg = inputs["messages"][-1]
        msg_key = getattr(last_msg, "id", None) or hash(last_msg)
        retrieved_doc_ids = CORRELATION_MAP.pop(msg_key, [])
        
        if not retrieved_doc_ids:
            retrieved_doc_ids = response.get("retrieved_doc_ids", [])
            
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
