import json
import time
import os
from contextvars import ContextVar

# Context variables to store request-scoped metrics
retrieved_doc_ids_var: ContextVar[list] = ContextVar("retrieved_doc_ids", default=[])

def log_event(
    event_type: str,
    query: str,
    latency: float,
    model_id: str,
    retrieved_doc_ids: list = None,
    token_usage: dict = None,
    error: str = None
):
    """
    Logs a structured event in JSON format to stdout.
    """
    log_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "event_type": event_type,
        "query": query,
        "latency_seconds": round(latency, 4),
        "model_id": model_id,
        "retrieved_doc_ids": retrieved_doc_ids or [],
        "token_usage": token_usage or {},
    }
    
    if error:
        log_data["error"] = error
        
    print(json.dumps(log_data))

def get_token_usage_from_metadata(metadata: dict) -> dict:
    """
    Extracts input and output tokens from LangChain response metadata.
    Supports Bedrock and Hugging Face providers.
    """
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0
    }
    
    if not metadata:
        return usage

    # Bedrock format
    if "usage" in metadata:
        usage["input_tokens"] = metadata["usage"].get("prompt_tokens", 0)
        usage["output_tokens"] = metadata["usage"].get("completion_tokens", 0)
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    # Alternative format (sometimes seen in other providers)
    elif "token_usage" in metadata:
        usage["input_tokens"] = metadata["token_usage"].get("prompt_tokens", 0)
        usage["output_tokens"] = metadata["token_usage"].get("completion_tokens", 0)
        usage["total_tokens"] = metadata["token_usage"].get("total_tokens", 0)
        
    return usage
