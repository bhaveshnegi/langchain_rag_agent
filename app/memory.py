import json
from typing import List, Dict, Optional
from cache import redis_client, get_hash
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

class ChatMemoryManager:
    """
    Manages chat history and summaries using Redis for persistence.
    Implements sliding window for short-term memory and summarization for long-term compression.
    """
    def __init__(self, session_id: str = "default", window_size: int = 4, token_threshold: int = 1000):
        self.session_id = session_id
        self.window_size = window_size
        self.token_threshold = token_threshold
        self.history_key = f"chat_history:{session_id}"
        self.summary_key = f"chat_summary:{session_id}"

    def _serialize_message(self, message: BaseMessage) -> Dict:
        return {"type": message.type, "content": message.content}

    def _deserialize_message(self, data: Dict) -> BaseMessage:
        if data["type"] == "human":
            return HumanMessage(content=data["content"])
        elif data["type"] == "ai":
            return AIMessage(content=data["content"])
        elif data["type"] == "system":
            return SystemMessage(content=data["content"])
        return BaseMessage(content=data["content"], type=data["type"])

    def add_message(self, message: BaseMessage):
        """Add a message to the session's chat history in Redis."""
        if not redis_client:
            return
        
        serialized = json.dumps(self._serialize_message(message))
        redis_client.rpush(self.history_key, serialized)
        # Set expiry for cleanup (e.g., 24 hours)
        redis_client.expire(self.history_key, 86400)

    def get_history(self) -> List[BaseMessage]:
        """Retrieve all messages for the session."""
        if not redis_client:
            return []
        
        raw_history = redis_client.lrange(self.history_key, 0, -1)
        return [self._deserialize_message(json.loads(m)) for m in raw_history]

    def get_windowed_history(self) -> List[BaseMessage]:
        """Retrieve the last N turns (window_size) of chat history."""
        if not redis_client:
            return []
        
        # window_size is turns, so we need 2 * window_size messages (Human + AI)
        limit = self.window_size * 2
        raw_history = redis_client.lrange(self.history_key, -limit, -1)
        return [self._deserialize_message(json.loads(m)) for m in raw_history]

    def set_summary(self, summary: str):
        """Store the conversation summary."""
        if not redis_client:
            return
        redis_client.set(self.summary_key, summary, ex=86400)

    def get_summary(self) -> Optional[str]:
        """Retrieve the conversation summary."""
        if not redis_client:
            return None
        return redis_client.get(self.summary_key)

    def clear_history(self):
        """Clear the history (usually called after summarization)."""
        if redis_client:
            redis_client.delete(self.history_key)

    def estimate_tokens(self, messages: List[BaseMessage]) -> int:
        """Rough estimation of tokens in the messages."""
        text = " ".join([m.content for m in messages])
        # Rough rule of thumb: 1 token ~ 4 characters or 0.75 words
        return len(text) // 4

    def should_summarize(self) -> bool:
        """Check if history exceeds the token threshold."""
        history = self.get_history()
        return self.estimate_tokens(history) > self.token_threshold
