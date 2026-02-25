import redis
import hashlib
import json
import os
from typing import Any, Optional

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", None)
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_binary_client = redis.from_url(REDIS_URL, decode_responses=False)
        print(f"--- Connected to Redis via URL ---")
    else:
        redis_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True 
        )
        redis_binary_client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=False
        )
        print(f"--- Connected to Redis at {REDIS_HOST}:{REDIS_PORT} ---")
except redis.ConnectionError:
    print(f"--- WARNING: Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}. Caching will be disabled. ---")
    redis_client = None
    redis_binary_client = None

def get_hash(text: str) -> str:
    """Generate a SHA-256 hash for a given text."""
    return hashlib.sha256(text.encode()).hexdigest()

def set_cache(key: str, value: Any, expire: int = 3600):
    """Store a value in Redis with an expiration time (default 1 hour)."""
    if redis_client:
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            redis_client.set(key, value, ex=expire)
        except Exception as e:
            print(f"Redis Cache Set Error: {e}")

def get_cache(key: str) -> Optional[Any]:
    """Retrieve a value from Redis."""
    if redis_client:
        try:
            value = redis_client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
        except Exception as e:
            print(f"Redis Cache Get Error: {e}")
    return None

def set_embedding_cache(key: str, vector: list[float], expire: int = 86400):
    """Store an embedding vector in Redis (default 24 hours)."""
    if redis_binary_client:
        try:
            # We store vectors as json strings for simplicity, or binary if preferred.
            # Here we use json for compatibility with the decode_responses=True client if needed.
            redis_client.set(f"emb:{key}", json.dumps(vector), ex=expire)
        except Exception as e:
            print(f"Redis Embedding Cache Set Error: {e}")

def get_embedding_cache(key: str) -> Optional[list[float]]:
    """Retrieve an embedding vector from Redis."""
    if redis_client:
        try:
            value = redis_client.get(f"emb:{key}")
            if value:
                return json.loads(value)
        except Exception as e:
            print(f"Redis Embedding Cache Get Error: {e}")
    return None

def set_llm_cache(prompt: str, answer: str, expire: int = 3600):
    """Store LLM response keyed by full prompt hash."""
    prompt_hash = get_hash(prompt)
    set_cache(f"llm:{prompt_hash}", answer, expire=expire)

def get_llm_cache(prompt: str) -> Optional[str]:
    """Retrieve LLM response keyed by full prompt hash."""
    prompt_hash = get_hash(prompt)
    return get_cache(f"llm:{prompt_hash}")
