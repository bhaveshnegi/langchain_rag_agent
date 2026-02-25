import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from cache import get_embedding_cache, set_embedding_cache, get_hash

load_dotenv()

class CachedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str) -> list[float]:
        query_hash = get_hash(text)
        cached_res = get_embedding_cache(query_hash)
        if cached_res:
            print(f"--- Embedding Cache HIT ---")
            return cached_res
        
        print(f"--- Embedding Cache MISS ---")
        embedding = super().embed_query(text)
        set_embedding_cache(query_hash, embedding)
        return embedding

embeddings = CachedHuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if __name__ == "__main__":
    query = "This is a test"
    # First call (Miss)
    vec1 = embeddings.embed_query(query)
    # Second call (Hit)
    vec2 = embeddings.embed_query(query)
    
    print(f"Embeddings dimension: {len(vec1)}")
    assert vec1 == vec2
    print("Verification successful: Cache working for embeddings.")