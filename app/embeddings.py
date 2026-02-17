import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

if __name__ == "__main__":
    test_vec = embeddings.embed_query("This is a test")
    print(f"Embeddings initialized. Vector dimension: {len(test_vec)}")