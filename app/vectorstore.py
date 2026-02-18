import os
from langchain_chroma import Chroma
from embeddings import embeddings

# Get absolute path to the directory where this file exists
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# The DB is located in the parent directory of 'app'
PERSIST_DIR = os.path.normpath(os.path.join(os.path.dirname(CURRENT_DIR), "chroma_langchain_db"))

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)

if __name__ == "__main__":
    print(f"Vector store initialized at: {PERSIST_DIR}")