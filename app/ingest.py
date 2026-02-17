from vectorstore import vector_store
from splitter import all_splits

def run_ingestion():
    document_ids = vector_store.add_documents(documents=all_splits)
    print(f"Ingested {len(document_ids)} documents.")
    print(f"First 3 document IDs: {document_ids[:3]}")
    return document_ids

if __name__ == "__main__":
    run_ingestion()