from langchain.tools import tool
from vectorstore import vector_store

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve information to help answer a query."""
    retrieved_docs = vector_store.similarity_search(query, k=2)
    
    # Track document IDs in ContextVar for observability
    from observability import retrieved_doc_ids_var
    doc_ids = []
    for doc in retrieved_docs:
        meta = doc.metadata
        doc_id = meta.get("source", "unknown")
        if "page" in meta:
            doc_id += f":page_{meta['page']}"
        doc_ids.append(doc_id)
    retrieved_doc_ids_var.set(doc_ids)

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

if __name__ == "__main__":
    # Test tool locally
    query = "What is task decomposition?"
    content, docs = retrieve_context.invoke(query)
    print(f"Retrieved {len(docs)} documents for query: '{query}'")