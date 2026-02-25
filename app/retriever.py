import os
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from vectorstore import vector_store
from loader import docs
from splitter import text_splitter

# 1. Prepare documents for BM25
all_splits = text_splitter.split_documents(docs)

# 2. Initialize BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = 10  # Retrieve more for re-ranking

# 3. Initialize Chroma Retriever
chroma_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4. Initialize Flashrank Ranker directly
ranker = Ranker()

class ManualHybridRetriever:
    def __init__(self, vector_retriever, bm25_retriever, ranker):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.ranker = ranker

    def invoke(self, query: str):
        # 1. Get docs from both sources
        v_docs = self.vector_retriever.invoke(query)
        b_docs = self.bm25_retriever.invoke(query)
        
        # 2. Combine and deduplicate
        combined_docs = []
        seen_texts = set()
        
        for doc in v_docs + b_docs:
            if doc.page_content not in seen_texts:
                combined_docs.append(doc)
                seen_texts.add(doc.page_content)
        
        # 3. Prepare for re-ranking
        passages = []
        for i, doc in enumerate(combined_docs):
            passages.append({
                "id": i,
                "text": doc.page_content,
                "meta": doc.metadata
            })
            
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # 4. Rerank
        results = self.ranker.rerank(rerank_request)
        
        # 5. Convert back to LangChain Documents (top 3)
        final_docs = []
        for res in results[:3]:
            # The result from flashrank contains the originalpassage info
            final_docs.append(Document(
                page_content=res["text"],
                metadata=res["meta"]
            ))
            
        return final_docs

# Instantiate the final retriever
final_retriever = ManualHybridRetriever(chroma_retriever, bm25_retriever, ranker)

if __name__ == "__main__":
    print("--- Testing Manual Hybrid Retriever with Re-ranking ---")
    query = "What information do you collect?"
    results = final_retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
