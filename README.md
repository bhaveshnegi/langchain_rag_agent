# LangChain RAG Agent & Chain

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system using LangChain. Designed for deep document analysis, it features **Multi-provider LLM support (AWS Bedrock & Hugging Face)**, **3-Layer Redis Caching**, **Persistent Smart Memory**, and **Production-Grade Observability**.

---

## ✨ Key Features

- **🔍 Hybrid Retrieval Pipeline**: Combines semantic vector search (Chroma) with keyword matching (BM25) for maximum recall.
- **🎯 Precise Re-ranking**: Integrates **Flashrank** to refine context and ensure the most relevant chunks reach the LLM.
- **☁️ Multi-Provider LLM**: Native support for **AWS Bedrock** (Mistral/Claude) and **Hugging Face** inference.
- **⚡ 3-Layer Caching**: Redis-backed caching for Embeddings, Retrieval results, and LLM completions to minimize latency and costs.
- **🧠 Smart Memory manager**: Persistent chat history with a sliding window and automatic conversation summarization.
- **📊 Production Observability**: Structured JSON logging with request-scoped metrics, token tracking, and latency profiling.
- **🛡️ Hardened Prompts**: Production-ready prompt templates with persona grounding, strict hallucination control, and few-shot examples.
- **🎨 Premium UI/UX**: A modern, glassmorphic chat interface with micro-animations and responsive design.

---

## 📂 Project Structure

### 🧩 Core Modules
- **`app/retriever.py`**: Hybrid search engine with Flashrank re-ranking logic.
- **`app/llm.py`**: Model factory supporting AWS Bedrock and Hugging Face with response caching.
- **`app/cache.py`**: Redis-based caching layer for embeddings and LLM responses.
- **`app/memory.py`**: Manages sliding window history and summarization logic.
- **`app/observability.py`**: Centralized logging and metadata extraction.
- **`app/prompts.py`**: Hardened system prompts and few-shot examples.
- **`app/vectorstore.py`**: Local Chroma DB management.
- **`app/embeddings.py`**: Cached embedding generation using `all-mpnet-base-v2`.

### 🚀 Execution Entry Points
- **`app/ingest.py`**: Batch processor for ingesting PDFs into the vector store.
- **`app/chain.py`**: Primary RAG pipeline using optimized middleware.
- **`app/server.py`**: FastAPI backend serving the RAG engine.
- **`index.html`**: Premium glassmorphic frontend.

---

## 🛠️ Setup & Execution

### 1. Prerequisites
Install Redis and create a `.env` file:
```env
# Provider (AWS or HF)
LLM=AWS

# AWS Credentials (if using AWS)
AWS_ACCESS_KEY_ID=your_id
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=ap-south-1

# Hugging Face (if using HF)
HUGGINGFACE_API_KEY=hf_...

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
```

### 2. Ingestion
Place PDFs in `data/` and run:
```powershell
python app/ingest.py
```

### 3. Start the Backend
```powershell
uvicorn app.server:app --reload
```

---

## 🧠 Advanced Architecture

### 3-Layer Caching Strategy
1. **Embedding Cache**: Hashes text chunks to skip redundant vector generation.
2. **Retrieval Cache**: Caches top-K results for identical queries.
3. **LLM Cache**: Hashes the final system prompt + history to return instant answers for repeat requests.

### Smart Memory Management
The system tracks conversation length. Once a token threshold is reached:
- The `ChatMemoryManager` invokes the LLM to generate a concise summary.
- Short-term history is cleared and replaced by the summary.
- A sliding window of the last $N$ messages is maintained for immediate context.

### Observability & Monitoring
Every request emits a structured JSON log:
```json
{
  "timestamp": "2024-...",
  "event_type": "chain_request",
  "query": "...",
  "latency_seconds": 1.2,
  "model_id": "mistral.7b-v0.2",
  "retrieved_doc_ids": ["uuid-1", "uuid-2"],
  "token_usage": {"input": 450, "output": 120}
}
```

---

## 📝 Document Evidence
The system enforces strict grounding. Every response includes:
- **Answer**: Concise factual response.
- **Evidence**: Direct quotes from the source.
- **Source Reasoning**: Logic explaining the extraction.
