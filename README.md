# LangChain RAG Agent & Chain

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Hugging Face.system designed for deep document analysis. Featuring **Hybrid Search**, **Flashrank Re-ranking**, and a **Premium Glassmorphic Interface**, it provides accurate, evidence-backed insights from your technical documentation and privacy policies.

---

## ✨ Key Features

- **🔍 Hybrid Retrieval Pipeline**: Combines semantic vector search (Chroma) with keyword matching (BM25) for maximum recall.
- **🎯 Precise Re-ranking**: Integrates **Flashrank** to refine context and ensure the most relevant chunks reach the LLM.
- **📂 Multi-Document Support**: Automatically scans and ingests all PDFs from the `data/` directory.
- **🎨 Premium UI/UX**: A modern, glassmorphic chat interface with animated backgrounds, desktop-grade micro-animations, and Lucide icons.
- **⚡ Middleware Chain**: Uses LangChain middleware (@dynamic_prompt) for ultra-fast, single-pass RAG execution.
- **🤖 ReAct Agent**: A secondary pattern implementing a Reasoning-and-Acting loop for complex multi-step queries.

---

## 📂 Project Structure

The codebase is engineered for modularity and scalability:

### 🧩 Core Modules
- **`app/retriever.py`**: The heart of the system. Implements manual hybrid search and Flashrank integration.
- **`app/loader.py`**: Batch processes PDFs using `PyPDFDirectoryLoader`.
- **`app/vectorstore.py`**: Manages the local **Chroma** persistent database.
- **`app/embeddings.py`**: Configures `all-mpnet-base-v2` for high-quality semantic vectors.
- **`app/llm.py`**: Connects to the Mistral-7B inference engine via Hugging Face.
- **`app/splitter.py`**: Handles recursive character splitting with optimized overlaps.
- **`app/tools.py`**: Defines the `retrieve_context` tool, which allows the agent to search the vector database.

### 🚀 Execution Entry Points
- **`app/chain.py`**: The primary RAG pipeline using the hybrid-rerank middleware.
- **`app/main.py`**: The standard ReAct agent implementation.
- **`app/server.py`**: FastAPI backend serving the analysis engine.
- **`index.html`**: The premium frontend application.

---

## 🛠️ Setup & Execution

### 1. Prerequisites
Create a `.env` file in the root with your credentials:
```env
HUGGINGFACE_API_KEY=hf_your_token_here
```

### 2. Ingestion
Drop your PDF documents into the `data/` folder, then run:
```powershell
python app/ingest.py
```

### 3. Start the Analysis Engine
Run the FastAPI backend:
```powershell
uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Open the Interface
Simply open `index.html` in your browser (or serve it via `python -m http.server 9000`).

---

## 🧠 Retrieval Architecture

Nexback uses a sophisticated 2-stage retrieval process:
1. **Candidate Retrieval**: Fetches top-$K$ candidates independently from **BM25** (keyword) and **Chroma** (vector).
2. **Flashrank Re-ranking**: Merges the results and uses a cross-encoder model to re-score every chunk, delivering only the most contextually relevant evidence to the LLM.

---

## 📝 Document Evidence
The system is tuned to provide transparency. Every answer is expected to include:
- **Answer**: The concise factual response.
- **Evidence**: Direct quotes from the source documents.
- **Source Reasoning**: A brief explanation of the logic applied.