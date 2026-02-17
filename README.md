# LangChain RAG Agent & Chain

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Hugging Face. It demonstrates two distinct architectural patterns for building RAG applications: a **ReAct Agent** and a **Middleware Chain**.

## 📂 Project Structure

The codebase is modularized for clarity and reusability:

### Core Modules
- **`llm.py`**: Initializes the Large Language Model (Mistral-7B) via the Hugging Face Serverless Inference API. It automatically loads API tokens from the `.env` file.
- **`embeddings.py`**: Configures the `sentence-transformers/all-mpnet-base-v2` model used to convert text into numerical vectors.
- **`loader.py`**: Handles web scraping. It uses `WebBaseLoader` to fetch specific content from a blog post.
- **`splitter.py`**: Breaks down large documents into smaller, overlapping chunks (1000 characters) to ensure context fits within LLM limits.
- **`vectorstore.py`**: Sets up the local **Chroma** database where text chunks and their embeddings are stored.
- **`ingest.py`**: The setup script. Run this to populate your local database with the blog post data.
- **`tools.py`**: Defines the `retrieve_context` tool, which allows the agent to search the vector database.

### Implementation Patterns
- **`main.py` (ReAct Agent)**: Implements a "Reasoning and Acting" (ReAct) flow. The AI agent uses tools to decide when and what to search for in the database before answering.
- **`chain.py` (Middleware Chain)**: Explores a more direct RAG pattern using **Middleware**. It uses `@dynamic_prompt` to automatically inject retrieved context into the system prompt *before* the model is called, skipping the agent loops.

---

## 🚀 How It Works

### 1. Ingestion Phase
When you run `ingest.py`, the following happens:
1. `loader.py` fetches the blog post html.
2. `splitter.py` divides the text into 63 sub-documents.
3. `vectorstore.py` (via `embeddings.py`) converts these chunks into vectors.
4. The vectors are saved to the `./chroma_langchain_db` folder.

### 2. Retrieval Phase
- **In `main.py`**: The agent is provided with the `retrieve_context` tool. When asked a question, it chooses to "retrieve" information if its internal knowledge is insufficient.
- **In `chain.py`**: The system is more "proactive." Every query triggers a similarity search in the background, and the results are appended to your instructions automatically via middleware.

---

## 🛠️ Setup & Execution

### Prerequisites
1. Create a `.env` file in the root directory with your Hugging Face API key:
   ```env
   HUGGINGFACE_API_KEY=hf_your_token_here
   ```
2. Ensure you have the virtual environment activated.

### Running the Project
1. **Populate the database**:
   ```powershell
   python ingest.py
   ```
2. **Run the ReAct Agent**:
   ```powershell
   python main.py
   ```
3. **Run the Middleware Chain**:
   ```powershell
   python chain.py
   ```

---

## 🧠 Why Two Versions?
- **The Agent (`main.py`)** is best for complex tasks where the AI might need to search multiple times or decide *not* to search.
- **The Chain (`chain.py`)** is faster and more reliable for standard Q&A tasks where you always want to provide context to the AI.