# ===============================
# Base Image (Lightweight, CPU)
# ===============================
FROM python:3.11-slim

LABEL maintainer="langchain_rag_agent" version="2.3"

# -------------------------------
# System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libgl1 curl build-essential \
 && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Create venv & install dependencies
# -------------------------------
WORKDIR /app

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .

RUN python -m venv /opt/venv && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip

# -------------------------------
# Copy Project Files
# -------------------------------
# Copy index.html and other root files needed
COPY index.html .
COPY .env . 

# Copy the app code
COPY app/ ./app/

# Optional: Copy data folders if they contain the vector store
# COPY data/ ./data/
# COPY chroma_langchain_db/ ./chroma_langchain_db/

# -------------------------------
# Security & Permissions
# -------------------------------
RUN groupadd -r appuser && useradd -r -g appuser appuser && \
    chown -R appuser:appuser /app

USER appuser

# -------------------------------
# Healthcheck + Port + CMD
# -------------------------------
EXPOSE 8000

CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
