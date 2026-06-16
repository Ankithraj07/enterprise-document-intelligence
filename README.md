---
title: Enterprise Document Intelligence
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Enterprise Document Intelligence System

RAG pipeline for querying internal company documents using natural language.

🔗 **Live Demo:** https://huggingface.co/spaces/Ankith07/enterprise-document-intelligence

---

## How It Works

```
User uploads PDF / DOCX / TXT
           ↓
Document chunking + HuggingFace embeddings (all-MiniLM-L6-v2)
           ↓
FAISS vector store (indexed locally)
           ↓
User asks a question
           ↓
Semantic search → retrieve top-k relevant chunks
           ↓
Groq LLaMA3-70B generates answer with context
           ↓
Response with source citations + confidence score
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| LLM | Groq LLaMA3-70B (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Vector Store | FAISS |
| Framework | LangChain 0.2.16 |
| Frontend | Streamlit |
| Containerization | Docker |

---

## Features

- Upload PDF, DOCX, TXT documents
- Natural language querying with conversation memory
- Source citations with page numbers
- Confidence scoring — High / Medium / Low via L2 distance
- Feedback loop that logs bad answers for improvement
- Re-index documents without touching terminal
- Fully Dockerized

---

## Run Locally

### Without Docker
```bash
git clone https://github.com/Ankithraj07/enterprise-document-intelligence.git
cd enterprise-document-intelligence
python -m venv venv
pip install -r requirements.txt
# Add GROQ_API_KEY=your_key_here to .env
python ingest.py
streamlit run app.py
```

### With Docker
```bash
docker-compose up --build
```

---

## Project Structure

| File | Purpose |
|------|---------|
| ingest.py | Document loading, chunking, embedding, FAISS indexing |
| rag_chain.py | RAG chain, LLM integration, confidence scoring, memory |
| feedback_store.py | Feedback logging and summary |
| app.py | Streamlit UI |
| Dockerfile | Container setup |
| docker-compose.yml | Multi-service orchestration |

---

## Two Pipelines

- **Ingestion pipeline** — runs once when documents are added
- **Query pipeline** — runs on every user question