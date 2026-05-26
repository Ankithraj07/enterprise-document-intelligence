# Enterprise Document Intelligence System

A RAG (Retrieval Augmented Generation) pipeline that lets you query
internal company documents using natural language.

## Tech Stack
- LangChain 0.2.16
- FAISS (vector store)
- HuggingFace Embeddings (all-MiniLM-L6-v2)
- Groq LLaMA3-70B (free API)
- Streamlit

## Features
- Upload PDF, DOCX, TXT documents
- Ask questions in natural language
- Get answers with source citations and page numbers
- Confidence scoring (High / Medium / Low)
- Conversation memory across questions
- Feedback loop that logs bad answers for improvement
- Re-index documents without touching the terminal

## Setup
1. Clone the repository
   git clone https://github.com/Ankithraj07/enterprise-document-intelligence.git

2. Create virtual environment
   python -m venv venv
   pip install -r requirements.txt

3. Add your Groq API key to .env
   GROQ_API_KEY=your_key_here

4. Drop documents into data/docs/

5. Run ingestion
   python ingest.py

6. Run the app
   streamlit run app.py

## Project Structure
- ingest.py — document loading, chunking, embedding, FAISS index creation
- rag_chain.py — retrieval chain, Groq LLM, confidence scoring, memory
- feedback_store.py — feedback logging and summary
- app.py — Streamlit UI