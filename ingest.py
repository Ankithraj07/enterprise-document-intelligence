import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_cha

load_dotenv()

DOCS_PATH = "data/docs"
VECTORSTORE_PATH = "vectorstore/faiss_index"

def load_documents(folder_path):
    """Load all PDFs, DOCX, and TXT files from a folder."""
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        docs = loader.load()
        # Tag each doc with its source filename — used for citations later
        for doc in docs:
            doc.metadata["source"] = filename
        documents.extend(docs)
        print(f"Loaded: {filename} ({len(docs)} pages/chunks)")
    return documents

def split_documents(documents):
    """Split into chunks. chunk_size and overlap are tunable."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,         # overlap prevents losing context at chunk boundaries
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} pages")
    return chunks

def build_vectorstore(chunks):
    """Embed chunks and store in FAISS."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}   # laptop-safe, no GPU needed
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")
    return vectorstore

if __name__ == "__main__":
    print("Starting ingestion...")
    docs = load_documents(DOCS_PATH)
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("Done. You can now run app.py") 
        
