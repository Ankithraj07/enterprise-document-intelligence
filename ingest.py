import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_PATH = "data/docs"
VECTORSTORE_PATH = "vectorstore/faiss_index"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

def validate_file(filepath, filename):
    size = os.path.getsize(filepath)
    if size > MAX_FILE_SIZE:
        print(f"Skipping {filename} — file too large ({size // 1024 // 1024}MB). Max is 10MB.")
        return False
    return True

def load_documents(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if not validate_file(filepath, filename):
            continue
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(filepath)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
        elif filename.endswith(".txt"):
            loader = TextLoader(filepath)
        else:
            continue
        docs = loader.load()
        for i, doc in enumerate(docs):
            doc.metadata["source"] = filename
            doc.metadata["file_type"] = filename.split(".")[-1].upper()
            doc.metadata["chunk_index"] = i
            if "page" not in doc.metadata:
                doc.metadata["page"] = f"chunk {i+1}"
        documents.extend(docs)
        print(f"Loaded: {filename} ({len(docs)} pages/chunks)")
    return documents

def get_chunk_config(documents):
    total_chars = sum(len(doc.page_content) for doc in documents)
    if total_chars < 10000:
        return 300, 30
    elif total_chars < 100000:
        return 500, 50
    else:
        return 1000, 100

def split_documents(documents):
    chunk_size, chunk_overlap = get_chunk_config(documents)
    print(f"Using chunk_size={chunk_size}, overlap={chunk_overlap} based on document size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
        
