import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

VECTORSTORE_PATH = "vectorstore/faiss_index"

PROMPT_TEMPLATE = """
You are an intelligent assistant helping employees find information from company documents.

Use ONLY the context provided below to answer the question.
If the answer is not in the context, say: "I could not find this information in the provided documents."
Always mention which document your answer came from.
Context.
{context}

Chat history:
{chat_history}

Question.
{question}

Answer (include document name as source):
"""

def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs = {"device":"cpu"}
    )
    
def load_vectorstore(embeddings):
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
def load_qa_chain():
    """
    Builds the full ConversationalRetrievalChain with memory.
    Returns both the chain and the vectorstore (needed for confidence scoring).
    """
    embeddings = load_embeddings()
    vectorstore = load_vectorstore(embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # FIXED: Replaced decommissioned 'llama3-70b-8192' with active 'llama-3.3-70b-versatile'
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

    # Memory keeps track of the full conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"         # must match chain's output key
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    return chain, vectorstore
    
def get_confidence(vectorstore , question):
    """
    Uses FAISS L2 distance scores to estimate how relevant
    the retrieved chunks are to the question.
    Lower L2 distance = chunks are closer to the question = higher confidence.
    """
    
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=4)
    
    if not docs_with_scores:
        return "Low", 999.0
    
    avg_score = sum(score for _, score in docs_with_scores) / len(docs_with_scores)
    
    # Adjusted thresholds based on real L2 distance ranges
    if avg_score < 1.0:
        confidence = "High"
    elif avg_score < 1.5:
        confidence = "Medium"
    else:
        confidence = "Low"
        
    return confidence, round(avg_score, 3)

def get_answer(chain , vectorstore , question):
    """ 
    Run a question through the chain.
    Returns answer, sources, confidence level, and raw score.
    """
    
    # Confidence check — done before chain call so we can warn user early
    confidence , score = get_confidence(vectorstore , question)
    
    # Chain call - memory is handled internally by ConversationalRetrievalChain
    result = chain.invoke({"question": question})
    
    answer = result["answer"]
    source_docs = result.get("source_documents" , [])
    
    # Deduplicate and format Source
    sources = []
    seen = set()
    for doc in source_docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        key = f"{source}-{page}"
        if key not in seen:
            sources.append({
                "file": source,
                "page": page,
                "snippet": doc.page_content[:150]
            })
            seen.add(key)
    
    return answer , sources , confidence , score



    

    

        
        

    

    

    


