import streamlit as st
import os
from rag_chain import load_qa_chain, get_answer
from feedback_store import log_feedback, get_feedback_summary

st.set_page_config(
    page_title = "Document Intelligence",
    page_icon = "📚",
    layout = "wide"
)

st.title("📚Enterprise Document Intelligence")
st.caption("Ask question about your company Documents. Answer include source citations and confidence scores.")

# -- Cache the chain AND vectorstore together --
@st.cache_resource
def get_chain_and_store():
    return load_qa_chain()    # returns (chain, vectorstore)

# --- Guard: check vector store exists ---
if not os.path.exists("vectorstore/faiss_index"):
    st.error("No vector store found. Run `python ingest.py` first to index your documents.")
    st.stop()

chain, vectorstore = get_chain_and_store()

# --- Sidebar ---
with st.sidebar:
    st.header("Indexed documents")
    doc_folder = "data/docs"
    if os.path.exists(doc_folder):
        files = [f for f in os.listdir(doc_folder) if f.endswith(('.pdf', '.docx', '.txt'))]
        if files:
            for f in files:
                st.write(f"📎 {f}")
        else:
            st.caption("No documents found in data/docs/")

    st.divider()

    # Re-index button (Differentiator 3)
    if st.button("🔄 Re-index documents"):
        with st.spinner("Indexing... this may take a minute."):
            import subprocess
            import sys # <-- Import sys package to dynamically find the running path
            
            # FIXED: sys.executable points explicitly to your active environment's python.exe
            result = subprocess.run(
                [sys.executable, "ingest.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.cache_resource.clear()
                st.success("Index updated. Please reload the page.")
            else:
                st.error(f"Indexing failed:\n{result.stderr}")

    st.divider()

    # Feedback summary
    st.subheader("Feedback summary")
    summary = get_feedback_summary()
    if summary["total"] == 0:
        st.caption("No feedback submitted yet.")
    else:
        st.metric("Total responses rated", summary["total"])
        st.metric("Satisfaction rate", summary["rate"])
        st.caption(f"👍 {summary['positive']}   👎 {summary['negative']}")

    st.divider()

    # Clear conversation memory button
    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = []
        # Reload chain to reset LangChain's internal memory too
        st.cache_resource.clear()
        st.success("Conversation cleared.")

    st.divider()
    st.caption("Stack: LangChain · FAISS · HuggingFace · Groq LLaMA3")

# --- Session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_comment_box" not in st.session_state:
    st.session_state.show_comment_box = False

# --- Main query input ---
question = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is the leave policy for new employees?"
)

if st.button("Get answer", type="primary") and question.strip():

    with st.spinner("Searching documents..."):
        # get_answer now returns 4 values including confidence
        answer, sources, confidence, score = get_answer(chain, vectorstore, question)

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "score": score
    })
    # Reset comment box when new question is asked
    st.session_state.show_comment_box = False

# --- Display latest answer ---
if st.session_state.chat_history:
    latest = st.session_state.chat_history[-1]

    st.subheader("Answer")

    # Confidence badge (Differentiator 1)
    confidence = latest["confidence"]
    score = latest["score"]

    if confidence == "High":
        st.success(f"Confidence: {confidence} (similarity score: {score}) — answer is well-supported by documents.")
    elif confidence == "Medium":
        st.warning(f"Confidence: {confidence} (similarity score: {score}) — answer may be partially supported.")
    else:
        st.error(f"Confidence: {confidence} (similarity score: {score}) — relevant content not found. Answer may be unreliable.")

    st.write(latest["answer"])

    # Source citations
    if latest["sources"]:
        st.subheader("Sources")
        for s in latest["sources"]:
            with st.expander(f"📄 {s['file']} — page {s['page']}"):
                st.caption("Relevant excerpt:")
                st.write(f"...{s['snippet']}...")
    else:
        st.info("No source chunks returned.")

    # Feedback
    st.divider()
    st.write("Was this answer helpful?")
    col1, col2 = st.columns([1, 6])

    with col1:
        if st.button("👍 Yes"):
            log_feedback(
                question=latest["question"],
                answer=latest["answer"],
                sources=latest["sources"],
                rating="positive"
            )
            st.success("Thanks!")

    with col2:
        if st.button("👎 No — report issue"):
            st.session_state.show_comment_box = True

    if st.session_state.show_comment_box:
        comment = st.text_input("What was wrong? (optional)")
        if st.button("Submit report"):
            log_feedback(
                question=latest["question"],
                answer=latest["answer"],
                sources=latest["sources"],
                rating="negative",
                comment=comment
            )
            st.warning("Logged. This helps improve the pipeline.")
            st.session_state.show_comment_box = False

# --- Conversation history (Differentiator 2 — visible memory) ---
if len(st.session_state.chat_history) > 1:
    st.divider()
    st.subheader("Conversation history")
    st.caption("LangChain memory keeps context across all questions below.")

    for entry in reversed(st.session_state.chat_history[:-1]):
        with st.expander(f"Q: {entry['question']}"):
            # Confidence badge in history too
            c = entry["confidence"]
            badge = "🟢" if c == "High" else "🟡" if c == "Medium" else "🔴"
            st.caption(f"{badge} Confidence: {c}  |  Score: {entry['score']}")
            st.write(entry["answer"])
            if entry["sources"]:
                st.caption("Sources: " + ", ".join(
                    f"{s['file']} p.{s['page']}" for s in entry["sources"]
                ))