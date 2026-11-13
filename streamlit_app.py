import streamlit as st
import time
import os

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import create_or_load_vector_store

from app import retrieve_documents, generate_answer, canonicalize_models, verify_answer

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="AI Research Copilot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 AI Research Copilot")
st.write("Upload multiple PDFs and ask questions across all documents.")

# ======================================================
# SESSION STATE
# ======================================================

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False


# ======================================================
# MULTI PDF UPLOAD
# ======================================================

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    all_documents = []

    with st.spinner("Processing documents..."):

        for file in uploaded_files:

            file_path = file.name

            with open(file_path, "wb") as f:
                f.write(file.getbuffer())

            docs = load_pdf(file_path)
            all_documents.extend(docs)

        chunks = chunk_documents(all_documents)
        vector_store = create_or_load_vector_store(chunks)

        st.session_state.vector_store = vector_store
        st.session_state.docs_loaded = True

    st.success(f"Loaded {len(uploaded_files)} documents and {len(chunks)} sections.")


# ======================================================
# QUESTION INPUT
# ======================================================

if st.session_state.docs_loaded:

    query = st.text_input("Ask a question about the documents")

    if query:

        with st.spinner("Thinking..."):

            retrieved_docs = retrieve_documents(query, st.session_state.vector_store)

            answer = generate_answer(query, retrieved_docs)
            answer = canonicalize_models(answer)

            confidence = verify_answer(answer, retrieved_docs)

        st.session_state.chat_history.append((query, answer, confidence))


# ======================================================
# CHAT HISTORY
# ======================================================

if st.session_state.chat_history:

    st.divider()

    for q, a, conf in reversed(st.session_state.chat_history):

        st.markdown(f"### ❓ {q}")

        if not a:
            st.warning("The document does not specify this information.")
            continue

        st.markdown("### 📖 Answer")
        st.write(a)

        if conf > 0.75:
            st.success("Answer verified against document context.")
        elif conf > 0.5:
            st.info("Answer likely supported by the document.")
        else:
            st.warning("Low confidence. Please verify in the document.")

        st.divider()
