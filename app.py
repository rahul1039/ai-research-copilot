import time
import shutil
import os
import ollama

from ingestion.loader import load_pdf
from ingestion.chunker import chunk_documents
from ingestion.embedder import create_or_load_vector_store


# ======================================================
# SYSTEM PROMPT (STRICT EXTRACTION ENGINE)
# ======================================================

SYSTEM_PROMPT = """
You are a professional document intelligence assistant.

Your task is to answer questions STRICTLY using the provided document context.

RULES:
1. Use ONLY the provided document context.
2. Do NOT use outside knowledge.
3. Do NOT hallucinate or invent information.
4. If the answer is not explicitly present in the context, respond exactly with:
   The document does not specify this information.

ANSWER STYLE:
- Be concise and factual.
- If the question asks for a definition, provide the definition from the document.
- If the question asks to list items, list only items present in the context.
- Do NOT output unrelated concepts.
"""

# ======================================================
# RESET VECTOR STORE
# ======================================================

def reset_vector_store():
    if os.path.exists("faiss_index"):
        shutil.rmtree("faiss_index")


# ======================================================
# CLEAN OUTPUT
# ======================================================


def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if stripped and stripped not in cleaned:
            cleaned.append(stripped)
    return "\n".join(cleaned).strip()


# ======================================================
# CANONICALIZE MODELS
# ======================================================

def canonicalize_models(answer_text):
    if not answer_text:
        return answer_text

    lines = answer_text.split("\n")
    normalized = []
    seen = set()

    for line in lines:
        name = line.strip()
        if not name:
            continue

        lower = name.lower()

        # Canonical mapping rules
        if "linearregression" in lower:
            canonical = "Linear Regression"
        elif "sgdregressor" in lower:
            canonical = "SGDRegressor"
        elif "decision tree" in lower and "regressor" in lower:
            canonical = "Decision Tree Regressor"
        elif "random forest" in lower and "regressor" in lower:
            canonical = "Random Forest Regressor"
        else:
            canonical = name

        if canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)

    return "\n".join(normalized)


# ======================================================
# RETRIEVAL
# ======================================================

def retrieve_documents(query, vector_store):

    query_lower = query.lower()

    results = vector_store.similarity_search_with_score(query, k=20)

    if not results:
        return []

    # Sort by similarity
    results.sort(key=lambda x: x[1])

    docs = [doc for doc, _ in results]

    # Intent filtering
    if "algorithm" in query_lower or "implement" in query_lower:
        docs = [
            d for d in docs
            if "regressor" in d.page_content.lower()
            or "linear regression" in d.page_content.lower()
            or "sgdregressor" in d.page_content.lower()
        ]

    return docs[:10]


# ======================================================
# ANSWER GENERATION
# ======================================================

def generate_answer(query, retrieved_docs):

    if not retrieved_docs:
        return None

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
Document Context:
{context}

User Question:
{query}

Answer the user's question using ONLY the document context.
If the information is not present, respond exactly with:
The document does not specify this information.
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={"temperature": 0.0, "num_predict": 800}
    )

    return clean_text(response["message"]["content"])


# ======================================================
# VERIFICATION PASS
# ======================================================

def verify_answer(answer, retrieved_docs):

    if not answer or "does not specify" in answer.lower():
        return 1.0  # confident none found

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    verification_prompt = f"""
Context:
{context}

Extracted Answer:
{answer}

Check whether every extracted model name appears explicitly in the context.
Respond ONLY with a confidence score between 0 and 1.
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": verification_prompt}],
        options={"temperature": 0.0, "num_predict": 50}
    )

    try:
        score = float(response["message"]["content"].strip())
        return score
    except:
        return 0.5  # uncertain


# ======================================================
# MAIN
# ======================================================

def main():
    print("\n🚀 Professional Document Intelligence System (Phase 3)\n")

    pdf_path = input("Enter PDF file name (with .pdf extension): ").strip()

    print("\nRebuilding document index...\n")
    reset_vector_store()

    try:
        documents = load_pdf(pdf_path)
    except Exception:
        print("❌ Could not load PDF.\n")
        return

    chunks = chunk_documents(documents)
    vector_store = create_or_load_vector_store(chunks)
    print(f"✅ Loaded {len(chunks)} sections successfully!")
    print("You can now chat with your document.")
    print("Type 'exit' to quit.\n")

    while True:
        query = input("Ask a question related to the PDF: ")

        if query.lower() == "exit":
            print("\n👋 Exiting.\n")
            break

        print("\n🤖 Thinking...\n")
        time.sleep(0.3)

        retrieved_docs = retrieve_documents(query, vector_store)

        answer = generate_answer(query, retrieved_docs)
        answer = canonicalize_models(answer)

        confidence = verify_answer(answer, retrieved_docs)

        if not answer:
            print("The document does not specify this information.\n")
        elif confidence < 0.60:
            # If verification is uncertain but answer exists, still show it
            for char in answer:
                print(char, end="", flush=True)
                time.sleep(0.002)
            print("\n")
        else:
            for char in answer:
                print(char, end="", flush=True)
                time.sleep(0.002)
            print("\n")

        print("=" * 70)


if __name__ == "__main__":
    main()