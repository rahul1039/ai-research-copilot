import ollama


SYSTEM_PROMPT = """
You are a strict enterprise document extraction system.

Rules:
- Answer ONLY using the provided document context.
- Do NOT use outside knowledge.
- Do NOT infer.
- Do NOT generalize.
- If the answer is not explicitly present, respond exactly:
  The document does not specify this information.
- Extract, do not synthesize.
- Be concise and structured.
"""


def generate_answer(query, retrieved_docs):

    if not retrieved_docs:
        return "The document does not specify this information."

    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
Document Context:
{context}

User Question:
{query}

Instructions:
- Extract ONLY the exact relevant information.
- If multiple items are explicitly mentioned, list them.
- If not explicitly stated, respond exactly:
  The document does not specify this information.
"""

    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 800
        }
    )

    return response["message"]["content"].strip()