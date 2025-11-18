# src/rag_chain.py
import os
import pickle
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from sentence_transformers import CrossEncoder


# === CONFIG ===
EMBED_MODEL = "all-MiniLM-L6-v2"

FAISS_PICKLE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vectorstore", "faiss_index.pkl")
)

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")  # Free


# === LOADING FAISS ===
def load_faiss():
    with open(FAISS_PICKLE, "rb") as f:
        return pickle.load(f)


# === EMBEDDING ===
def embed_text(texts: List[str]) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL)
    return model.encode(texts, convert_to_numpy=True)


# === RETRIEVAL ===
# === RETRIEVAL (with relevance filtering) ===
import numpy as np

def retrieve(query: str, k: int = 4) -> List[dict]:
    store = load_faiss()
    index = store["index"]
    texts = store["texts"]
    metas = store["metadatas"]

    # Embed the query
    q_emb = embed_text([query])

    # Search across ALL embeddings
    D, I = index.search(q_emb, len(texts))  # we retrieve everything

    # Group chunks by source document
    doc_scores = {}
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(texts):
            continue
        doc_name = metas[idx]["source"]
        if doc_name not in doc_scores:
            doc_scores[doc_name] = []
        doc_scores[doc_name].append((dist, idx))

    if not doc_scores:
        return []  # failsafe

    # Find the best-matching document (lowest avg distance)
    best_doc = min(doc_scores.keys(),
                   key=lambda d: np.mean([x[0] for x in doc_scores[d]]))

    # Sort chunks inside the chosen document
    sorted_chunks = sorted(doc_scores[best_doc], key=lambda x: x[0])

    # IMPORTANT:
    # If the document has fewer than k chunks, we return all available
    sorted_chunks = sorted_chunks[:k]

    # Prepare results
    results = []
    for _, idx in sorted_chunks:
        results.append({
            "text": texts[idx],
            "metadata": metas[idx]
        })

    return results




# === GENERATION (NEW HF API: chat_completion) ===
def generate_answer(question: str, context_chunks: List[dict]):
    if HF_TOKEN is None:
        return "❌ Missing HuggingFace API TOKEN (HUGGINGFACEHUB_API_TOKEN)."

    context = "\n\n".join(
        f"[Source: {c['metadata'].get('source','unknown')}]\n{c['text']}"
        for c in context_chunks
    )

    system_prompt = (
    "You are Histophi — an accurate Retrieval-Augmented-Generation assistant.\n"
    "RULES:\n"
    "1. You MUST answer ONLY using the context provided.\n"
    '2. If the answer is NOT in the context, respond ONLY with: "I don\'t know."\n'
    "3. Do NOT reuse or summarize earlier questions in this conversation.\n"
    "4. Do NOT invent follow-up questions or answers.\n"
    "5. Do NOT include information not explicitly found in the context.\n"
    "6. Your answer must be direct, short, and factual."
)



    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

    client = InferenceClient(
        model=HF_MODEL,
        token=HF_TOKEN
    )

    result = client.chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=256,
        temperature=0.0
    )

    # HF returns -> {"choices":[{"message":{"content": "..."} }]}
    try:
        return result["choices"][0]["message"]["content"]
    except:
        return str(result)


# === MAIN ===
def answer_question(question: str, k: int = 4):
    chunks = retrieve(question, k)
    answer = generate_answer(question, chunks)
    return {"answer": answer, "sources": chunks}
