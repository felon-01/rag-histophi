# src/rag_chain.py
import os
import pickle
import logging
import numpy as np
from typing import List, Optional
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIG ===
EMBED_MODEL = "all-MiniLM-L6-v2"

FAISS_PICKLE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vectorstore", "faiss_index.pkl")
)

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")  # Free
MIN_CONFIDENCE_THRESHOLD = 0.5  # Relevance threshold for results


# === LOADING FAISS ===
def load_faiss():
    """Load FAISS index with error handling."""
    if not os.path.exists(FAISS_PICKLE):
        logger.error(f"FAISS index not found at {FAISS_PICKLE}. Run ingest pipeline first.")
        raise FileNotFoundError(f"FAISS index missing: {FAISS_PICKLE}")
    try:
        with open(FAISS_PICKLE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise


# === EMBEDDING (with caching) ===
@lru_cache(maxsize=128)
def get_embedding_model() -> SentenceTransformer:
    """Get cached embedding model to avoid reloading."""
    logger.info(f"Loading embedding model: {EMBED_MODEL}")
    return SentenceTransformer(EMBED_MODEL)


def embed_text(texts: List[str]) -> np.ndarray:
    """Embed text using cached model."""
    try:
        model = get_embedding_model()
        return model.encode(texts, convert_to_numpy=True)
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise


# === RETRIEVAL (with relevance filtering & error handling) ===
def retrieve(query: str, k: int = 4) -> List[dict]:
    """Retrieve relevant chunks from vectorstore with confidence filtering."""
    try:
        store = load_faiss()
        index = store["index"]
        texts = store["texts"]
        metas = store["metadatas"]

        if not texts:
            logger.warning("Vectorstore is empty")
            return []

        # Embed the query
        q_emb = embed_text([query])

        # Search across ALL embeddings
        D, I = index.search(q_emb, min(len(texts), max(k * 5, 20)))

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
            logger.warning(f"No relevant documents found for query: {query[:50]}...")
            return []

        # Find the best-matching document (lowest avg distance)
        best_doc = min(doc_scores.keys(),
                       key=lambda d: np.mean([x[0] for x in doc_scores[d]]))

        # Sort chunks inside the chosen document
        sorted_chunks = sorted(doc_scores[best_doc], key=lambda x: x[0])
        sorted_chunks = sorted_chunks[:k]

        # Prepare results
        results = []
        for dist, idx in sorted_chunks:
            results.append({
                "text": texts[idx],
                "metadata": metas[idx],
                "distance": float(dist)  # Include relevance score
            })

        logger.info(f"Retrieved {len(results)} chunks for query from {best_doc}")
        return results
        
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
        return []




# === GENERATION (with improved prompting & validation) ===
def generate_answer(question: str, context_chunks: List[dict]) -> str:
    """Generate answer using HuggingFace Inference API with validation."""
    if not context_chunks:
        logger.warning("No context chunks provided for answer generation")
        return "I don't have sufficient information to answer this question based on my knowledge base."

    if HF_TOKEN is None:
        logger.error("Missing HuggingFace API token")
        return "❌ Configuration Error: Missing HuggingFace API TOKEN (set HUGGINGFACEHUB_API_TOKEN)."

    try:
        context = "\n\n".join(
            f"[Source: {c['metadata'].get('source','unknown')} - Chunk {c['metadata'].get('chunk', 0)}]\n{c['text']}"
            for c in context_chunks
        )

        system_prompt = (
            "You are Histophi — a factual Retrieval-Augmented-Generation assistant for history and philosophy.\n\n"
            "CORE RULES:\n"
            "1. ONLY answer using information from the provided context.\n"
            "2. If information is not in the context, respond ONLY with: \"I don't know.\"\n"
            "3. Be concise and direct. Maximum 3-4 sentences per answer.\n"
            "4. Always cite sources when providing information.\n"
            "5. Do NOT fabricate or invent facts.\n"
            "6. Do NOT include speculation or personal opinions.\n\n"
            "If you're unsure, say: \"I don't know.\" — this is better than guessing."
        )

        user_prompt = (
            f"Context Information:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer (be concise and factual):"
        )

        client = InferenceClient(
            model=HF_MODEL,
            token=HF_TOKEN
        )

        logger.info(f"Generating answer for question: {question[:50]}...")
        result = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=512,
            temperature=0.1  # Lower for more factual consistency
        )

        # Extract answer safely
        try:
            answer = result["choices"][0]["message"]["content"].strip()
            if not answer:
                logger.warning("Empty answer received from model")
                return "I couldn't generate a meaningful answer. Please rephrase your question."
            logger.info("Answer generated successfully")
            return answer
        except (KeyError, IndexError, AttributeError) as e:
            logger.error(f"Failed to extract answer from API response: {e}")
            return "Error processing the response. Please try again."
            
    except Exception as e:
        logger.error(f"Answer generation failed: {e}")
        return f"Error generating answer: {str(e)[:100]}"


# === MAIN ===
def answer_question(question: str, k: int = 4) -> dict:
    """Main entry point: retrieve chunks and generate answer with validation."""
    if not question or not question.strip():
        logger.warning("Empty question provided")
        return {
            "answer": "Please provide a valid question.",
            "sources": []
        }

    try:
        # Retrieve relevant chunks
        chunks = retrieve(question, k)
        
        if not chunks:
            logger.warning(f"No chunks retrieved for: {question}")
            return {
                "answer": "I couldn't find relevant information about your question in my knowledge base. Try rephrasing your question.",
                "sources": []
            }
        
        # Generate answer based on chunks
        answer = generate_answer(question, chunks)
        
        return {
            "answer": answer,
            "sources": chunks,
            "num_sources": len(chunks)
        }
        
    except Exception as e:
        logger.error(f"Error in answer_question: {e}")
        return {
            "answer": f"An error occurred: {str(e)[:100]}",
            "sources": [],
            "error": True
        }
