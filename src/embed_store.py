# src/embed_store.py
import os, pickle, json
from pathlib import Path
from typing import List
import faiss
from sentence_transformers import SentenceTransformer

# Config
EMBED_MODEL = "all-MiniLM-L6-v2"   # compact + fast
INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore", "faiss_index.pkl"))

def project_root(): return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
def chunks_file(): return os.path.join(project_root(), "data", "chunks.jsonl")

def load_chunks() -> List[dict]:
    path = Path(chunks_file())
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run src/ingest.py first.")
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_faiss_index(chunks: List[dict], persist_path: str = INDEX_PATH):
    os.makedirs(os.path.dirname(persist_path), exist_ok=True)
    encoder = SentenceTransformer(EMBED_MODEL)
    texts = [c["page_content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print("Encoding texts (this may take a bit)...")
    embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    store = {"index": index, "texts": texts, "metadatas": metadatas, "dim": dim}
    with open(persist_path, "wb") as f:
        pickle.dump(store, f)
    print("Saved FAISS index to:", persist_path)

def load_faiss_store(persist_path: str = INDEX_PATH):
    with open(persist_path, "rb") as f:
        store = pickle.load(f)
    return store

if __name__ == "__main__":
    chunks = load_chunks()
    build_faiss_index(chunks)
