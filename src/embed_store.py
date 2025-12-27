# src/embed_store.py
import os
import pickle
import json
import logging
from pathlib import Path
from typing import List

import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
EMBED_MODEL = "all-MiniLM-L6-v2"   # compact + fast
INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectorstore", "faiss_index.pkl"))

def project_root(): 
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def chunks_file(): 
    return os.path.join(project_root(), "data", "chunks.jsonl")

def load_chunks() -> List[dict]:
    """Load chunks from JSONL with error handling."""
    path = Path(chunks_file())
    if not path.exists():
        logger.error(f"Chunks file not found: {path}")
        raise FileNotFoundError(f"{path} not found. Run src/ingest.py first.")
    
    chunks = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipped invalid JSON at line {i}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Failed to load chunks: {e}")
        raise
    
    if not chunks:
        raise ValueError("No chunks found in file")
    logger.info(f"Loaded {len(chunks)} chunks")
    return chunks

def build_faiss_index(chunks: List[dict], persist_path: str = INDEX_PATH):
    """Build FAISS index from chunks with error handling."""
    if not chunks:
        raise ValueError("Cannot build index with empty chunks")
    
    try:
        os.makedirs(os.path.dirname(persist_path), exist_ok=True)
        
        logger.info(f"Loading embedding model: {EMBED_MODEL}")
        encoder = SentenceTransformer(EMBED_MODEL)
        
        texts = [c["page_content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        logger.info(f"Encoding {len(texts)} texts (this may take a moment)...")
        embeddings = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        logger.info(f"Creating FAISS index with dimension {embeddings.shape[1]}")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        store = {"index": index, "texts": texts, "metadatas": metadatas, "dim": dim}
        
        logger.info(f"Persisting index to {persist_path}")
        with open(persist_path, "wb") as f:
            pickle.dump(store, f)
        
        logger.info(f"✓ Successfully built FAISS index with {len(texts)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to build FAISS index: {e}")
        raise

def load_faiss_store(persist_path: str = INDEX_PATH):
    """Load FAISS store with validation."""
    if not os.path.exists(persist_path):
        raise FileNotFoundError(f"FAISS index not found: {persist_path}")
    
    try:
        with open(persist_path, "rb") as f:
            store = pickle.load(f)
        logger.info(f"Loaded FAISS index with {len(store['texts'])} chunks")
        return store
    except Exception as e:
        logger.error(f"Failed to load FAISS store: {e}")
        raise

if __name__ == "__main__":
    try:
        logger.info("Starting FAISS index build...")
        chunks = load_chunks()
        build_faiss_index(chunks)
        logger.info("✓ FAISS index built successfully!")
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        exit(1)
