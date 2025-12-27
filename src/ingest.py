# src/ingest.py
import os
import json
import logging
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make path resolution robust
def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def data_dir() -> str:
    return os.path.join(project_root(), "data")

def load_txt_files(data_dir_path: str) -> List[dict]:
    """Load all .txt files from directory with encoding detection."""
    docs = []
    p = Path(data_dir_path)
    
    if not p.exists():
        logger.error(f"Data directory not found: {data_dir_path}")
        raise FileNotFoundError(f"Data directory missing: {data_dir_path}")
    
    txt_files = sorted(p.glob("*.txt"))
    if not txt_files:
        logger.warning(f"No .txt files found in {data_dir_path}")
        return []
    
    logger.info(f"Found {len(txt_files)} text files")
    
    for f in txt_files:
        try:
            # Try UTF-8 first, fallback to cp1252
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                logger.warning(f"UTF-8 decode failed for {f.name}, using cp1252")
                text = f.read_text(encoding="cp1252", errors="ignore")
            
            if text.strip():  # Only add non-empty files
                docs.append({"id": f.name, "text": text})
                logger.info(f"Loaded {f.name} ({len(text)} chars)")
            else:
                logger.warning(f"Skipped empty file: {f.name}")
                
        except Exception as e:
            logger.error(f"Failed to load {f.name}: {e}")
            continue
    
    return docs

def chunk_documents(docs: List[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> List[dict]:
    """Split documents into overlapping chunks."""
    logger.info(f"Splitting documents (chunk_size={chunk_size}, overlap={chunk_overlap})")
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = []
    for doc in docs:
        try:
            split = splitter.split_text(doc["text"])
            logger.info(f"{doc['id']}: {len(split)} chunks created")
            
            for i, s in enumerate(split):
                chunks.append({
                    "page_content": s,
                    "metadata": {"source": doc["id"], "chunk": i}
                })
        except Exception as e:
            logger.error(f"Failed to chunk {doc['id']}: {e}")
            continue
    
    return chunks

def save_chunks_jsonl(chunks: List[dict], out_path: str):
    """Save chunks to JSONL format with error handling."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        with open(out_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
        
        logger.info(f"✓ Saved {len(chunks)} chunks to {out_path}")
        
    except Exception as e:
        logger.error(f"Failed to save chunks: {e}")
        raise

def main():
    """Main ingestion pipeline."""
    try:
        ddir = data_dir()
        logger.info(f"Loading texts from: {ddir}")
        
        docs = load_txt_files(ddir)
        if not docs:
            logger.error("No documents loaded. Check data directory.")
            return
        
        logger.info(f"✓ Loaded {len(docs)} files")
        
        chunks = chunk_documents(docs)
        if not chunks:
            logger.error("No chunks created. Check document content.")
            return
        
        logger.info(f"✓ Created {len(chunks)} chunks")
        
        out = os.path.join(project_root(), "data", "chunks.jsonl")
        save_chunks_jsonl(chunks, out)
        
        logger.info("✓ Ingest complete!")
        logger.info("Next step: python src/embed_store.py  # Build FAISS index")
        
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
