# src/ingest.py
import os, json
from typing import List
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Make path resolution robust
def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def data_dir() -> str:
    return os.path.join(project_root(), "data")

def load_txt_files(data_dir_path: str) -> List[dict]:
    docs = []
    p = Path(data_dir_path)
    for f in sorted(p.glob("*.txt")):
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:
            text = f.read_text(encoding="cp1252", errors="ignore")
        docs.append({"id": f.name, "text": text})
    return docs

def chunk_documents(docs: List[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for doc in docs:
        split = splitter.split_text(doc["text"])
        for i, s in enumerate(split):
            chunks.append({
                "page_content": s,
                "metadata": {"source": doc["id"], "chunk": i}
            })
    return chunks

def save_chunks_jsonl(chunks: List[dict], out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

def main():
    ddir = data_dir()
    print("Loading texts from:", ddir)
    docs = load_txt_files(ddir)
    print(f"Loaded {len(docs)} files.")
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    out = os.path.join(project_root(), "data", "chunks.jsonl")
    save_chunks_jsonl(chunks, out)
    print("Saved chunks to:", out)
    print("Now run: python src/embed_store.py  # to build FAISS index")

if __name__ == "__main__":
    main()
