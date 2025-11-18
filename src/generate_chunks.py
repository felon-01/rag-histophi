# src/generate_chunks.py
import os
import json

DATA_DIR = "data"
OUT_PATH = "data/chunks.jsonl"

def load_txt_files(data_dir: str):
    texts = []
    for fname in sorted(os.listdir(data_dir)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(data_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                texts.append((fname, f.read()))
    return texts

def split_text(text, chunk_size=800, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

def main():
    texts = load_txt_files(DATA_DIR)
    all_chunks = []

    for fname, content in texts:
        chunks = split_text(content)
        for i, ch in enumerate(chunks):
            all_chunks.append({
                "page_content": ch,
                "metadata": {"source": f"{fname}_chunk_{i+1}"}
            })

    # Save JSONL
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")

    print(f"✔ Created {len(all_chunks)} chunks → {OUT_PATH}")

if __name__ == "__main__":
    main()
