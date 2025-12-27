# 📚 Histophi - History & Philosophy RAG

A **Retrieval-Augmented-Generation (RAG)** system that answers questions about history and philosophy based on your document collection.

## ✨ Features

- **Semantic Search**: Find relevant documents using AI-powered embeddings
- **Accurate Answers**: Generate answers using context from your knowledge base
- **Source Citations**: Every answer comes with source references
- **Web Interface**: Easy-to-use Streamlit UI
- **Error Handling**: Graceful degradation with informative error messages
- **Caching**: Optimized performance with model caching

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- [HuggingFace API Token](https://huggingface.co/settings/tokens)

### 2. Setup

```bash
# Clone the repository
cd rag-histophi

# Create a virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt.txt
```

### 3. Configure Environment

```bash
# Copy the template
cp .env.example .env

# Edit .env with your HuggingFace API token
# HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### 4. Prepare Knowledge Base

```bash
# Step 1: Ingest and chunk documents
python src/ingest.py

# Step 2: Build vector index
python src/embed_store.py
```

### 5. Run the Application

```bash
streamlit run src/app.py
```

The app will open at `http://localhost:8501`

## 📁 Project Structure

```
rag-histophi/
├── src/
│   ├── app.py              # Streamlit web interface
│   ├── rag_chain.py        # Core RAG logic (retrieval + generation)
│   ├── ingest.py           # Document ingestion & chunking
│   ├── embed_store.py      # Embedding & FAISS index building
│   ├── app.css             # Styling
│   └── test_smoke.py       # Basic tests
├── data/
│   ├── *.txt              # Input documents (history/philosophy)
│   └── chunks.jsonl       # Generated chunks (created by ingest.py)
├── vectorstore/
│   └── faiss_index.pkl    # Vector embeddings index
├── requirements.txt.txt   # Python dependencies
├── .env.example           # Environment template
└── README.md
```

## 🔧 How It Works

### Pipeline

1. **Ingestion** (`ingest.py`)
   - Reads all `.txt` files from `data/` directory
   - Splits documents into overlapping chunks (500 chars, 100 char overlap)
   - Saves chunks to `data/chunks.jsonl`

2. **Embedding** (`embed_store.py`)
   - Loads chunks from JSONL file
   - Encodes chunks using `all-MiniLM-L6-v2` embeddings
   - Builds FAISS index for fast similarity search
   - Persists index to `vectorstore/faiss_index.pkl`

3. **Retrieval** (`rag_chain.py` → `retrieve()`)
   - Embeds user query
   - Searches FAISS index for most relevant chunks
   - Groups results by source document
   - Returns top-k most relevant chunks

4. **Generation** (`rag_chain.py` → `generate_answer()`)
   - Creates context from retrieved chunks
   - Sends query + context to HuggingFace model
   - Model generates factual answer from context
   - Returns answer with source references

## 🎯 Configuration

### Model Selection

Edit `.env` to use different models:

```bash
# Free tier (default, recommended for first-time setup)
HF_MODEL=HuggingFaceH4/zephyr-7b-beta

# More capable (requires acceptance)
HF_MODEL=meta-llama/Llama-2-7b-chat-hf

# Alternative options
HF_MODEL=mistralai/Mistral-7B-Instruct-v0.1
HF_MODEL=NousResearch/Nous-Hermes-2-Mistral-7B-DPO
```

### Retrieval Parameters

In the Streamlit app sidebar:
- **Number of chunks**: How many source chunks to retrieve (1-10)
- **Show distances**: Display relevance scores for debugging

### Chunk Size

Edit `src/ingest.py`:
```python
def chunk_documents(docs, chunk_size=500, chunk_overlap=100):
    # Adjust these values:
    # - chunk_size: bigger = more context, slower retrieval
    # - chunk_overlap: overlap helps preserve meaning across chunks
```

## 📊 Adding Documents

Place `.txt` files in the `data/` directory, then reingest:

```bash
python src/ingest.py      # Create new chunks
python src/embed_store.py # Rebuild index
```

**Restart the app** after rebuilding the index.

## 🐛 Troubleshooting

### "FAISS index not found"
```bash
# Rebuild the index
python src/ingest.py
python src/embed_store.py
```

### "Missing HuggingFace API TOKEN"
```bash
# Ensure .env is configured
cat .env

# Or set directly in terminal
export HUGGINGFACEHUB_API_TOKEN=your_token
```

### "No relevant sources found"
- Try rephrasing your question
- Check that documents are loaded: `python src/ingest.py`
- Verify embeddings: `python src/embed_store.py`

### Slow responses
- Reduce chunk size in `ingest.py` for faster retrieval
- Use fewer chunks (k parameter) in the sidebar
- Consider using GPU support: replace `faiss-cpu` with `faiss-gpu`

## 🔒 Privacy & Security

- **API Calls**: Only your HuggingFace token leaves your machine
- **Document Embeddings**: Stored locally in `vectorstore/`
- **No Data Tracking**: This app doesn't track or store query history

## 📈 Performance Tips

1. **Caching**: Models are cached after first load
2. **Batch Queries**: Embed multiple documents at once
3. **Index Size**: More documents = slower retrieval (normal tradeoff)
4. **GPU**: Use `faiss-gpu` for large indexes (>1M chunks)

## 🧪 Testing

Run basic smoke tests:

```bash
python src/test_smoke.py
```

## 🤝 Contributing

Ideas for improvements:
- [ ] Add query expansion for better retrieval
- [ ] Implement reranking for relevance
- [ ] Support different embedding models
- [ ] Add conversation memory
- [ ] Export answers as PDF/Markdown

## 📚 Resources

- [HuggingFace Inference API](https://huggingface.co/docs/api-inference)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Streamlit Docs](https://docs.streamlit.io/)

## 📝 License

MIT License - See LICENSE file for details

## 🙋 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Review error messages in the Streamlit console
3. Ensure all dependencies are installed: `pip install -r requirements.txt.txt`
4. Verify your `.env` configuration

---

Made with ❤️ for history and philosophy enthusiasts
