"""Histophi - History & Philosophy RAG Web Application"""
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load .env BEFORE importing rag_chain
load_dotenv()

import streamlit as st
from rag_chain import answer_question
from pathlib import Path

# Configure logging for this app
logger.info("Starting Histophi application...")

# --- Page Configuration ---
st.set_page_config(
    page_title="rag-histophi",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Custom CSS ---
css_path = Path(__file__).parent / "app.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)
else:
    logger.warning("CSS file not found, using defaults")

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("---")
    
    # Advanced settings
    with st.expander("Advanced Options"):
        k = st.slider("Number of source chunks (k)", 1, 10, 4,
                      help="Higher values provide more context but may include less relevant info")
        temp_info = st.checkbox("Show retrieval distances", value=False,
                               help="Display relevance scores for retrieved chunks")
    
    st.markdown("---")
    st.markdown("""
    ### About Histophi
    A Retrieval-Augmented-Generation system for history & philosophy.
    
    **Features:**
    - Semantic search across documents
    - AI-powered answers with citations
    - Source transparency
    """)

# --- UI Header ---
st.markdown("<h1 class='title'>📚 Histophi</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Accurate answers about history & philosophy, powered by RAG</p>", unsafe_allow_html=True)

# --- Input Section ---
st.markdown("### Ask a Question")
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input(
        "Your question",
        placeholder="e.g., Who was Marcus Aurelius and what did he believe?",
        label_visibility="collapsed"
    )

with col2:
    ask_button = st.button("🔍 Ask", use_container_width=True)

# --- Handle Query ---
if ask_button and query.strip():
    with st.spinner("🔄 Searching knowledge base and generating answer..."):
        try:
            result = answer_question(query, int(k))

            # Check for errors
            if result.get("error"):
                st.error(f"❌ {result['answer']}")
            else:
                # Answer Section
                st.markdown("### 📋 Answer")
                st.info(result["answer"])

                # Sources Section
                if result["sources"]:
                    st.markdown(f"### 📖 Sources ({result.get('num_sources', 0)} chunks)")
                    
                    for idx, src in enumerate(result["sources"], 1):
                        meta = src["metadata"]
                        origin = meta.get("source", "unknown")
                        chunk_num = meta.get("chunk", "?")
                        distance = src.get("distance", None)
                        
                        with st.expander(f"📄 Source {idx}: {origin} (Chunk #{chunk_num})"):
                            st.markdown(f"**Source:** `{origin}` | **Chunk:** `{chunk_num}`")
                            if temp_info and distance is not None:
                                st.markdown(f"**Relevance Score:** `{distance:.4f}`")
                            st.markdown("---")
                            st.text(src['text'][:500] + ("..." if len(src['text']) > 500 else ""))
                else:
                    st.warning("⚠️ No relevant sources found. Try rephrasing your question.")
                    
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            st.error(f"❌ An error occurred: {str(e)[:200]}")
            st.info("💡 Please check that:")
            st.markdown("""
            - HUGGINGFACEHUB_API_TOKEN is set in your environment
            - The knowledge base has been initialized (run `python src/ingest.py` and `python src/embed_store.py`)
            - Your internet connection is active
            """)

elif ask_button:
    st.warning("Please enter a question.")

# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.85em;'>
    <p>Histophi uses Retrieval-Augmented-Generation to provide accurate answers based on provided documents.</p>
    <p><em>Answers are limited to information in the knowledge base. Unknown answers will be clearly marked.</em></p>
</div>
""", unsafe_allow_html=True)

