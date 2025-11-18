# src/app.py
# src/app.py
import os
from dotenv import load_dotenv

# Load .env BEFORE importing rag_chain
load_dotenv()

import streamlit as st
from rag_chain import answer_question
from pathlib import Path


# --- Load Custom CSS ---
css_path = Path(__file__).parent / "app.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# --- Streamlit Config ---
st.set_page_config(page_title="rag-histophi", layout="wide")

# --- UI Header ---
st.markdown("<h1 class='title'>Histophi — History & Philosophy RAG</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask questions based on your ingested documents.</p>", unsafe_allow_html=True)

# Input layout
col1, col2 = st.columns([4, 1])

with col1:
    query = st.text_input("Enter your question", placeholder="e.g., Who was Marcus Aurelius?")

with col2:
    k = st.slider("Number of chunks to retrieve (k)", 1, 10, 4)


ask = st.button("Ask Histophi")

# --- Handle Query ---
if ask and query.strip():
    with st.spinner("Retrieving relevant chunks & generating answer..."):
        try:
            result = answer_question(query, int(k))

            # Answer
            st.markdown("<h2 class='section-title'>Answer</h2>", unsafe_allow_html=True)
            st.write(result["answer"])

            # Sources
            st.markdown("<h2 class='section-title'>Sources</h2>", unsafe_allow_html=True)
            for src in result["sources"]:
                meta = src["metadata"]
                origin = meta.get("source", "unknown")

                st.markdown(f"""
                <div class='chunk-box'>
                    <strong>Source:</strong> {origin}<br><br>
                    <pre>{src['text'][:450]}...</pre>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

