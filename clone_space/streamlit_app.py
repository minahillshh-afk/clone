"""Streamlit front end for Minahil's digital clone.
It reuses the retrieval (BM25 + dense + RRF) and generation code in app.py."""
import os

import streamlit as st

import app  # gradio is only imported inside app.build_ui, so it is not needed here

st.set_page_config(page_title="Minahil's AI Clone", page_icon="🤖")

# On Streamlit Community Cloud the keys live in Secrets, never in the code.
for _name in ("GROQ_API_KEY", "HF_TOKEN"):
    try:
        if _name in st.secrets:
            os.environ.setdefault(_name, st.secrets[_name])
    except Exception:
        pass


@st.cache_resource(show_spinner="Loading my notes and the embedding model...")
def get_retriever():
    return app.Retriever()


st.title("Minahil's Digital Clone")
st.error("⚠️ This is an AI clone, not the real person.")
st.write("Ask me anything about my studies, projects and work. I only answer from my own notes.")

retriever = get_retriever()

question = st.text_input("Your question", placeholder="e.g. What is thecandlescent?")
col1, col2 = st.columns([2, 1])
mode = col1.radio("Retrieval mode", app.MODES, index=2, horizontal=True)
k = col2.slider("Chunks to retrieve", 1, 5, 3)

if st.button("Ask", type="primary"):
    q = question.strip()
    if not q:
        st.warning("Please type a question first.")
    else:
        with st.spinner("Thinking..."):
            results = retriever.search(q, mode, k)
            answer = app.generate(q, results)
        st.subheader("Answer (AI clone)")
        st.write(answer)
        st.subheader("Retrieved chunks (source and score)")
        st.markdown(app.format_chunks(results, mode))