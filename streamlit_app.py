"""
ESC Evidence — Evidence-based Clinical Intelligence
Streamlit deployment of the Medical Guideline RAG Assistant.

Run locally:
    streamlit run streamlit_app.py

Deploy on Streamlit Community Cloud:
    1. Push this folder to a GitHub repo.
    2. On share.streamlit.io, create a new app pointing at streamlit_app.py.
    3. In the app's Settings -> Secrets, add:
           OPENROUTER_API_KEY = "sk-or-..."
    4. Either commit PDFs into the data/ folder, or upload them from the
       sidebar once the app is running.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
from openai import OpenAI

from rag_pipeline import (
    RAGConfig,
    RAGIndex,
    build_documents,
    rag_query,
)

# ------------------------------------------------------------------
# Page setup + light theming to match the original ESC Evidence look
# ------------------------------------------------------------------

st.set_page_config(
    page_title="ESC Evidence — Clinical Intelligence",
    page_icon="🫀",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --accent: #4ECDC4; }
    .stChatMessage { border-radius: 12px; }
    .source-card {
        border: 1px solid rgba(78, 205, 196, 0.35);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.5rem;
        background: rgba(78, 205, 196, 0.06);
    }
    .source-card .src-title { font-weight: 600; }
    .source-card .src-meta { font-size: 0.85rem; opacity: 0.75; }
    </style>
    """,
    unsafe_allow_html=True,
)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Sidebar: configuration
# ------------------------------------------------------------------

with st.sidebar:
    st.title("🫀 ESC Evidence")
    st.caption("Evidence-based Clinical Intelligence")

    st.subheader("1. API key")
    default_key = st.secrets.get("OPENROUTER_API_KEY", "")
    api_key = st.text_input(
        "OpenRouter API key",
        value=default_key,
        type="password",
        help="Stored only for this session. Prefer setting it in Streamlit secrets "
             "as OPENROUTER_API_KEY so you don't have to paste it every time.",
    )

    st.subheader("2. Source PDFs")
    bundled_pdfs = sorted(DATA_DIR.glob("*.pdf"))
    if bundled_pdfs:
        st.caption(f"Found {len(bundled_pdfs)} PDF(s) already in data/:")
        for p in bundled_pdfs:
            st.caption(f"• {p.name}")

    uploaded_files = st.file_uploader(
        "Upload additional guideline PDF(s)", type=["pdf"], accept_multiple_files=True,
    )

    with st.expander("3. Retrieval settings", expanded=False):
        top_k_final = st.slider("Final passages returned (top_k_final)", 1, 10, 4)
        candidate_k = st.slider("Candidate pool before reranking", 5, 60, 30)
        use_reranker = st.checkbox("Use cross-encoder reranker", value=True)
        model = st.selectbox(
            "Generation model (OpenRouter)",
            [
                "meta-llama/llama-3.3-70b-instruct",
                "meta-llama/llama-3.1-8b-instruct",
                "openai/gpt-4o-mini",
                "anthropic/claude-3.5-haiku",
            ],
            index=0,
        )

    build_clicked = st.button("🔧 Build / rebuild index", type="primary", use_container_width=True)

# ------------------------------------------------------------------
# Resolve the set of PDFs to index: bundled + uploaded (saved to a
# session-scoped temp folder so uploads survive reruns without re-saving).
# ------------------------------------------------------------------

upload_dir = DATA_DIR / "_uploaded"
upload_dir.mkdir(exist_ok=True)

if uploaded_files:
    for f in uploaded_files:
        dest = upload_dir / f.name
        if not dest.exists():
            dest.write_bytes(f.getbuffer())

all_pdf_paths = sorted(DATA_DIR.glob("*.pdf")) + sorted(upload_dir.glob("*.pdf"))

if not all_pdf_paths:
    st.info(
        "👋 No PDFs indexed yet. Add clinical guideline PDFs to the **data/** folder "
        "in this repo, or upload some from the sidebar, then click **Build / rebuild index**."
    )
    st.stop()

if not api_key:
    st.warning(
        "No OpenRouter API key set. Add one in the sidebar, or set OPENROUTER_API_KEY "
        "in Streamlit secrets, before asking questions."
    )

# ------------------------------------------------------------------
# Cached model + index loading
# ------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Loading reranker model...")
def load_reranker(model_name: str):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(model_name)


def _pdf_fingerprint(paths: list[Path]) -> str:
    h = hashlib.md5()
    for p in paths:
        h.update(p.name.encode())
        h.update(str(p.stat().st_mtime_ns).encode())
    return h.hexdigest()


@st.cache_resource(show_spinner="Extracting PDFs, embedding, and building the index...")
def build_index(pdf_paths: list[Path], fingerprint: str, cfg: RAGConfig) -> RAGIndex:
    documents = build_documents(pdf_paths, cfg)
    embedder = load_embedder(cfg.embed_model_name)
    reranker = load_reranker(cfg.reranker_model_name) if cfg.use_reranker else None
    return RAGIndex(documents, embedder, reranker, cfg)


config = RAGConfig(
    top_k_final=top_k_final,
    candidate_k=candidate_k,
    use_reranker=use_reranker,
    openrouter_model=model,
)

fingerprint = _pdf_fingerprint(all_pdf_paths) + str(config)

if build_clicked:
    build_index.clear()

with st.spinner("Preparing index (first run downloads the embedding + reranker models)..."):
    index = build_index(all_pdf_paths, fingerprint, config)

if not index.documents:
    st.error("No text could be extracted from the provided PDFs. Check that they contain selectable text (not scanned images).")
    st.stop()

st.sidebar.success(f"Index ready: {len(index.documents)} chunks from {len(all_pdf_paths)} PDF(s).")

# ------------------------------------------------------------------
# Chat UI
# ------------------------------------------------------------------

st.title("Medical Guideline RAG Assistant")
st.caption(
    "Ask clinical questions grounded strictly in the indexed guideline PDFs. "
    "This tool does not provide medication names and is not a substitute for professional medical advice."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role": ..., "content": ..., "sources": [...]}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📚 Evidence sources ({len(msg['sources'])})"):
                for s in msg["sources"]:
                    page = (
                        str(s["page_start"]) if s["page_start"] == s["page_end"]
                        else f'{s["page_start"]}-{s["page_end"]}'
                    )
                    score = f"{s['reranker_score']:.3f}" if s.get("reranker_score") is not None else "N/A"
                    st.markdown(
                        f"""<div class="source-card">
                        <div class="src-title">{s.get('title') or s.get('source')}</div>
                        <div class="src-meta">Source: {s.get('source')} · Page {page} · Section: {s.get('section')} · Score: {score}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

query = st.chat_input("Ask a question about the indexed guidelines...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if not api_key:
        with st.chat_message("assistant"):
            st.error("Please provide an OpenRouter API key in the sidebar first.")
        st.stop()

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    # Build short chat history for conversational context (last few turns only)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1][-6:]
    ]

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating an answer..."):
            try:
                result = rag_query(index, client, query, chat_history=history)
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"📚 Evidence sources ({len(result['sources'])})"):
                for s in result["sources"]:
                    page = (
                        str(s["page_start"]) if s["page_start"] == s["page_end"]
                        else f'{s["page_start"]}-{s["page_end"]}'
                    )
                    score = f"{s['reranker_score']:.3f}" if s.get("reranker_score") is not None else "N/A"
                    st.markdown(
                        f"""<div class="source-card">
                        <div class="src-title">{s.get('title') or s.get('source')}</div>
                        <div class="src-meta">Source: {s.get('source')} · Page {page} · Section: {s.get('section')} · Score: {score}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
