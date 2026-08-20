# ESC Evidence — Medical Guideline RAG Assistant (Streamlit)

A Streamlit port of the `Medical_Guideline_RAG_Assistant.ipynb` notebook:
PDF extraction → section-aware chunking → hybrid (dense + BM25) retrieval →
cross-encoder reranking → grounded generation via OpenRouter, with an
Evidence Sources panel under every answer.

## What changed vs. the notebook

- **No Colab, no FastAPI/cloudflared tunnel, no hand-written HTML.** The
  whole thing is one Streamlit app (`streamlit_app.py`) plus a plain-Python
  pipeline module (`rag_pipeline.py`).
- **No ChromaDB.** Dense retrieval uses in-memory NumPy cosine similarity
  instead. For a handful of guideline PDFs this is fast and it sidesteps
  ChromaDB's SQLite version requirement, which commonly breaks on hosted
  platforms like Streamlit Community Cloud.
- Everything else — chunking, section-heading detection, BM25, Reciprocal
  Rank Fusion, the cross-encoder reranker, the system prompt (including the
  "don't name medications" rule), and citation stripping — is carried over
  as-is.

## Project layout

```
streamlit_app.py         # UI: sidebar config, chat interface, index building
rag_pipeline.py           # Pure-Python RAG pipeline (no Streamlit dependency)
requirements.txt
data/                      # Put guideline PDFs here to bundle them with the app
.streamlit/
  config.toml              # Dark/teal theme matching the original design
  secrets.toml.example     # Template for your OpenRouter key
```

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml and paste your real OPENROUTER_API_KEY
streamlit run streamlit_app.py
```

Then either drop PDFs into `data/` before starting, or upload them from the
sidebar once the app is running, and click **Build / rebuild index**.

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repo (public or private).
2. Go to https://share.streamlit.io → **New app** → point it at your repo
   and `streamlit_app.py`.
3. In the app's **Settings → Secrets**, paste:
   ```toml
   OPENROUTER_API_KEY = "sk-or-v1-..."
   ```
4. If you want PDFs bundled rather than uploaded each time, commit them into
   `data/` before deploying (GitHub has a 100 MB per-file limit — for larger
   guideline PDFs, use the in-app uploader instead, or Git LFS).
5. Deploy. First load will take a minute or two while it downloads the
   embedding model (`intfloat/multilingual-e5-base`, ~1.1 GB) and reranker
   (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90 MB) — these are cached after
   that via `@st.cache_resource`.

### A note on resources

The embedding model is fairly large. Streamlit Community Cloud's free tier
gives you ~1 GB RAM historically (community-tier limits have varied — check
your current plan), which can be tight once you add the reranker and
PyTorch. If you hit memory errors:

- Swap `EMBED_MODEL_NAME` in `rag_pipeline.py`'s `RAGConfig` to something
  lighter, e.g. `sentence-transformers/all-MiniLM-L6-v2` (drop the
  `"query: "/"passage: "` prefixing in `RAGIndex._embed` too, since that's
  an E5-specific convention).
- Turn off the reranker (`use_reranker = False` in the sidebar) — hybrid
  RRF retrieval alone still works reasonably well.
- Deploy elsewhere with more headroom instead (Render, Fly.io, a small VM,
  Hugging Face Spaces with a bigger tier, etc.) — the app doesn't depend on
  anything Streamlit-Cloud-specific.

## Other notes

- Your original PDF (`ehae178.pdf`, the 2024 ESC Hypertension Guidelines)
  wasn't in the files you uploaded here, so it isn't bundled in `data/`.
  Add it (or any other guideline PDFs) before deploying, or upload it from
  the sidebar at runtime.
- The system prompt still refuses to name specific medications/drugs — that
  behavior is unchanged from the notebook.
- Chat history is kept in `st.session_state` per browser session; it's not
  persisted anywhere, matching the original's ephemeral Colab-tunnel setup.
