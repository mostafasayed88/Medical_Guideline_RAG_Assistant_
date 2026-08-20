"""
Core RAG pipeline: PDF extraction -> chunking -> hybrid (dense + BM25) retrieval
-> cross-encoder reranking -> LLM generation via OpenRouter.

This is a direct port of the logic from Medical_Guideline_RAG_Assistant.ipynb,
refactored into plain functions/classes with no Colab or notebook dependencies,
so it can run inside a Streamlit app (or anywhere else).

Difference from the notebook: ChromaDB is replaced with a simple in-memory
NumPy cosine-similarity search. For a corpus of this size (a handful of PDFs)
that's plenty fast, and it avoids ChromaDB's SQLite version requirements,
which commonly break on hosted platforms like Streamlit Community Cloud.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ============================================================
# Config
# ============================================================


@dataclass
class RAGConfig:
    chunk_size: int = 1200
    chunk_overlap: int = 300

    top_k_dense: int = 15
    top_k_sparse: int = 15
    candidate_k: int = 30
    top_k_final: int = 4
    use_reranker: bool = True

    embed_model_name: str = "intfloat/multilingual-e5-base"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    max_context_chars: int = 12000

    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct"


# ============================================================
# 1. PDF extraction (section-aware)
# ============================================================

SECTION_KEYWORDS = {
    "abstract", "introduction", "background", "methods", "materials and methods",
    "results", "discussion", "conclusion", "conclusions", "recommendations",
    "references", "acknowledgements", "acknowledgments", "limitations", "summary",
    "objectives", "diagnosis", "treatment", "management", "epidemiology",
    "definitions", "clinical presentation", "risk factors", "screening",
    "follow-up", "evidence", "appendix", "key points", "gaps in evidence",
}
_HEADING_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9\s\-/,&()]{2,80}$")
_HEADING_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z0-9\s\-/,&()]{2,60}$")

# Class of Recommendation (I, IIa, IIb, III) / Level of Evidence (A, B, C, D)
# table labels -- e.g. "I A", "IIa B", "III C" -- look like all-caps headings
# but are actually table cells and must be excluded explicitly.
_COR_TOKENS = {"I", "II", "III", "IIA", "IIB"}
_LOE_TOKENS = {"A", "B", "C", "D"}


def is_cor_loe_label(line: str) -> bool:
    tokens = line.strip().split()
    if not tokens or len(tokens) > 2:
        return False
    return all(t.upper() in _COR_TOKENS or t.upper() in _LOE_TOKENS for t in tokens)


def looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 90:
        return False
    if is_cor_loe_label(line):
        return False
    if _HEADING_NUMBERED_RE.match(line):
        return True
    words = line.split()
    if _HEADING_ALLCAPS_RE.match(line) and 2 <= len(words) <= 8:
        return True
    if line.lower().strip(" .:") in SECTION_KEYWORDS:
        return True
    return False


def make_doc_id(filename: str) -> str:
    return hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]


def get_title(reader: PdfReader, filename: str) -> str:
    try:
        meta_title = (reader.metadata.title or "").strip() if reader.metadata else ""
    except Exception:
        meta_title = ""
    if meta_title:
        return meta_title
    stem = Path(filename).stem
    stem = re.sub(r"[_\-]+", " ", stem).strip()
    return stem.title() if stem else filename


def extract_section_segments(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    doc_id = make_doc_id(pdf_path.name)
    title = get_title(reader, pdf_path.name)
    segments = []
    current_section = "Unknown section"

    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        buf = []
        for line in raw.split("\n"):
            if looks_like_heading(line):
                if buf:
                    text = "\n".join(buf).strip()
                    if text:
                        segments.append({
                            "source": pdf_path.name, "doc_id": doc_id, "title": title,
                            "page": i, "section": current_section, "text": text,
                        })
                    buf = []
                current_section = line.strip()
                continue
            buf.append(line)

        if buf:
            text = "\n".join(buf).strip()
            if text:
                segments.append({
                    "source": pdf_path.name, "doc_id": doc_id, "title": title,
                    "page": i, "section": current_section, "text": text,
                })
    return segments


def merge_into_sections(segments: list[dict]) -> list[dict]:
    blocks = []
    for seg in segments:
        if (blocks and blocks[-1]["source"] == seg["source"]
                and blocks[-1]["section"] == seg["section"]):
            blocks[-1]["text"] += " " + seg["text"]
            blocks[-1]["page_end"] = seg["page"]
        else:
            blocks.append({
                "source": seg["source"], "doc_id": seg["doc_id"], "title": seg["title"],
                "section": seg["section"], "page_start": seg["page"],
                "page_end": seg["page"], "text": seg["text"],
            })
    return blocks


# ============================================================
# 2. Clean + chunk
# ============================================================


def clean_text(text: str) -> str:
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def build_documents(pdf_paths: list[Path], config: RAGConfig) -> list[dict]:
    all_segments = []
    for fpath in pdf_paths:
        if not fpath.exists():
            continue
        all_segments.extend(extract_section_segments(fpath))

    if not all_segments:
        return []

    all_sections = merge_into_sections(all_segments)

    documents = []
    doc_id_counter = 0
    for section_index, sec in enumerate(all_sections):
        cleaned = clean_text(sec["text"])
        section_chunks = chunk_text(cleaned, config.chunk_size, config.chunk_overlap)
        for chunk_index, c in enumerate(section_chunks):
            documents.append({
                "id": str(doc_id_counter),
                "source": sec["source"],
                "doc_id": sec.get("doc_id", ""),
                "title": sec.get("title", sec["source"]),
                "page": sec["page_start"],
                "page_start": sec["page_start"],
                "page_end": sec["page_end"],
                "section": sec.get("section", "Unknown section"),
                "section_index": section_index,
                "chunk_index": chunk_index,
                "chunks_in_section": len(section_chunks),
                "text": c,
            })
            doc_id_counter += 1
    return documents


# ============================================================
# 3. Answer citation-stripping
# ============================================================

_CITATION_PAREN_RE = re.compile(r"\s*\([^()]*\.pdf[^()]*\)", re.IGNORECASE)
_CITATION_BRACKET_RE = re.compile(r"\s*\[[^\[\]]*\.pdf[^\[\]]*\]", re.IGNORECASE)
_CITATION_BARE_RE = re.compile(
    r"\s*[\w\-]+\.pdf(?:\s*,\s*(?:p\.?|page|pages|section)[^.,;\n]*)*",
    re.IGNORECASE,
)


def clean_generated_answer(text: str) -> str:
    if not text:
        return text
    cleaned = _CITATION_PAREN_RE.sub("", text)
    cleaned = _CITATION_BRACKET_RE.sub("", cleaned)
    cleaned = _CITATION_BARE_RE.sub("", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n[ \t]+", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ============================================================
# 4. The index: embeddings + BM25, wrapped in one object
# ============================================================


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class RAGIndex:
    """Holds the built corpus, dense embeddings, and BM25 index for one set of PDFs."""

    def __init__(self, documents: list[dict], embedder: SentenceTransformer,
                 reranker: Optional[CrossEncoder], config: RAGConfig):
        self.documents = documents
        self.embedder = embedder
        self.reranker = reranker
        self.config = config

        self.id_to_idx = {d["id"]: i for i, d in enumerate(documents)}

        if documents:
            texts = [d["text"] for d in documents]
            self.corpus_embeddings = self._embed(texts, is_query=False)
            self.tokenized_corpus = [tokenize(t) for t in texts]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.corpus_embeddings = np.zeros((0, 1))
            self.bm25 = None

    def _embed(self, texts: list[str], is_query: bool) -> np.ndarray:
        prefixed = [("query: " if is_query else "passage: ") + t for t in texts]
        embs = self.embedder.encode(prefixed, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(embs)

    # -------- retrieval --------

    def dense_search(self, query: str, top_k: int) -> list[int]:
        if not self.documents:
            return []
        q_emb = self._embed([query], is_query=True)[0]
        sims = self.corpus_embeddings @ q_emb  # cosine sim, both normalized
        top_idx = np.argsort(sims)[::-1][:top_k]
        return [int(i) for i in top_idx]

    def sparse_search(self, query: str, top_k: int) -> list[int]:
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [int(i) for i in top_idx]

    @staticmethod
    def reciprocal_rank_fusion(rank_lists: list[list[int]], k: int = 60) -> list[int]:
        scores: dict[int, float] = {}
        for ranked_list in rank_lists:
            for rank, doc_idx in enumerate(ranked_list, start=1):
                scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_idx for doc_idx, _ in ranked]

    def hybrid_search(self, query: str) -> list[int]:
        cfg = self.config
        dense = self.dense_search(query, top_k=cfg.top_k_dense)
        sparse = self.sparse_search(query, top_k=cfg.top_k_sparse)
        fused = self.reciprocal_rank_fusion([dense, sparse])
        return fused[:cfg.candidate_k]

    def rerank(self, query: str, candidate_indices: list[int], top_k: int) -> list[dict]:
        if not candidate_indices:
            return []
        if not self.config.use_reranker or self.reranker is None:
            return [self.documents[i].copy() for i in candidate_indices[:top_k]]

        pairs = [(query, self.documents[i]["text"]) for i in candidate_indices]
        raw_scores = self.reranker.predict(pairs)
        ranked = sorted(zip(candidate_indices, raw_scores), key=lambda x: x[1], reverse=True)

        results = []
        for doc_idx, raw_score in ranked[:top_k]:
            normalized = max(0.0, min(1.0, float(raw_score) / 10.0))
            r = self.documents[doc_idx].copy()
            r["reranker_raw_score"] = float(raw_score)
            r["reranker_score"] = normalized
            results.append(r)
        return results

    def retrieve(self, query: str) -> list[dict]:
        candidates = self.hybrid_search(query)
        return self.rerank(query, candidates, top_k=self.config.top_k_final)


# ============================================================
# 5. System prompt + context formatting
# ============================================================

SYSTEM_PROMPT = """
You are a precise medical research assistant. Answer ONLY from the
provided context excerpts from the source PDFs.

RULES:
- If the user's question asks for the name(s) of a medication/drug/treatment
  (e.g. "what medicine", "which drug", "name of treatment", "write name of
  medicant") for any condition, DO NOT answer it — even if the context
  contains drug names. Respond only with:
  "I can't provide medication names. Please consult a licensed healthcare
  professional or refer to the source document directly."
- Do not list, enumerate, or partially reveal drug/medication names in any
  form (brand name, generic name, or drug class) under any framing of the
  question, including indirect ones like "what is used to treat X" or
  "what options exist for X".
- Never use outside knowledge, assumptions, or memory.
- Never invent diagnoses, treatments, doses, contraindications.
- If the context is insufficient, say so clearly.
- If sources conflict, mention the conflict.
- Do not infer COR or LOE unless explicitly stated.
- Preserve all conditions and qualifiers such as "if tolerated",
  "should be considered", and "may be considered".
- Distinguish diagnosis/evaluation, treatment, target, COR, and LOE.

CITATIONS:
- Every claim you make MUST be grounded in the provided context — never
  answer from outside knowledge.
- Do NOT write inline citations in your answer: no filenames, no "(page
  X)", no "(Section Y)", no ".pdf" mentions, no source names anywhere in
  the prose. The exact source, page, and section for every retrieved
  passage is already shown to the user separately in an Evidence Sources
  panel, so repeating it inline is redundant.
- Just answer the clinical question in plain prose/bullets, grounded in
  the context, without naming or pointing to sources yourself.

STYLE:
- Answer directly and concisely.
- Use short headings and bullets for complex answers.
- Avoid unnecessary repetition.

FINAL CHECK:
Before answering, first check whether the question is asking for a
medication/drug name. If yes, use the refusal response above and stop.
Otherwise, verify that every clinical claim is supported by the provided
context. Do not add any inline citation, filename, page, or section
reference to the answer text — the retrieved context is the only source
of truth, and its provenance is shown separately, not in your prose.
"""


def format_context(results: list[dict], max_chars: int) -> str:
    if not results:
        return "NO RELEVANT CONTEXT WAS RETRIEVED."

    context_parts = []
    total_chars = 0
    for i, result in enumerate(results, start=1):
        source = result.get("source", "source")
        title = result.get("title", "title")
        page_start = result.get("page_start", result.get("page", "page"))
        page_end = result.get("page_end", page_start)
        page_display = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        section = result.get("section", "section")
        text = result.get("text", "").strip()
        reranker_score = result.get("reranker_score")

        block = f"--- CONTEXT {i} ---\n\nTITLE: {title}\nSOURCE: {source}\nPAGE: {page_display}\nSECTION: {section}\n"
        if reranker_score is not None:
            block += f"RETRIEVAL SCORE: {reranker_score:.4f}\n"
        block += f"\nTEXT:\n{text}"
        block = block.strip()

        if total_chars + len(block) > max_chars:
            break
        context_parts.append(block)
        total_chars += len(block)

    return "\n\n".join(context_parts)


# ============================================================
# 6. Generation (OpenRouter) + end-to-end query
# ============================================================


def generate_answer(client, query: str, context: str, chat_history: Optional[list[dict]],
                     model: str, temperature: float = 0.3) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history)
    user_turn = f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"
    messages.append({"role": "user", "content": user_turn})

    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature,
    )
    return response.choices[0].message.content


def rag_query(index: RAGIndex, client, query: str, chat_history: Optional[list[dict]] = None) -> dict:
    results = index.retrieve(query)
    context = format_context(results, max_chars=index.config.max_context_chars)
    raw_answer = generate_answer(
        client, query, context, chat_history, model=index.config.openrouter_model,
    )
    answer = clean_generated_answer(raw_answer)

    sources = [
        {
            "source": r.get("source"),
            "doc_id": r.get("id"),
            "title": r.get("title"),
            "page_start": r.get("page_start", r.get("page")),
            "page_end": r.get("page_end", r.get("page")),
            "section": r.get("section"),
            "reranker_score": r.get("reranker_score"),
        }
        for r in results
    ]
    return {"answer": answer, "sources": sources}
