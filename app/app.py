# app/app.py
import time
import pickle
import numpy as np
import streamlit as st

# === Models ===
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
from scipy.stats import zscore

# ---------- Caching ----------

@st.cache_resource
def load_dense_model(name: str = "all-MiniLM-L6-v2"):
    return SentenceTransformer(name)

@st.cache_resource
def load_reranker(name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    try:
        return CrossEncoder(name)
    except Exception as e:
        st.warning(f"Could not load re-ranker ({name}). Falling back to no re-rank.\n{e}")
        return None

@st.cache_resource
def load_faiss(path: str):
    return faiss.read_index(path)

@st.cache_resource
def load_bm25(path: str):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj["bm25"], obj["chunks"]


# ---------- Retrieval primitives ----------

def dense_search(query, model, faiss_index, topk=50):
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(qv.astype(np.float32), int(topk))
    return I[0], D[0]  # indices, scores

def bm25_search(query, bm25_obj, topk=50):
    scores = bm25_obj.get_scores(query.split())
    idx = np.argsort(scores)[::-1][:int(topk)]
    return idx, scores[idx]

def hybrid_merge(dense_idx, dense_scores, sparse_idx, sparse_scores, alpha=0.6, n_final=50):
    # z-score normalize each list before weighted sum
    ds = zscore(dense_scores) if len(dense_scores) > 1 else dense_scores
    ss = zscore(sparse_scores) if len(sparse_scores) > 1 else sparse_scores
    dmap = {int(i): float(ds[k]) for k, i in enumerate(dense_idx)}
    smap = {int(i): float(ss[k]) for k, i in enumerate(sparse_idx)}
    keys = set(dmap) | set(smap)
    merged = []
    for k in keys:
        score = alpha * dmap.get(k, 0.0) + (1 - alpha) * smap.get(k, 0.0)
        merged.append((k, score))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:n_final]

def rerank(query, candidates, passages, reranker, topk_final=5, batch_size=32):
    """candidates: list[(chunk_idx, hybrid_score)] ; passages aligned to candidates by position."""
    if reranker is None:
        # No re-ranking available; return topk_final from first stage
        return candidates[:topk_final], None
    pairs = [(query, passages[pos]) for pos, _ in enumerate(candidates)]
    scores = reranker.predict(pairs, batch_size=batch_size)
    scored = [(candidates[pos][0], float(scores[pos])) for pos in range(len(candidates))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk_final], scores  # topk and raw scores


# ---------- Packing for “answer + citations” (stub) ----------

def pack_for_answer(query, chunks, ranked_indices, max_chars=1800):
    header = (
        "Answer the QUESTION using ONLY the PASSAGES.\n"
        "Cite sources in [DOC:PAGE] after each claim. If insufficient evidence, say 'Not enough evidence.'\n\n"
        f"QUESTION: {query}\n\nPASSAGES:\n"
    )
    body, used = "", []
    for i in ranked_indices:
        c = chunks[i]
        tag = f"[{c['doc_id']}:{c['page']}] "
        space_left = max_chars - len(body)
        if space_left <= 0:
            break
        snippet = c["text"][:space_left].strip()
        if not snippet:
            continue
        body += f"{tag}{snippet}\n\n"
        used.append(f"{c['doc_id']}:{c['page']}")
    return header + body, used


# ---------- UI ----------

st.set_page_config(page_title="LongFormRAG — Finance/Macro (Hybrid + Re-rank)", layout="wide")
st.title("LongFormRAG — Finance/Macro")
st.caption("Hybrid dense–sparse retrieval over IMF/BIS PDFs with optional cross-encoder re-ranking.")

with st.sidebar:
    st.subheader("Artifacts")
    faiss_path = st.text_input("FAISS index", "data/faiss.index")
    bm25_path = st.text_input("BM25 pickle", "data/bm25.json")

    st.subheader("Retrieval Settings")
    alpha = st.slider("Hybrid α (dense weight)", 0.0, 1.0, 0.6, 0.05)
    topk_dense = st.number_input("Top-k (dense)", 10, 200, 50, 10)
    topk_sparse = st.number_input("Top-k (BM25)", 10, 200, 50, 10)
    n_candidates = st.number_input("Candidates to re-rank", 10, 200, 50, 10)
    topk_final = st.number_input("Final top-k to show", 1, 20, 5, 1)
    use_rerank = st.checkbox("Use re-ranker (CrossEncoder)", True)

    st.subheader("Answer (stub)")
    show_answer = st.checkbox("Pack answer with citations (stub)", False)
    max_ctx_chars = st.slider("Max packed characters", 400, 4000, 1800, 200)

# Load models/data lazily when needed
col_q, col_controls = st.columns([2, 1])
with col_q:
    query = st.text_input("Ask a question", "What are the key global financial risks in 2025?")
    run = st.button("Search")

# Footer for timings
timing = {}

if run:
    try:
        t0 = time.time()
        bm25, chunks = load_bm25(bm25_path)
        index = load_faiss(faiss_path)
        enc = load_dense_model("all-MiniLM-L6-v2")
        timing["load"] = time.time() - t0

        # First-stage retrieval
        t1 = time.time()
        di, ds = dense_search(query, enc, index, topk=topk_dense)
        t_dense = time.time() - t1

        t2 = time.time()
        si, ss = bm25_search(query, bm25, topk=topk_sparse)
        t_sparse = time.time() - t2

        t3 = time.time()
        merged = hybrid_merge(di, ds, si, ss, alpha=alpha, n_final=int(n_candidates))
        t_merge = time.time() - t3

        cand_idx = [int(i) for i, _ in merged]
        cand_passages = [chunks[i]["text"] for i in cand_idx]

        # Optional re-rank
        t4 = time.time()
        reranker = load_reranker() if use_rerank else None
        reranked, raw_scores = rerank(
            query,
            merged,
            cand_passages,
            reranker,
            topk_final=int(topk_final),
            batch_size=32,
        )
        t_rerank = time.time() - t4

        timing.update({"dense": t_dense, "bm25": t_sparse, "merge": t_merge, "rerank": t_rerank})

        # Display results
        st.markdown("### Top Passages" + (" (re-ranked)" if use_rerank else " (first stage)"))
        for rank, (idx, score) in enumerate(reranked, 1):
            rec = chunks[int(idx)]
            with st.container(border=True):
                st.markdown(f"**[{rank}]** `{rec['doc_id']} p.{rec['page']}` — score={score:.3f}")
                st.write(rec["text"][:1200])

        # Optional: pack an answer (stub)
        if show_answer:
            st.markdown("---")
            st.markdown("### Answer (stub with citations)")
            top_indices = [int(i) for i, _ in reranked]
            packed, used = pack_for_answer(query, chunks, top_indices, max_chars=int(max_ctx_chars))
            # For now, show the packed prompt as a placeholder for your LLM call.
            st.code(packed[:2000], language="markdown")
            st.write("**Citations (tags inserted):** ", used)

        # Timings
        st.markdown("---")
        st.caption(
            f"Timings: load={timing['load']:.2f}s | dense={timing['dense']:.2f}s | "
            f"bm25={timing['bm25']:.2f}s | merge={timing['merge']:.2f}s | rerank={timing['rerank']:.2f}s"
        )

    except Exception as e:
        st.error(str(e))
