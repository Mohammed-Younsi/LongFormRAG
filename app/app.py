import streamlit as st
import pickle, faiss
from sentence_transformers import SentenceTransformer
from scipy.stats import zscore
import numpy as np

st.set_page_config(page_title="LongFormRAG (starter)", layout="wide")

st.title("LongFormRAG — Hybrid Retrieval (Starter)")
st.caption("Dense (MiniLM) + BM25 with simple hybrid merge. This starter does retrieval only (no re-ranker or generator yet).")

col1, col2 = st.columns([2,1])
with col2:
    faiss_path = st.text_input("FAISS index path", "data/faiss.index")
    bm25_path = st.text_input("BM25 pickle path", "data/bm25.json")
    alpha = st.slider("Hybrid α (dense weight)", 0.0, 1.0, 0.6, 0.05)
    topk_dense = st.number_input("Top-k dense", 10, 200, 50, 10)
    topk_sparse = st.number_input("Top-k BM25", 10, 200, 50, 10)
    topk_final = st.number_input("Top-k final", 1, 20, 5, 1)

with col1:
    query = st.text_input("Ask a question", "What are the prerequisites?")
    if st.button("Search"):
        try:
            index = faiss.read_index(faiss_path)
            with open(bm25_path, "rb") as f:
                obj = pickle.load(f)
                bm25 = obj["bm25"]
                chunks = obj["chunks"]
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

            # dense
            qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            D, I = index.search(qv.astype(np.float32), int(topk_dense))
            di, ds = I[0], D[0]

            # sparse
            scores = bm25.get_scores(query.split())
            si = np.argsort(scores)[::-1][:int(topk_sparse)]
            ss = scores[si]

            # merge
            dsn = zscore(ds) if len(ds) > 1 else ds
            ssn = zscore(ss) if len(ss) > 1 else ss
            dmap = {int(i): float(dsn[k]) for k,i in enumerate(di)}
            smap = {int(i): float(ssn[k]) for k,i in enumerate(si)}
            keys = set(dmap) | set(smap)
            merged = []
            for k in keys:
                score = alpha * dmap.get(k, 0.0) + (1 - alpha) * smap.get(k, 0.0)
                merged.append((k, score))
            merged.sort(key=lambda x: x[1], reverse=True)
            merged = merged[:int(topk_final)]

            for rank, (idx, score) in enumerate(merged, 1):
                rec = chunks[int(idx)]
                st.markdown(f"**[{rank}]** `{rec['doc_id']} p.{rec['page']}` — score={score:.3f}")
                st.write(rec["text"][:800])

        except Exception as e:
            st.error(str(e))
