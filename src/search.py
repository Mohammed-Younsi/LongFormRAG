import argparse, pickle, json, numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

def dense_search(query, model, faiss_index, topk=50):
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(qv.astype(np.float32), topk)
    return I[0], D[0]

def bm25_search(query, bm25_obj, topk=50):
    scores = bm25_obj.get_scores(query.split())
    idx = np.argsort(scores)[::-1][:topk]
    return idx, scores[idx]

def hybrid_merge(dense_idx, dense_scores, sparse_idx, sparse_scores, alpha=0.6, n_final=5, n_total=None):
    # Normalize scores
    if n_total is None:
        n_total = max(len(dense_idx), len(sparse_idx))
    # zscore each
    ds = zscore(dense_scores) if len(dense_scores) > 1 else dense_scores
    ss = zscore(sparse_scores) if len(sparse_scores) > 1 else sparse_scores
    # map to dict
    dmap = {int(i): float(ds[k]) for k,i in enumerate(dense_idx)}
    smap = {int(i): float(ss[k]) for k,i in enumerate(sparse_idx)}
    keys = set(dmap) | set(smap)
    merged = []
    for k in keys:
        d = dmap.get(k, 0.0)
        s = smap.get(k, 0.0)
        score = alpha * d + (1 - alpha) * s
        merged.append((k, score))
    merged.sort(key=lambda x: x[1], reverse=True)
    return merged[:n_final]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--topk_dense", type=int, default=50)
    ap.add_argument("--topk_sparse", type=int, default=50)
    ap.add_argument("--topk_final", type=int, default=5)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    # Load artifacts
    import faiss
    index = faiss.read_index(args.faiss_index)
    with open(args.bm25, "rb") as f:
        obj = pickle.load(f)
        bm25 = obj["bm25"]
        chunks = obj["chunks"]

    model = SentenceTransformer(args.model)

    # Searches
    di, ds = dense_search(args.query, model, index, args.topk_dense)
    si, ss = bm25_search(args.query, bm25, args.topk_sparse)

    merged = hybrid_merge(di, ds, si, ss, alpha=args.alpha, n_final=args.topk_final)

    print("\n=== Top Passages ===")
    for rank, (idx, score) in enumerate(merged, 1):
        rec = chunks[int(idx)]
        preview = rec["text"][:240].replace("\n"," ")
        print(f"[{rank}] score={score:.3f} | {rec['doc_id']} p.{rec['page']}")
        print(f"     {preview}...")
    print("")

if __name__ == "__main__":
    main()
