import json, argparse, numpy as np
from sentence_transformers import SentenceTransformer
import faiss, pickle
from src.search import hybrid_merge
from src.rerank import load_reranker, rerank as ce_rerank

def load_chunks(bm25_path):
    with open(bm25_path, "rb") as f:
        obj = pickle.load(f)
    return obj["chunks"], obj["bm25"]

def run(query, chunks, bm25, faiss_index, model, alpha, topk_dense, topk_sparse, reranker, topk_final):
    qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = faiss_index.search(qv.astype("float32"), topk_dense)
    di, ds = I[0], D[0]
    scores = bm25.get_scores(query.split())
    idx = np.argsort(scores)[::-1][:topk_sparse]
    si, ss = idx, scores[idx]
    merged = hybrid_merge(di, ds, si, ss, alpha=alpha, n_final=max(topk_final,50))
    cand_idx = [int(i) for i,_ in merged]
    cand_passages = [chunks[i]["text"] for i in cand_idx]
    reranked = ce_rerank(query, merged, cand_passages, reranker, topk_final=topk_final)
    return [int(i) for i,_ in reranked], [int(i) for i,_ in merged]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--topk_dense", type=int, default=50)
    ap.add_argument("--topk_sparse", type=int, default=50)
    ap.add_argument("--topk_final", type=int, default=5)
    args = ap.parse_args()

    chunks, bm25 = load_chunks(args.bm25)
    index = faiss.read_index(args.faiss_index)
    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = load_reranker()

    # map doc_id->chunk indices
    ids_by_doc = {}
    for i,c in enumerate(chunks):
        ids_by_doc.setdefault(c["doc_id"], []).append(i)

    def hit_at_k(ranked, gold_doc_id, k):
        gold = set(ids_by_doc.get(gold_doc_id, []))
        return 1.0 if any(i in gold for i in ranked[:k]) else 0.0

    eval_items = [json.loads(l) for l in open(args.eval)]
    mrr = []
    ndcg5 = []
    for item in eval_items:
        r_ce, r_first = run(item["query"], chunks, bm25, index, emb, args.alpha, args.topk_dense, args.topk_sparse, reranker, args.topk_final)
        # MRR@5
        gold = set(ids_by_doc.get(item["gold_doc_id"], []))
        rr = 0.0
        for rank, idx in enumerate(r_ce[:5], 1):
            if idx in gold:
                rr = 1.0 / rank
                break
        mrr.append(rr)
        # nDCG@5 (binary relevance)
        dcg = sum((1.0/np.log2(r+1)) for r,i in enumerate(r_ce[:5],1) if i in gold)
        idcg = 1.0  # best-case one relevant at rank 1
        ndcg5.append(dcg/idcg)
    print(f"MRR@5: {np.mean(mrr):.3f} | nDCG@5: {np.mean(ndcg5):.3f} | n={len(eval_items)}")

if __name__ == "__main__":
    main()

