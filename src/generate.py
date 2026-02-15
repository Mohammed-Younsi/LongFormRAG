# src/generate.py
import argparse
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.search import hybrid_merge
from src.rerank import load_reranker, rerank as ce_rerank
from src.pack import pack
from src.citations import coverage, tags_exist_in_packed, should_abstain
from src.llm_ollama import generate_json_ollama_chat

def _append_run(path, query, answer, citations, confidence, coverage, used_tags):
    if not path:
        return
    import json, time, os
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({
            "ts": time.time(),
            "query": query,
            "answer": answer,
            "citations": citations,
            "confidence": confidence,
            "coverage": coverage,
            "used_tags": used_tags,
        }) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--faiss_index", required=True)

    # retrieval knobs
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--topk_dense", type=int, default=50)
    ap.add_argument("--topk_sparse", type=int, default=50)
    ap.add_argument("--topk_final", type=int, default=5)

    # abstention + packing
    ap.add_argument("--abstain_threshold", type=float, default=0.15)
    ap.add_argument("--max_ctx_chars", type=int, default=1800)

    # LLM (Ollama)
    ap.add_argument("--model_name", default="gemma3:4b", help="Ollama model name (see `ollama list`)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_new_tokens", type=int, default=220)
    ap.add_argument("--save", default=None, help="Append results to this JSONL file")


    args = ap.parse_args()

    # --- load artifacts ---
    with open(args.bm25, "rb") as f:
        obj = pickle.load(f)
    chunks, bm25 = obj["chunks"], obj["bm25"]
    index = faiss.read_index(args.faiss_index)
    enc = SentenceTransformer("all-MiniLM-L6-v2")
    reranker = load_reranker()

    # --- retrieve ---
    qv = enc.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(qv.astype("float32"), args.topk_dense)
    di, ds = I[0], D[0]

    s_all = bm25.get_scores(args.query.split())
    si = np.argsort(s_all)[::-1][:args.topk_sparse]
    ss = s_all[si]

    merged = hybrid_merge(di, ds, si, ss, alpha=args.alpha, n_final=max(args.topk_final, 50))

    cand_idx = [int(i) for i, _ in merged]
    cand_passages = [chunks[i]["text"] for i in cand_idx]

    # --- re-rank ---
    reranked = ce_rerank(
        args.query,
        candidates=merged,
        passages=cand_passages,
        model=reranker,
        topk_final=args.topk_final,
    )
    top_idx = [int(i) for i, _ in reranked]
    rerank_scores = [float(s) for _, s in reranked]

    # --- abstain on weak evidence ---
    if should_abstain(rerank_scores, args.abstain_threshold):
        answer = "Not enough evidence in the corpus."
        cited = []
        conf = 0.2
        cov = 0.0
        used = [f"{chunks[i]['doc_id']}:{chunks[i]['page']}" for i in top_idx]

        print("\n=== ANSWER ===\n" + answer)
        print("\nCITATIONS:", cited)
        print("CONFIDENCE:", round(conf, 3))
        print("COVERAGE:", round(cov, 2))
        print("USED_TAGS:", used)

        _append_run(args.save, args.query, answer, cited, conf, cov, used)
        return


    # --- pack context ---
    prompt, used = pack(args.query, chunks, top_idx, max_chars=args.max_ctx_chars)

    # --- generate via Ollama ---
    result = generate_json_ollama_chat(
        model=args.model_name,
        prompt=prompt,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
    )

    # --- robust parsing & defaults ---
    answer = (result.get("answer") or "").strip()
    cited = result.get("citations") or []

    # confidence: coerce to float with a safe default
    conf = result.get("confidence", 0.5)
    try:
        conf = float(conf)
    except Exception:
        conf = 0.5

    # fallback: if citations array is empty, try extracting from the answer text
    if not cited:
        import re
        cited = [
            f"{m.group(1)}:{m.group(2)}"
            for m in re.finditer(r"\[([^\]:]+):(\d+)\]", answer)
        ]

    # --- validate citations ---
    cov = coverage(answer)
    ok_tags = tags_exist_in_packed(cited, used)

    # If citations/coverage are weak, abstain gracefully (conf is already set)
    if cov < 0.6 or not ok_tags:
        answer = "Not enough evidence."
        cited = []
        conf = min(conf, 0.3)

    # --- print final ---
    print("\n=== ANSWER ===\n" + answer)
    print("\nCITATIONS:", cited)
    print("CONFIDENCE:", round(conf, 3))
    print("COVERAGE:", round(cov, 2))
    print("USED_TAGS:", used)
    
    _append_run(args.save, args.query, answer, cited, conf, cov, used)




if __name__ == "__main__":
    main()
