# src/rerank.py
from typing import List, Tuple
from sentence_transformers import CrossEncoder

def load_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name)

def rerank(
    query: str,
    candidates: List[Tuple[int, float]],   # (chunk_idx, hybrid_score)
    passages: list,                        # aligned with candidates' order
    model: CrossEncoder,
    topk_final: int = 5,
    batch_size: int = 32,
):
    # IMPORTANT: index by position, not by chunk_idx
    pairs = [(query, passages[pos]) for pos, _ in enumerate(candidates)]
    scores = model.predict(pairs, batch_size=batch_size)

    # Keep original chunk indices with new scores
    scored = [(candidates[pos][0], float(scores[pos])) for pos in range(len(candidates))]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:topk_final]
