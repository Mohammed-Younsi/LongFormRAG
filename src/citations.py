# src/citations.py
import re
import numpy as np

_TAG = re.compile(r"\[([^\]:]+):(\d+)\]")

def coverage(answer: str) -> float:
    """% of sentences that contain at least one [DOC:PAGE] tag."""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip()]
    if not sents:
        return 0.0
    cited = sum(1 for s in sents if _TAG.search(s))
    return cited / len(sents)

def tags_exist_in_packed(citations, packed_used):
    """
+     Accept exact matches. If a citation is like 'DOC:18', map it to a unique
+     tag in packed_used that ends with ':18'. Reject if ambiguous.
+     """
    allowed = set(packed_used)
    for c in citations:
        if c in allowed:
            continue
         # try short form mapping: ':page'
        if ":" in c:
            page = c.split(":")[-1]
            candidates = [t for t in packed_used if t.endswith(":" + page)]
            if len(candidates) == 1:
                 # treat as matched
                continue
        return False
    return True

def should_abstain(rerank_scores, tau: float = 0.15) -> bool:
    """Abstain if the mean re-ranker score is below a threshold."""
    if not rerank_scores:
        return True
    return float(np.mean(rerank_scores)) < tau
