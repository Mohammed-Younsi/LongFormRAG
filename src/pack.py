# src/pack.py
def pack(query, chunks, ranked_indices, max_chars=1800, per_passage=350):
    header = (
        "Answer the QUESTION using ONLY the PASSAGES.\n"
        "Cite sources in [DOC:PAGE] after each claim.\n"
        "If evidence is insufficient, say 'Not enough evidence.'\n\n"
        f"QUESTION: {query}\n\nPASSAGES:\n"
    )
    body, used = "", []
    for i in ranked_indices:
        if len(body) >= max_chars:
            break
        c = chunks[i]
        tag = f"[{c['doc_id']}:{c['page']}] "
        snippet = (c["text"] or "").strip()[:per_passage]
        if not snippet:
            continue
        entry = f"{tag}{snippet}\n\n"
        if len(body) + len(entry) > max_chars:
            remain = max_chars - len(body)
            entry = f"{tag}{snippet[:max(0, remain - len(tag) - 2)]}\n\n"
        body += entry
        used.append(f"{c['doc_id']}:{c['page']}")
    # Add an explicit list of allowed tags to force exact copying
    taglist = "VALID TAGS: " + ", ".join(f"[{t}]" for t in used) + "\n\n"
    return header + taglist + body, used
