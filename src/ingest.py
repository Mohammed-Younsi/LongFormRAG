import argparse, os, json, fitz, re, hashlib
from nltk.tokenize import sent_tokenize

def clean_text(txt: str) -> str:
    txt = txt.replace("\u00ad", "")  # soft hyphen
    txt = re.sub(r'[ \t]+', ' ', txt)
    txt = re.sub(r'\n{2,}', '\n', txt)
    return txt.strip()

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    for pno in range(len(doc)):
        page = doc[pno]
        text = page.get_text("text")
        yield pno+1, clean_text(text)

def sentence_chunks(text, max_tokens=800, overlap=0.15):
    # naive tokenization by words; sentence-aware concatenation
    sents = [s.strip() for s in sent_tokenize(text) if s.strip()]
    tokens = []
    chunks = []
    cur = []
    def tok_count(s): return len(s.split())
    max_words = max_tokens
    stride_words = int(max_words * (1 - overlap))
    i = 0
    while i < len(sents):
        cur = []
        words = 0
        j = i
        while j < len(sents) and words + tok_count(sents[j]) <= max_words:
            cur.append(sents[j])
            words += tok_count(sents[j])
            j += 1
        if cur:
            chunks.append(" ".join(cur))
        i = i + max(1, stride_words // max(1, int(sum(len(c.split()) for c in cur)/len(cur)) )) if cur else j+1
        if not cur:  # fallback to avoid infinite loops
            i = j
    return chunks

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Folder with PDFs")
    ap.add_argument("--out_json", required=True, help="Output JSONL for chunks")
    ap.add_argument("--max_tokens", type=int, default=800)
    ap.add_argument("--overlap", type=float, default=0.15)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    n_chunks = 0
    with open(args.out_json, "w", encoding="utf-8") as fout:
        for fname in os.listdir(args.input_dir):
            if not fname.lower().endswith(".pdf"):
                continue
            path = os.path.join(args.input_dir, fname)
            doc_id = os.path.splitext(fname)[0]
            for page_no, text in extract_pages(path):
                if not text.strip():
                    continue
                for chunk_text in sentence_chunks(text, args.max_tokens, args.overlap):
                    rec = {
                        "chunk_id": sha1(f"{doc_id}-{page_no}-{chunk_text[:60]}"),
                        "doc_id": doc_id,
                        "page": page_no,
                        "text": chunk_text
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_chunks += 1
    print(f"Wrote {n_chunks} chunks to {args.out_json}")

if __name__ == "__main__":
    main()
