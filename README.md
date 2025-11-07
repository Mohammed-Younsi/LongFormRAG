# Trust-but-Verify: Retrieval-Augmented QA with Citation Grounding and Hallucination Audits

Minimal starter scaffold for a **hands-on RAG** project suitable for CSC 3310 final project and portfolio.
This repo focuses on the *first steps*: parsing PDFs, chunking, hybrid retrieval (dense + BM25), and a simple Q&A loop.

## Quickstart

### 1) Create environment
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt
```

### 2) Ingest documents
Put your PDFs into `data/raw/`. Then run:
```bash
python -m src.ingest --input_dir data/raw --out_json data/chunks.jsonl
```

### 3) Build indices (dense + BM25)
```bash
python -m src.index --chunks data/chunks.jsonl --faiss_index data/faiss.index --bm25 data/bm25.json
```

### 4) Ask questions (CLI demo)
```bash
python -m src.search --query "What is the leave policy?"   --chunks data/chunks.jsonl --faiss_index data/faiss.index --bm25 data/bm25.json
```

### 5) (Optional) Streamlit app
```bash
streamlit run app/app.py
```

## Project Structure
```
LongFormRAG/
  app/                # simple Streamlit demo
  configs/            # config templates
  data/               # put raw PDFs here; indices & chunks saved here
  src/                # python modules for ingest, index, search
  requirements.txt
  README.md
```

## Notes
- Default embedding: `sentence-transformers/all-MiniLM-L6-v2` (fast, 384-d).
- BM25 via `rank_bm25`. Hybrid score = α * cosine + (1-α) * bm25_zscore.
- This is a starter; you will add re-ranking, audits (NLI), abstention, and evaluation later.
