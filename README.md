<<<<<<< ours
(cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 9acfd0c26554e6b0a3ed7c69e11553207793df08..2720041514ca912606d7e982c05b93b0a0749135 100644
--- a/README.md
+++ b/README.md
@@ -6,45 +6,67 @@ This repo focuses on the *first steps*: parsing PDFs, chunking, hybrid retrieval
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
 
-### 5) (Optional) Streamlit app
+### 5) LLM-backed generator (Ollama)
+Make sure [Ollama](https://ollama.com) is running locally and that you have pulled the configured model (default `gemma3:4b`). Then run hybrid retrieval + rerank + LLM generation end-to-end:
+```bash
+python -m src.generate \
+  --query "What are the security controls?" \
+  --bm25 data/bm25.json \
+  --faiss_index data/faiss.index \
+  --model_name gemma3:4b \
+  --max_new_tokens 220 \
+  --temperature 0.2
+```
+Useful knobs: `--alpha` (hybrid weighting), `--topk_dense` / `--topk_sparse` / `--topk_final` (retrieval depth), `--abstain_threshold` (when to return "Not enough evidence"), and `--max_ctx_chars` (context budget sent to the LLM).
+
+**Interpreting the output**
+- `ANSWER`: The natural-language response synthesized from the retrieved passages.
+- `CITATIONS`: The minimal set of chunk IDs the LLM claims support the answer. Each ID maps to the `doc_id:page_or_section` in `data/chunks.jsonl`.
+- `CONFIDENCE`: Model-estimated probability that the answer is sufficiently supported (0–1).
+- `COVERAGE`: Fraction of the supporting evidence the LLM believes it has cited; low values suggest under-citation.
+- `USED_TAGS`: All retrieved chunk IDs considered during generation (includes non-cited candidates) — helpful for debugging or manual cross-checks.
+
+If you see fabricated citation IDs, re-run with a lower `--max_new_tokens` or higher `--abstain_threshold`, or inspect `data/chunks.jsonl` to verify the expected tags exist.
+
+### 6) (Optional) Streamlit app
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
 
EOF
)
=======
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

### 5) LLM-backed generator (Ollama)
Make sure [Ollama](https://ollama.com) is running locally and that you have pulled the configured model (default `gemma3:4b`). Then run hybrid retrieval + rerank + LLM generation end-to-end:
```bash
python -m src.generate \
  --query "What are the security controls?" \
  --bm25 data/bm25.json \
  --faiss_index data/faiss.index \
  --model_name gemma3:4b \
  --max_new_tokens 220 \
  --temperature 0.2
```
Useful knobs: `--alpha` (hybrid weighting), `--topk_dense` / `--topk_sparse` / `--topk_final` (retrieval depth), `--abstain_threshold` (when to return "Not enough evidence"), and `--max_ctx_chars` (context budget sent to the LLM).

**Interpreting the output**
- `ANSWER`: The natural-language response synthesized from the retrieved passages.
- `CITATIONS`: The minimal set of chunk IDs the LLM claims support the answer. Each ID maps to the `doc_id:page_or_section` in `data/chunks.jsonl`.
- `CONFIDENCE`: Model-estimated probability that the answer is sufficiently supported (0–1).
- `COVERAGE`: Fraction of the supporting evidence the LLM believes it has cited; low values suggest under-citation.
- `USED_TAGS`: All retrieved chunk IDs considered during generation (includes non-cited candidates) — helpful for debugging or manual cross-checks.

If you see fabricated citation IDs, re-run with a lower `--max_new_tokens` or higher `--abstain_threshold`, or inspect `data/chunks.jsonl` to verify the expected tags exist.

### 6) (Optional) Streamlit app
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
>>>>>>> theirs
