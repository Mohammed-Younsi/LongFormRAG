import argparse, json, os, pickle, numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import StandardScaler

def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def build_dense(chunks, model_name):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return emb

def save_faiss(emb, index_path):
    import faiss, numpy as np
    # Ensure correct dtype/layout for FAISS
    emb = np.ascontiguousarray(emb.astype('float32'))
    dim = emb.shape[1]
    # Exact inner-product index (cosine, since we normalized)
    index = faiss.IndexFlatIP(dim)
    index.add(emb)
    faiss.write_index(index, index_path)


def build_bm25(chunks):
    # simple word tokenization
    corpus = [c["text"].split() for c in chunks]
    bm25 = BM25Okapi(corpus)
    return bm25

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--faiss_index", required=True)
    ap.add_argument("--bm25", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.faiss_index), exist_ok=True)
    os.makedirs(os.path.dirname(args.bm25), exist_ok=True)

    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks")
    emb = build_dense(chunks, args.model)
    save_faiss(emb, args.faiss_index)

    bm25 = build_bm25(chunks)
    with open(args.bm25, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)

    print(f"Saved FAISS -> {args.faiss_index} and BM25 -> {args.bm25}")

if __name__ == "__main__":
    main()
