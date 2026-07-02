import faiss
import json
import re


MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
META_PATH  = "embeddings_meta.json"

_model = None
_index = None
_meta  = None  # list of catalog items, positionally aligned with the FAISS index

def _load_model():
    global _model
    if _model is None:
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model

def build_index(catalog: list[dict]):
    if not catalog:
        print("Catalog is empty. Skipping index build.")
        return
    model = _load_model()
    # Build search text for each item using ACTUAL catalog fields
    for item in catalog:
        keys = ", ".join(item.get("keys", []))
        languages = ", ".join(item.get("languages", []))
        job_levels = ", ".join(item.get("job_levels", []))
        item["search_text"] = f"""
{item.get('name', '')}
{item.get('description', '')}
Keys: {keys}
Languages: {languages}
Job levels: {job_levels}
Duration: {item.get('duration', 'N/A')}
Remote: {item.get('remote', 'N/A')}
""".strip()
    texts = [item["search_text"] for item in catalog]
    embeddings = model.encode(texts, normalize_embeddings=True,
                              show_progress_bar=True, batch_size=64)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)       # cosine sim via inner product on L2-normalized vectors
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(catalog, f)
    print(f"Built index with {index.ntotal} vectors, dim={dim}")

def load_index():
    global _index, _meta
    try:
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
    except Exception as e:
        print(f"Error loading index: {e}")

def retrieve(query: str, top_k: int = 15) -> list[dict]:
    """Return top_k catalog items most semantically similar to query."""
    if _index is None or _meta is None:
        return []
    model = _load_model()
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    scores, indices = _index.search(q_vec, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        item = dict(_meta[idx])
        item["_score"] = float(score)
        results.append(item)
    return results

def _find_by_name(name_substring: str) -> dict | None:
    """Find a catalog item by exact or substring name match."""
    if _meta is None:
        return None
    for item in _meta:
        if item.get("name", "").lower() == name_substring.lower():
            m = dict(item)
            m["_score"] = 0.0  # placeholder score
            return m
    return None


