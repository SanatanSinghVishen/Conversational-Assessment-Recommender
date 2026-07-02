import faiss
import json
import re

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "faiss_index.bin"
META_PATH  = "embeddings_meta.json"

_model = None
_index = None
_meta  = None  # list of catalog items, positionally aligned with the FAISS index

def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
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

# Cross-cutting assessments that are relevant to virtually every hiring scenario
# but are never retrieved via semantic search because users describe roles/skills,
# not assessment product names.
CROSS_CUTTING_NAMES = [
    "Occupational Personality Questionnaire OPQ32r",
    "SHL Verify Interactive G+",
]

def multi_retrieve(messages: list[dict], top_k: int = 25) -> list[dict]:
    """
    Run multiple FAISS queries to cover diverse topics in a conversation.
    
    Strategy:
    1. Run a holistic query with all user context combined
    2. Run per-turn queries for each user message
    3. Extract individual topic keywords and run focused queries
    4. Merge and deduplicate by name, keeping highest score
    5. Guarantee cross-cutting assessments (OPQ32r, Verify G+) are included
       without competing for score-based slots
    """
    if _index is None or _meta is None:
        return []
    
    user_messages = [m["content"] for m in messages if m["role"] == "user"]
    if not user_messages:
        return []
    
    seen = {}  # name -> item (with best score)
    
    def _add_results(items):
        for item in items:
            name = item.get("name")
            if name not in seen or item["_score"] > seen[name]["_score"]:
                seen[name] = item
    
    # Query 1: holistic combined query (capped at 512 chars)
    combined = " ".join(user_messages)[:512]
    _add_results(retrieve(combined, top_k=15))
    
    # Query 2: per-turn queries to capture distinct topics
    for msg in user_messages:
        _add_results(retrieve(msg, top_k=10))
    
    # Query 3: extract individual topic keywords and query each
    # Split on commas, semicolons, "and", periods to isolate individual skills/tools
    all_text = " ".join(user_messages)
    fragments = re.split(r'[,;.]\s*|\band\b', all_text)
    for frag in fragments:
        frag = frag.strip()
        if len(frag) > 3 and len(frag) < 100:  # skip noise
            _add_results(retrieve(frag, top_k=5))
    
    # Sort topic-specific results by score descending
    topic_results = sorted(seen.values(), key=lambda x: x["_score"], reverse=True)
    
    # We want to ensure CROSS_CUTTING_NAMES are in the final top_k list.
    # First, separate out the cross-cutting items from the topic results.
    final_cross_cutting = []
    filtered_topic_results = []
    
    for item in topic_results:
        if item.get("name") in CROSS_CUTTING_NAMES:
            final_cross_cutting.append(item)
        else:
            filtered_topic_results.append(item)
            
    # Add any missing cross-cutting items that weren't retrieved at all
    found_cc_names = {item.get("name") for item in final_cross_cutting}
    for cc_name in CROSS_CUTTING_NAMES:
        if cc_name not in found_cc_names:
            item = _find_by_name(cc_name)
            if item:
                final_cross_cutting.append(item)
                
    # Combine: take top (top_k - len(cross_cutting)) topic results, then append cross_cutting
    max_topic = top_k - len(final_cross_cutting)
    final = filtered_topic_results[:max_topic] + final_cross_cutting
    
    return final[:top_k]
