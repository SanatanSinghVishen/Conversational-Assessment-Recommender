"""Diagnostic: parse all traces, test FAISS multi_retrieve locally."""
import sys, os, re, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retriever import multi_retrieve, load_index, build_index
from catalog import load_catalog

def normalize_name(name):
    name = re.sub(r'\s*[-\u2013\u2014]\s*', '-', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name.lower()

def parse_markdown_trace(content):
    conversation = []
    turns = re.split(r"###\s*Turn\s*\d+", content, flags=re.IGNORECASE)
    for turn in turns[1:]:
        user_match = re.search(r"\*\*User\*\*\s*> (.*?)(?=\*\*Agent\*\*|\Z)", turn, re.DOTALL | re.IGNORECASE)
        if user_match:
            user_text = user_match.group(1).replace("\n> ", "\n").strip()
            conversation.append({"role": "user", "content": user_text})
    expected_shortlist = []
    for line in content.splitlines():
        match = re.match(r"\|\s*\d+\s*\|\s*(.*?)\s*\|", line.strip())
        if match:
            expected_shortlist.append(match.group(1).strip())
    return {
        "conversation": conversation,
        "expected_shortlist": list(dict.fromkeys(expected_shortlist))
    }

# Rebuild index with fixed search_text
if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index.bin")):
    print("Rebuilding FAISS index...")
    catalog = load_catalog("catalog.json")
    build_index(catalog)

load_index()

traces_dir = os.path.join(os.path.dirname(__file__), "traces")
total_recall_10 = []
total_recall_all = []
for f in sorted(os.listdir(traces_dir)):
    if not f.endswith('.md'):
        continue
    with open(os.path.join(traces_dir, f), encoding='utf-8') as fp:
        content = fp.read()
    trace = parse_markdown_trace(content)
    
    print(f"\n{'='*60}")
    print(f"TRACE: {f}")
    print(f"  Expected ({len(trace['expected_shortlist'])}): {trace['expected_shortlist']}")
    
    # Use multi_retrieve like agent does
    results = multi_retrieve(trace["conversation"], top_k=25)
    retrieved_names = [r.get("name") for r in results]
    
    expected = set(normalize_name(n) for n in trace["expected_shortlist"])
    
    # Check recall at 10
    got10 = set(normalize_name(n) for n in retrieved_names[:10])
    overlap10 = expected & got10
    recall10 = len(overlap10) / len(expected) if expected else 0.0
    total_recall_10.append(recall10)
    
    # Check if expected items are ANYWHERE in the full 25 results (what LLM sees)
    got_all = set(normalize_name(n) for n in retrieved_names)
    overlap_all = expected & got_all
    recall_all = len(overlap_all) / len(expected) if expected else 0.0
    total_recall_all.append(recall_all)
    
    print(f"  FAISS top-10 names: {retrieved_names[:10]}")
    print(f"  Recall@10 = {recall10:.3f}")
    print(f"  Recall@ALL (in top 25) = {recall_all:.3f}")
    
    missed = expected - got_all
    if missed:
        print(f"  MISSED entirely: {missed}")

print(f"\nMean FAISS Recall@10: {sum(total_recall_10)/len(total_recall_10):.3f}")
print(f"Mean FAISS Recall@ALL (what LLM sees): {sum(total_recall_all)/len(total_recall_all):.3f}")
