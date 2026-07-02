import requests
import json
import os
import re

def normalize_name(name: str) -> str:
    """Normalize assessment names for comparison (strip extra whitespace, normalize dashes)."""
    name = re.sub(r'\s*[-–—]\s*', '-', name)  # normalize dashes and surrounding spaces
    name = re.sub(r'\s+', ' ', name).strip()   # collapse whitespace
    return name.lower()

def evaluate_trace(trace: dict, api_url: str) -> dict:
    """
    Simulates the evaluator's replay harness.
    trace = {"persona": "...", "expected_shortlist": ["name1", "name2", ...],
             "conversation": [{"role": "user", "content": "..."}, ...]}
    """
    messages = []
    final_recommendations = []

    for turn in trace.get("conversation", []):
        if turn["role"] != "user":
            continue
        messages.append({"role": "user", "content": turn["content"]})
        try:
            resp = requests.post(f"{api_url}/chat", json={"messages": messages}, timeout=30)
            body = resp.json()
            messages.append({"role": "assistant", "content": body.get("reply", "")})

            if body.get("recommendations"):
                final_recommendations = [r["name"] for r in body["recommendations"]]

            if body.get("end_of_conversation"):
                break
        except Exception as e:
            print(f"Error calling API: {e}")
            break

    # Compute Recall@10 with normalized names
    expected = set(normalize_name(n) for n in trace.get("expected_shortlist", []))
    retrieved_top10 = set(normalize_name(n) for n in final_recommendations[:10])
    recall = len(expected & retrieved_top10) / len(expected) if expected else 0.0
    return {"recall": recall, "got": list(retrieved_top10), "expected": list(expected)}

def parse_markdown_trace(content: str) -> dict:
    conversation = []
    # Split the document by "### Turn"
    turns = re.split(r"###\s*Turn\s*\d+", content, flags=re.IGNORECASE)
    for turn in turns[1:]: # skip the first part before Turn 1
        user_match = re.search(r"\*\*User\*\*\s*> (.*?)(?=\*\*Agent\*\*|\Z)", turn, re.DOTALL | re.IGNORECASE)
        if user_match:
            # Clean up blockquotes
            user_text = user_match.group(1).replace("\n> ", "\n").strip()
            conversation.append({"role": "user", "content": user_text})
            
    # Expected shortlist: Names from tables
    expected_shortlist = []
    for line in content.splitlines():
        match = re.match(r"\|\s*\d+\s*\|\s*(.*?)\s*\|", line.strip())
        if match:
            expected_shortlist.append(match.group(1).strip())
            
    return {
        "conversation": conversation,
        "expected_shortlist": list(dict.fromkeys(expected_shortlist))
    }

if __name__ == "__main__":
    traces_dir = os.path.join(os.path.dirname(__file__), "traces")
    api = os.getenv("API_URL", "http://localhost:8000")
    results = []
    
    if not os.path.exists(traces_dir):
        print(f"Traces directory not found: {traces_dir}")
        exit(1)
        
    trace_files = [f for f in os.listdir(traces_dir) if f.endswith('.md')]
    if not trace_files:
        print(f"No traces found in {traces_dir}")
        exit(1)

    for f in trace_files:
        filepath = os.path.join(traces_dir, f)
        try:
            with open(filepath, 'r', encoding='utf-8') as fp:
                content = fp.read()
                
            trace = parse_markdown_trace(content)
            if not trace["conversation"]:
                print(f"Skipping {f} - no User turns found")
                continue
                
            result = evaluate_trace(trace, api)
            results.append(result["recall"])
            print(f"{f}: Recall@10 = {result['recall']:.3f}")
            print(f"  Got: {result['got']}")
            print(f"  Expected: {result['expected']}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if results:
        print(f"\nMean Recall@10: {sum(results)/len(results):.3f}")
