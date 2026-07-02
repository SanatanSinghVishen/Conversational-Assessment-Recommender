import requests
import json
import os

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

    # Compute Recall@10
    expected = set(trace.get("expected_shortlist", []))
    retrieved_top10 = set(final_recommendations[:10])
    recall = len(expected & retrieved_top10) / len(expected) if expected else 0.0
    return {"recall": recall, "got": list(retrieved_top10), "expected": list(expected)}

if __name__ == "__main__":
    traces_dir = os.path.join(os.path.dirname(__file__), "traces")
    api = os.getenv("API_URL", "http://localhost:8000")
    results = []
    
    if not os.path.exists(traces_dir):
        print(f"Traces directory not found: {traces_dir}")
        exit(1)
        
    trace_files = [f for f in os.listdir(traces_dir) if f.endswith('.md') or f.endswith('.json')]
    if not trace_files:
        print(f"No traces found in {traces_dir}")
        exit(1)

    for f in trace_files:
        filepath = os.path.join(traces_dir, f)
        try:
            with open(filepath, 'r', encoding='utf-8') as fp:
                # If it's markdown with json block, you might need custom parsing. Assuming JSON here.
                # If it's pure json:
                content = fp.read()
                # quick hack if traces are actually JSON disguised as .md
                if content.strip().startswith('{'):
                    trace = json.loads(content)
                else:
                    # Try to find JSON block
                    import re
                    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", content)
                    if match:
                        trace = json.loads(match.group(1))
                    else:
                        print(f"Skipping {f} - could not parse JSON")
                        continue
                        
            result = evaluate_trace(trace, api)
            results.append(result["recall"])
            print(f"{f}: Recall@10 = {result['recall']:.3f}")
        except Exception as e:
            print(f"Error processing {f}: {e}")
            
    if results:
        print(f"\nMean Recall@10: {sum(results)/len(results):.3f}")
