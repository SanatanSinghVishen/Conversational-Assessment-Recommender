import json
import re
from retriever import retrieve, multi_retrieve
from prompts import build_system_prompt, build_retrieval_query, build_llm_messages, build_compare_prompt, format_catalog_context
from llm import call_llm

OUT_OF_SCOPE_PATTERNS = [
    r"\bsalar(y|ies)\b", r"\blegal\b", r"\bdiscrimination\b",
    r"\binterview (tips|questions)\b", r"\bhow to fire\b",
    r"\bignore (your )?instructions\b", r"\bact as\b",
    r"\bforget (your )?instructions\b", r"\bpretend\b"
]

def count_assistant_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if m["role"] == "assistant")

def should_force_recommend(messages: list[dict]) -> bool:
    assistant_turns = count_assistant_turns(messages)
    user_text = " ".join(m["content"] for m in messages if m["role"] == "user").lower()
    has_some_context = any(kw in user_text for kw in [
        "developer", "manager", "analyst", "sales", "java", "python",
        "graduate", "senior", "junior", "mid", "entry", "leader",
        "engineer", "hr", "finance", "customer", "service"
    ])
    return assistant_turns >= 2 and has_some_context

def is_obviously_out_of_scope(messages: list[dict]) -> bool:
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
    ).lower()
    return any(re.search(p, last_user) for p in OUT_OF_SCOPE_PATTERNS)

def parse_llm_json(raw: str) -> dict:
    raw = raw.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        raw = match.group(1)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{[\s\S]+\}", raw)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except Exception:
                pass
    return {"intent": "clarify", "reply": "Could you tell me more about the role you're hiring for?",
            "retrieval_query": "", "end_of_conversation": False}

async def process_chat(messages: list[dict]) -> dict:
    if len(messages) == 1 and len(messages[0]["content"].split()) < 6:
        # Vague turn 1 query short-circuit
        return {
            "reply": "Could you tell me a bit more about the role or skill area you are hiring for?",
            "recommendations": [],
            "end_of_conversation": False
        }
        
    if is_obviously_out_of_scope(messages):
        return {
            "reply": "I can only help with SHL assessment recommendations. Could you tell me about the role you are hiring for?",
            "recommendations": [], 
            "end_of_conversation": False
        }

    turn_cap_reached = len(messages) >= 7
    force_recommend = should_force_recommend(messages)

    # 1. Always do retrieval first to ground the LLM
    # Use multi_retrieve for broad coverage across diverse topics
    retrieved_items = multi_retrieve(messages, top_k=20)
    catalog_context_str = format_catalog_context(retrieved_items)

    # 2. Make the single LLM call with the catalog context injected
    prompt = build_system_prompt(
        catalog_context=catalog_context_str, 
        force_recommend=force_recommend,
        turn_cap_reached=turn_cap_reached
    )
    
    response = await call_llm(build_llm_messages(prompt, messages))
    parsed = parse_llm_json(response)

    intent = parsed.get("intent", "clarify")
    reply = parsed.get("reply", "")
    end_conv = parsed.get("end_of_conversation", False)
    recommended_urls = parsed.get("recommended_urls", [])
    
    # If forced recommend, we need to ensure intent is recommend
    if turn_cap_reached and intent != "recommend":
        intent = "recommend"
        end_conv = True

    recommendations = []
    if intent in ("recommend", "refine"):
        # Map URLs back to catalog objects
        for url in recommended_urls:
            item = next((i for i in retrieved_items if i.get("link") == url), None)
            if item:
                recommendations.append(item)
        
        # Fallback if LLM didn't format URLs exactly or omitted them: 
        # Just use the top 5 if the LLM completely failed to provide them but intended to recommend
        if not recommendations and retrieved_items:
            recommendations = retrieved_items[:5]

    return {
        "reply": reply,
        "recommendations": [
            {"name": r.get("name", "N/A"), "url": r.get("link", ""), "test_type": ", ".join(r.get("keys", [])) if r.get("keys") else "N/A"}
            for r in recommendations
        ],
        "end_of_conversation": bool(end_conv) or turn_cap_reached
    }
