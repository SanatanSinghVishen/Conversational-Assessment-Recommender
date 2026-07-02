import json
import re
from retriever import retrieve
from prompts import build_system_prompt, format_catalog_context, build_retrieval_query, build_llm_messages, build_compare_prompt
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

    phase1_prompt = build_system_prompt(
        catalog_context="", 
        force_recommend=force_recommend,
        turn_cap_reached=turn_cap_reached
    )
    
    phase1_response = await call_llm(build_llm_messages(phase1_prompt, messages))
    parsed = parse_llm_json(phase1_response)

    intent = parsed.get("intent", "clarify")
    retrieval_query = parsed.get("retrieval_query", "")
    
    # If forced recommend, we need to ensure intent is recommend
    if turn_cap_reached and intent != "recommend":
        intent = "recommend"
        retrieval_query = build_retrieval_query(messages)

    retrieved_items = []
    if intent in ("recommend", "refine", "compare") and retrieval_query:
        retrieved_items = retrieve(retrieval_query, top_k=15)
    elif intent in ("recommend", "refine") and not retrieval_query:
        retrieval_query = build_retrieval_query(messages)
        retrieved_items = retrieve(retrieval_query, top_k=15)

    if intent == "compare" and retrieved_items:
        compare_prompt = build_compare_prompt(retrieved_items, messages)
        compare_response = await call_llm(build_llm_messages(compare_prompt, messages))
        compare_parsed = parse_llm_json(compare_response)
        reply = compare_parsed.get("reply", parsed.get("reply", ""))
        end_conv = compare_parsed.get("end_of_conversation", False)
        recommendations = []
    else:
        reply = parsed.get("reply", "")
        end_conv = parsed.get("end_of_conversation", False)
        recommendations = retrieved_items[:10] if intent in ("recommend", "refine") else []

    return {
        "reply": reply,
        "recommendations": [
            {"name": r.get("name", "N/A"), "url": r.get("url", ""), "test_type": r.get("test_type", "N/A")}
            for r in recommendations
        ],
        "end_of_conversation": bool(end_conv) or turn_cap_reached
    }
