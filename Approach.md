# Approach: SHL Conversational Assessment Recommender

## 1. Architecture Overview
The system is built as a stateless FastAPI web service designed for high throughput and rapid responses. 
- **Scraper**: A one-time Python script using BeautifulSoup extracts the catalog into `catalog.json`. 
- **Retriever**: FAISS is used for in-memory, zero-latency vector retrieval using Sentence-Transformers. 
- **Agent Orchestration**: `agent.py` drives a 2-phase state machine that calls the LLM for intent classification, executes the FAISS retrieval, and optionally calls the LLM a second time for grounded comparisons.
- **LLM**: We use `gemini-1.5-flash` via OpenRouter to ensure low latency and high instruction-following adherence, ensuring we meet the strict 30-second API timeout limit.

## 2. Retrieval Setup
The system uses **FAISS FlatIP** (inner product on normalized vectors, achieving cosine similarity) loaded in memory at server startup to guarantee sub-millisecond retrieval speeds. 
- **Embedding Model**: `all-MiniLM-L6-v2` was chosen because it runs natively on CPU, is incredibly lightweight (80MB), and provides excellent semantic matching for short product descriptions.
- **Search Text**: During index building, we concatenate the product name, description, job levels, duration, and test types into a single rich text block. This allows a user's multi-faceted constraint (e.g., "personality test for senior managers") to match strongly.
- **Retrieval Threshold**: We retrieve the top 15 results (`top_k=15`) to ensure a diverse set of options before filtering down to the final shortlist.

## 3. Agent Design
The core brain of the application lives in `agent.py`. It routes the conversation into one of four distinct intents (`clarify`, `recommend`, `refine`, `compare`).
- **Intent Detection**: The LLM determines the intent in its initial pass based on the user's input, outputting a strict JSON schema.
- **Turn Cap & Forcing Rules**: We enforce a hard limit of 8 conversational turns. If the conversation drags, a `should_force_recommend` heuristic (triggered after 2 assistant turns) explicitly injects a forcing prompt, guaranteeing the user eventually receives a shortlist rather than endless questions.
- **Scope Refusal**: An array of Regex patterns (`is_obviously_out_of_scope`) intercepts prompt-injection or off-topic questions (e.g., salary, firing advice) *before* the LLM is called, instantly saving latency and preventing hallucinations.

## 4. Prompt Design
The `SYSTEM_PROMPT_TEMPLATE` is engineered for strict schema compliance and boundary enforcement.
- **Catalog Injection**: Instead of flooding the LLM context with the entire 200-item catalog, the prompt only injects the top retrieved items after Phase 2. This keeps the prompt short and the LLM tightly grounded.
- **JSON Structure**: The LLM outputs a single JSON block containing its natural language reply, intent, and generated `retrieval_query`.
- **Temperature**: We set the LLM temperature to `0.1`. Creative hallucinations (especially regarding URLs) are catastrophic in this use case; low temperature ensures the model reliably follows the markdown structure and logic constraints.

## 6. Evaluation
Using the provided `eval/run_eval.py` script against the 10 public traces, the application achieves a **Mean Recall@10** between **0.25 and 0.43**, heavily dependent on the LLM's query generation capability. The system successfully passes all hard constraints (no hallucinated URLs, perfect Pydantic schema compliance, and turn-cap enforcement). The conversational traces prove the agent gracefully drops items when asked (e.g., removing OPQ32r) and refuses out-of-scope requests.

## 7. AI Tools Used
- **Google Deepmind Agentic Coding**: Used to analyze the take-home specification, rapidly refactor the FAISS and FastAPI architectures to ensure strict spec compliance, and diagnose the End-to-End Recall metric ceilings.
- **Anthropic Claude 3.5 Sonnet**: Used in earlier iterations to draft the regex boundary patterns and the baseline `multi_retrieve` heuristics.
