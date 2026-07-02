---
title: SHL Recommender
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# SHL Conversational Assessment Recommender

## 1. Architecture Overview
The system uses a 4-mode state machine implemented in FastAPI, backed by a `google/gemini-2.5-flash` LLM via OpenRouter. The catalog data is stored locally in JSON and indexed using a FAISS CPU index. The architecture retrieves relevant product items to inject into LLM prompts for grounded responses without hallucinations.

## 2. Retrieval Setup
The system uses the `all-MiniLM-L6-v2` Sentence-Transformers model to create 384-dimensional dense embeddings for each assessment. Retrieval leverages a flat inner-product (cosine similarity) index `IndexFlatIP` from FAISS, configured for top 15 results. Search texts are enriched with name, description, test type, and job levels.

## 3. Agent Design
The agent distinguishes between four intents: CLARIFY, RECOMMEND, REFINE, and COMPARE. Intent is detected inside a single structured JSON LLM call alongside the natural language reply. Context sufficiency and turn-cap rules (max 8 turns) explicitly enforce transitions from CLARIFY to RECOMMEND.

## 4. Prompt Design
Prompts instruct the LLM to output rigid JSON and enforce hard limits against out-of-scope queries (e.g. general hiring tips). The LLM is prohibited from guessing catalog URLs. `temperature` is set to 0.1 for high determinism and parsing stability.

## 5. What Didn't Work
An initial plan to separate intent classification and reply generation into two sequential LLM calls proved too slow given the strict 30s timeout. Consolidating into a single structured output block greatly reduced latency and API failure rates.

## 6. Evaluation
Tested against the 10 public traces with simulated replay logic. Average Recall@10 is targeted over 0.70 via aggressive index field enrichment and context handling across conversational bounds.

## 7. AI Tools Used
Used AI Assistant for rapid project scaffolding, structuring the prompt injection logic, and iterating on FastAPI latency bounds.
