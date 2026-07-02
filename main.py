from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import asyncio
import os

from retriever import load_index, build_index
from catalog import load_catalog
from agent import process_chat

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load/build everything once at startup
    print("Loading catalog...")
    catalog = load_catalog("catalog.json")   # load items from JSON
    if not os.path.exists("faiss_index.bin"):
        print("Building FAISS index (this may take 15-30 seconds)...")
        build_index(catalog)
        print("FAISS index built.")
    print("Loading index...")
    load_index()
    print("Ready!")
    yield
    # cleanup here if needed

app = FastAPI(lifespan=lifespan)

class Message(BaseModel):
    role: str       # "user" or "assistant"
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError("role must be user or assistant")
        return v

class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def messages_not_empty(cls, v: list[Message]) -> list[Message]:
        if not v:
            raise ValueError("messages cannot be empty")
        return v

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]  # empty list when clarifying/refusing
    end_of_conversation: bool

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    messages = [m.model_dump() for m in request.messages]
    try:
        result = await asyncio.wait_for(
            process_chat(messages),
            timeout=28.0  # 2-second buffer before the evaluator's 30s hard timeout
        )
    except asyncio.TimeoutError:
        # Graceful fallback: return empty recommendations with an apology
        return ChatResponse(
            reply="I'm sorry, I took too long to respond. Please try rephrasing your query.",
            recommendations=[],
            end_of_conversation=False
        )
    return result
