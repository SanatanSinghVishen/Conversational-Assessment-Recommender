import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI client to point to OpenRouter
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY", "dummy")
)

async def call_llm(messages: list[dict]) -> str:
    try:
        response = await client.chat.completions.create(
            model="google/gemini-1.5-flash",
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        return response.choices[0].message.content or "{}"
    except Exception as e:
        print(f"Error calling LLM: {e}")
        return "{}"
