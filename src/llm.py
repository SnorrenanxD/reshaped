# src/llm.py
# Wraps the two LLM backends behind one call: Gemini API first, local Ollama as fallback.
import os
import json
import google.genai as genai
from google.genai import types
import ollama

OLLAMA_MODEL = "qwen3:8b"  # local fallback, used when Gemini is unreachable
GEMINI_MODEL = "gemini-3.1-flash-lite"  # primary model

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
_ollama_warmed = False  # only send the Ollama warm-up ping once per process


def _ensure_ollama_warm():
    """First real Ollama call is slow to load the model; ping it once ahead of time."""
    global _ollama_warmed
    if not _ollama_warmed:
        try:
            ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}], keep_alive="30m")
        except Exception:
            pass
        _ollama_warmed = True


def _to_gemini_prompt(messages: list[dict]) -> str:
    """Gemini takes one prompt string, not a chat-style messages list, so flatten it."""
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def generate_structured(messages: list[dict], schema: dict, think: bool = False) -> tuple[dict, str]:
    """Gemini first, Ollama fallback. Returns (result, model_used)."""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_to_gemini_prompt(messages),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return json.loads(response.text), GEMINI_MODEL
    except Exception as e:
        print(f"Gemini structured call failed, falling back to Ollama: {e}")

    _ensure_ollama_warm()
    response = ollama.chat(
        model=OLLAMA_MODEL, messages=messages, format=schema, think=think, keep_alive="30m"
    )
    return json.loads(response["message"]["content"]), OLLAMA_MODEL