# src/llm.py
import os
import json
import google.genai as genai
from google.genai import types
import ollama

OLLAMA_MODEL = "qwen3:8b"
GEMINI_MODEL = "gemini-3.1-flash-lite"

gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
_ollama_warmed = False


def _ensure_ollama_warm():
    global _ollama_warmed
    if not _ollama_warmed:
        try:
            ollama.chat(model=OLLAMA_MODEL, messages=[{"role": "user", "content": "hi"}], keep_alive="30m")
        except Exception:
            pass
        _ollama_warmed = True


def _to_gemini_prompt(messages: list[dict]) -> str:
    return "\n\n".join(f"{m['role']}: {m['content']}" for m in messages)


def stream_response(messages: list[dict]):
    """User-facing antwoord. Gemini eerst (komt in één keer), anders echte token-streaming via Ollama."""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_to_gemini_prompt(messages),
        )
        yield response.text
        return
    except Exception as e:
        print(f"Gemini failed, falling back to Ollama: {e}")

    _ensure_ollama_warm()
    stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True, keep_alive="30m")
    for chunk in stream:
        yield chunk["message"]["content"]


def generate_structured(messages: list[dict], schema: dict, think: bool = False) -> dict:
    """Voor select_sections/verify_match. Gemini eerst, Ollama als fallback."""
    try:
        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_to_gemini_prompt(messages),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini structured call failed, falling back to Ollama: {e}")

    _ensure_ollama_warm()
    response = ollama.chat(
        model=OLLAMA_MODEL, messages=messages, format=schema, think=think, keep_alive="30m"
    )
    return json.loads(response["message"]["content"])