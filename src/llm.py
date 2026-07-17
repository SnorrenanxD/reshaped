# src/llm.py
import ollama

MODEL_NAME = "qwen3:8b"

def stream_response(messages: list[dict]):
    stream = ollama.chat(
        model=MODEL_NAME, 
        messages=messages, 
        stream=True, 
        keep_alive="30m"
    )
    for chunk in stream:
        yield chunk["message"]["content"]

def structured_response(messages: list[dict], schema: dict, think: bool = False) -> str:
    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        format=schema,
        think=think,
        keep_alive="30m",
    )
    return response["message"]["content"]

def warm_up():
    ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": "hi"}], keep_alive="30m")