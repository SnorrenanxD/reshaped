import ollama

MODEL_NAME = "qwen3:8b"

def stream_response(messages: list[dict]):
    """Receives messages and returns a stream of responses from the LLM."""
    stream = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        yield chunk["message"]["content"]