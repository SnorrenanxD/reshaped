from src.llm import stream_response

def handle_query(messages: list[dict]):
    """
    Forwards message to the LLM and returns a stream or responses.
    """
    return stream_response(messages)