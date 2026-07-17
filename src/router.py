# src/router.py
from src.select import select_sections, verify_match
from src.llm import stream_response

CONFIDENCE_RANK = {"high": 0, "medium": 1, "low": 2}
ESCALATE_MESSAGE = "Not found with confidence in the SMS. Please consult the Master or DPA."

ACTION_PROMPT = """Section "{title}":
{text}

Question: "{query}"

Turn this into a clear action plan:
1. SITUATION (one sentence)
2. IMMEDIATE ACTIONS (numbered, only what the text states or clearly implies)
3. CONDITIONAL steps (mark clearly if they depend on circumstances)
4. REPORTING (who must be notified, which form)

Use only the text above. Do not invent steps."""


def resolve_section(query: str, chunks: list[dict]) -> tuple[dict | None, str]:
    """Returns (verified chunk or None, path taken: 'direct' | 'verified' | 'escalate')."""
    candidates = select_sections(query, chunks)
    if not candidates:
        return None, "escalate"

    by_id = {c["id"]: c for c in chunks}
    highs = [c for c in candidates if c["confidence"] == "high"]

    if len(highs) == 1:
        return by_id.get(highs[0]["id"]), "direct"

    ordered = sorted(candidates, key=lambda c: CONFIDENCE_RANK[c["confidence"]])
    for candidate in ordered:
        chunk = by_id.get(candidate["id"])
        if chunk and verify_match(query, chunk):
            return chunk, "verified"

    return None, "escalate"


def handle_query(query: str, chunks: list[dict]):
    chunk, _ = resolve_section(query, chunks)
    if chunk is None:
        yield ESCALATE_MESSAGE
        return

    prompt = ACTION_PROMPT.format(title=chunk["title"], text=chunk["text"], query=query)
    yield from stream_response([{"role": "user", "content": prompt}])
