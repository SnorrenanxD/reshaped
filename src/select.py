# src/select.py
# LLM-based retrieval: which manual section(s) match a query, and how confidently.
import json
from src.llm import generate_structured

# select_sections: candidate section ids with a confidence each.
SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["id", "confidence"],
            },
        }
    },
    "required": ["matches"],
}
PROMPT = """You are matching a crew member's question to sections of a ship safety manual.

Available sections (id: title):
{index}

Question: "{query}"

Confidence rules:
- "high": the section title names the exact same situation as the question
- "medium": plausibly related, but you are inferring rather than matching directly
- "low": a weak guess, only relevant if nothing better exists
- If no section genuinely addresses the situation, return an empty list. Do not force a match.

Return the 1-3 most relevant section ids with your confidence."""

# verify_match: yes/no second-pass check used when retrieval isn't already "high" confidence.
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["yes", "no"]}
    },
    "required": ["answer"],
}
VERIFY_PROMPT = """Section "{title}":
{text}

Question: "{query}"

Is this the correct procedure category for this situation, even if the situation is minor and the section text focuses on more severe cases?
Answer only yes or no."""


def select_sections(query: str, chunks: list[dict]) -> list[dict]:
    """Asks the LLM which sections plausibly match the query, with a confidence each."""
    index = "\n".join(f"{c['id']}: {c['title']}" for c in chunks)
    prompt = PROMPT.format(index=index, query=query)

    result, _ = generate_structured([{"role": "user", "content": prompt}], SCHEMA)
    known_ids = {c["id"] for c in chunks}
    return [m for m in result["matches"] if m["id"] in known_ids]  # drop any hallucinated id


def verify_match(query: str, chunk: dict) -> bool:
    """Re-reads one section against the query to confirm it's actually the right one."""
    prompt = VERIFY_PROMPT.format(title=chunk["title"], text=chunk["text"], query=query)
    result, _ = generate_structured([{"role": "user", "content": prompt}], VERIFY_SCHEMA, think=False)
    return result["answer"] == "yes"
