# test_select.py
import json
from src.router import resolve_section

def load_chunks(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


test_cases = [
    ("loss of steering", "8.9"),
    ("man overboard", "8.3"),
    ("we can't steer the vessel anymore", "8.9"),
    ("someone fell off the boat", "8.3"),
    ("stearing compleatly lost", "8.9"),
    ("mna overbord emergency", "8.3"),
    ("fire in the engine room", "8.7"),
    ("we had a collision, everyone is fine", "8.5"),
    ("almost fell overboard but nothing happened", "9.3"),
    ("crewmember cut their finger, minor", "8.8"),
    ("how much ibuprofen can I give someone", None),
    ("how do I reset the radar display", None),
    ("9.3", "9.3"),
    ("what do I do", None),
]

if __name__ == "__main__":
    chunks = load_chunks("data/processed/sections.json")

    correct = 0
    path_counts = {"direct": 0, "verified": 0, "escalate": 0}

    for query, expected in test_cases:
        chunk, path = resolve_section(query, chunks)
        got_id = chunk["id"] if chunk else None
        hit = got_id == expected
        correct += hit
        path_counts[path] += 1

        marker = "✓" if hit else "✗"
        print(f"{marker}  [{path:9s}] query: {query}")
        print(f"           verwacht={expected}  gekregen={got_id}")

    print(f"\nScore: {correct}/{len(test_cases)}")
    print(f"Paden: {path_counts}")
