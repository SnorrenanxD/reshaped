# debug_finger.py
import json
from src.select import select_sections, verify_match

chunks = json.load(open("data/processed/sections.json"))
by_id = {c["id"]: c for c in chunks}
query = "crewmember cut their finger, minor"

print("=== select_sections, 3x achter elkaar (check op LLM-ruis) ===")
for i in range(3):
    candidates = select_sections(query, chunks)
    print(f"run {i+1}:", candidates)

print("\n=== verify_match op 8.8 specifiek ===")
result = verify_match(query, by_id["8.8"])
print("8.8 verified:", result)
print("8.8 text (eerste 300 tekens):")
print(by_id["8.8"]["text"][:300])