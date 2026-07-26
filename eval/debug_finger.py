# debug_finger.py
# Manual check for one tricky query: minor injury vs. the more severe first-aid section.
import json
from src.select import select_sections, verify_match

chunks = json.load(open("data/processed/sections.json"))
by_id = {c["id"]: c for c in chunks}
query = "crewmember cut their finger, minor"

print("=== select_sections, run 3x in a row (checking for LLM noise) ===")
for i in range(3):
    candidates = select_sections(query, chunks)
    print(f"run {i+1}:", candidates)

print("\n=== verify_match against 8.8 specifically ===")
result = verify_match(query, by_id["8.8"])
print("8.8 verified:", result)
print("8.8 text (first 300 chars):")
print(by_id["8.8"]["text"][:300])