# src/search.py
# DEPRECATED: This module is no longer used. The search functionality has been moved to the llm in src/select.py.

import re
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_retriever(sections_path: str) -> Retriever:
    chunks = load_chunks(sections_path)
    return Retriever(chunks)


def load_chunks(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def searchable_text(chunk: dict) -> str:
    return f"{chunk['id']} {chunk['title']} {chunk['text']}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)+|[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [searchable_text(c) for c in chunks]

        self.bm25 = BM25Okapi([tokenize(c) for c in corpus])

        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = np.asarray(
            self.embed_model.encode(corpus, normalize_embeddings=True)
        )

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(scores)

    def search(self, query: str, top_k: int = 3, alpha: float = 0.5) -> list[dict]:
        bm25_scores = self._normalize(np.array(self.bm25.get_scores(tokenize(query))))

        query_emb = np.asarray(
            self.embed_model.encode([query], normalize_embeddings=True)
        )[0]
        embed_scores = self._normalize(self.embeddings @ query_emb)

        combined = alpha * bm25_scores + (1 - alpha) * embed_scores
        top = np.argsort(combined)[::-1][:top_k]

        return [{"chunk": self.chunks[i], "score": float(combined[i])} for i in top]


if __name__ == "__main__":
    chunks = load_chunks("data/processed/sections.json")
    retriever = Retriever(chunks)

    test_cases = [
        # Duidelijk
        ("loss of steering", "8.9"),
        ("man overboard", "8.3"),

        # Geherformuleerd / parafrase
        ("we can't steer the vessel anymore", "8.9"),
        ("someone fell off the boat", "8.3"),

        # Typo's / stress
        ("stearing compleatly lost", "8.9"),
        ("mna overbord emergency", "8.3"),

        # Legitiem ambigu — meerdere secties zijn relevant
        ("fire in the engine room", None),          # 8.7 én mogelijk 10.2
        ("we had a collision, everyone is fine", None),  # 8.5 én 9.4

        # Onderscheid dat vaak misgaat: near miss vs. echt incident
        ("almost fell overboard but nothing happened", "9.3"),   # NIET 8.3!
        ("crewmember cut their finger, minor", "8.8"),

        # Out-of-scope — hoort NIET zelfverzekerd te matchen
        ("how much ibuprofen can I give someone", None),
        ("how do I reset the radar display", None),

        # Exacte code
        ("SMF 9.3", "SMF 9.3"),

        # Contextloos
        ("what do I do", None),
    ]

    for query, expected in test_cases:
        print(f"\nquery: {query}  (verwacht: {expected})")
        for r in retriever.search(query, top_k=3):
            marker = " <-- verwacht" if r["chunk"]["id"] == expected else ""
            print(f"  {r['score']:.3f}  {r['chunk']['id']:12s} {r['chunk']['title']}{marker}")

    for query in ["man overboard", "8.10", "something is tangled in the propeller"]:
        print(f"\nquery: {query}")
        for r in retriever.search(query, top_k=3):
            print(f"  {r['score']:.3f}  {r['chunk']['id']:12s}  {r['chunk']['title']}")

    failing_queries = [
        "we can't steer the vessel anymore",
        "someone fell off the boat",
        "mna overbord emergency",
    ]

    for query in failing_queries:
        print(f"\n=== {query} ===")
        print("puur embeddings (alpha=0):")
        for r in retriever.search(query, top_k=3, alpha=0.0):
            print(f"  {r['score']:.3f}  {r['chunk']['id']:12s} {r['chunk']['title']}")
        print("puur BM25 (alpha=1):")
        for r in retriever.search(query, top_k=3, alpha=1.0):
            print(f"  {r['score']:.3f}  {r['chunk']['id']:12s} {r['chunk']['title']}")
