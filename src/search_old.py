# src/search_old.py
# DEPRECATED: no longer used. Retrieval moved to the LLM-based approach in src/select.py.

import re
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_retriever(sections_path: str) -> Retriever:
    """Loads chunks from disk and builds a ready-to-query Retriever."""
    chunks = load_chunks(sections_path)
    return Retriever(chunks)


def load_chunks(path: str) -> list[dict]:
    """Loads the ingested sections from disk."""
    with open(path) as f:
        return json.load(f)


def searchable_text(chunk: dict) -> str:
    """Fields concatenated for keyword/embedding search."""
    return f"{chunk['id']} {chunk['title']} {chunk['text']}"


def tokenize(text: str) -> list[str]:
    """Words and dotted section numbers (e.g. "8.3"), for BM25."""
    return re.findall(r"\d+(?:\.\d+)+|[a-z0-9]+", text.lower())


class Retriever:
    """Hybrid BM25 + embedding search over the ingested sections."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        corpus = [searchable_text(c) for c in chunks]

        self.bm25 = BM25Okapi([tokenize(c) for c in corpus])

        self.embed_model = SentenceTransformer(EMBED_MODEL)
        self.embeddings = np.asarray(
            self.embed_model.encode(corpus, normalize_embeddings=True)
        )

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max scale to [0, 1] so BM25 and embedding scores are comparable."""
        lo, hi = scores.min(), scores.max()
        return (scores - lo) / (hi - lo) if hi - lo > 1e-9 else np.zeros_like(scores)

    def search(self, query: str, top_k: int = 3, alpha: float = 0.5) -> list[dict]:
        """Top-k chunks by a weighted blend of BM25 and embedding similarity."""
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
        # Clear-cut
        ("loss of steering", "8.9"),
        ("man overboard", "8.3"),

        # Reworded / paraphrased
        ("we can't steer the vessel anymore", "8.9"),
        ("someone fell off the boat", "8.3"),

        # Typos / under stress
        ("stearing compleatly lost", "8.9"),
        ("mna overbord emergency", "8.3"),

        # Legitimately ambiguous — more than one section is relevant
        ("fire in the engine room", None),               # 8.7 and possibly 10.2
        ("we had a collision, everyone is fine", None),  # 8.5 and 9.4

        # Distinction that's easy to get wrong: near miss vs. actual incident
        ("almost fell overboard but nothing happened", "9.3"),   # NOT 8.3!
        ("crewmember cut their finger, minor", "8.8"),

        # Out-of-scope — should NOT confidently match anything
        ("how much ibuprofen can I give someone", None),
        ("how do I reset the radar display", None),

        # Exact code
        ("SMF 9.3", "SMF 9.3"),

        # No context given
        ("what do I do", None),
    ]

    for query, expected in test_cases:
        print(f"\nquery: {query}  (expected: {expected})")
        for r in retriever.search(query, top_k=3):
            marker = " <-- expected" if r["chunk"]["id"] == expected else ""
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
        print("pure embeddings (alpha=0):")
        for r in retriever.search(query, top_k=3, alpha=0.0):
            print(f"  {r['score']:.3f}  {r['chunk']['id']:12s} {r['chunk']['title']}")
        print("pure BM25 (alpha=1):")
        for r in retriever.search(query, top_k=3, alpha=1.0):
            print(f"  {r['score']:.3f}  {r['chunk']['id']:12s} {r['chunk']['title']}")
