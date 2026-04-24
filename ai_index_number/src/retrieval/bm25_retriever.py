# src/retrieval/bm25_retriever.py
# Author: [Your Name] | Index: [Your Index Number]
# BM25 keyword-based retriever using rank-bm25

from rank_bm25 import BM25Okapi
from typing import List, Tuple, Dict
import re


def _tokenize(text: str) -> List[str]:
    """Lowercase and tokenize text into words."""
    return re.findall(r'\b[a-z]{2,}\b', text.lower())


class BM25Retriever:
    """Keyword-based retrieval using BM25Okapi."""

    def __init__(self):
        self.bm25: BM25Okapi | None = None
        self.chunks: List[Dict] = []

    def build(self, chunks: List[Dict]) -> None:
        """Build BM25 index from chunk texts."""
        self.chunks = chunks
        tokenized = [_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Return top-k chunks ranked by BM25 score.
        Returns list of (chunk_dict, raw_bm25_score).
        """
        if self.bm25 is None:
            return []
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Pair each chunk with its score, sort descending
        ranked = sorted(
            zip(self.chunks, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        return [(chunk, float(score)) for chunk, score in ranked[:top_k]]
