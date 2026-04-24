# tests/test_retrieval.py
# Author: [Your Name] | Index: [Your Index Number]

import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.retrieval.vector_store  import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.scoring       import compute_final_scores


MOCK_CHUNKS = [
    {"chunk_id": "c1", "source": "election", "text": "NPP won the 2024 election with 53% of votes",
     "keywords": ["npp","election","votes"], "year": 2024},
    {"chunk_id": "c2", "source": "budget",   "text": "Total revenue for 2025 is 200 billion cedis",
     "keywords": ["revenue","budget","cedis"], "year": 2025},
    {"chunk_id": "c3", "source": "election", "text": "NDC parliamentary seats totalled 137",
     "keywords": ["ndc","parliament","seats"], "year": 2024},
]


def test_vector_store_add_and_search():
    """VectorStore should return ranked results from FAISS."""
    dim = 8
    vs = VectorStore(dim=dim)
    rng = np.random.default_rng(42)
    embs = rng.random((3, dim)).astype(np.float32)
    # L2-normalize
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    vs.add(embs, MOCK_CHUNKS)

    query = rng.random((1, dim)).astype(np.float32)
    query = query / np.linalg.norm(query)
    results = vs.search(query, top_k=2)

    assert len(results) == 2
    assert all(isinstance(s, float) for _, s in results)
    print(f"  VectorStore: returned {len(results)} results ✓")


def test_bm25_ranked_results():
    """BM25 should return results sorted by score descending."""
    bm25 = BM25Retriever()
    bm25.build(MOCK_CHUNKS)
    results = bm25.search("NPP election votes", top_k=3)
    assert len(results) > 0
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True), "BM25 results should be sorted descending"
    print(f"  BM25: top chunk = {results[0][0]['chunk_id']}, score = {results[0][1]:.3f} ✓")


def test_score_merging():
    """compute_final_scores should return merged, deduplicated results."""
    vec_results  = [(MOCK_CHUNKS[0], 0.92), (MOCK_CHUNKS[1], 0.70)]
    bm25_results = [(MOCK_CHUNKS[1], 4.5),  (MOCK_CHUNKS[2], 3.1)]
    scored = compute_final_scores(vec_results, bm25_results, "NPP election 2024")
    assert len(scored) == 3, "Should deduplicate: 3 unique chunks"
    final_scores = [s["final_score"] for _, s in scored]
    assert final_scores == sorted(final_scores, reverse=True), "Must be sorted descending"
    print(f"  Score merging: {len(scored)} unique chunks, top score = {final_scores[0]:.4f} ✓")


if __name__ == "__main__":
    print("Running retrieval tests...")
    test_vector_store_add_and_search()
    test_bm25_ranked_results()
    test_score_merging()
    print("All retrieval tests passed ✓")
