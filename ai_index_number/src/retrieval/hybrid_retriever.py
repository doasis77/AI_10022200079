# src/retrieval/hybrid_retriever.py
# Author: [Your Name] | Index: [Your Index Number]
# Combines dense vector retrieval (FAISS) and keyword retrieval (BM25)
# then applies domain-specific scoring to produce a final ranked list.

from typing import List, Tuple, Dict
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.embedder import embed_query
from src.retrieval.scoring import compute_final_scores


class HybridRetriever:
    """
    End-to-end hybrid retrieval:
      1. FAISS dense search
      2. BM25 keyword search
      3. Merge & apply domain-specific scoring
      4. Return top-k re-ranked chunks
    """

    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever):
        self.vector_store = vector_store
        self.bm25 = bm25

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        candidate_pool: int = 15
    ) -> List[Tuple[Dict, Dict]]:
        """
        Retrieve and re-rank chunks for a given query.

        Args:
            query: Natural language query string
            top_k: Number of final chunks to return
            candidate_pool: How many candidates to pull from each retriever
                            before merging (larger = more diverse pool)

        Returns:
            List of (chunk, score_breakdown) sorted by final_score descending
        """
        # 1. Dense retrieval
        q_emb = embed_query(query)
        vec_results = self.vector_store.search(q_emb, top_k=candidate_pool)

        # 2. Keyword retrieval
        bm25_results = self.bm25.search(query, top_k=candidate_pool)

        # 3. Merge + domain-specific scoring
        scored = compute_final_scores(vec_results, bm25_results, query)

        return scored[:top_k]
