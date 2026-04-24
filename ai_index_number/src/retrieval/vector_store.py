# src/retrieval/vector_store.py
# Author: [Your Name] | Index: [Your Index Number]
# Manually builds and queries a FAISS index for dense vector retrieval

import faiss
import numpy as np
from typing import List, Tuple, Dict


class VectorStore:
    """
    Wraps a FAISS flat inner-product index.
    Since embeddings are L2-normalized, inner product == cosine similarity.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # Inner Product index
        self.chunks: List[Dict] = []         # Parallel list of chunk metadata

    def add(self, embeddings: np.ndarray, chunks: List[Dict]) -> None:
        """Add embeddings and their corresponding chunks to the store."""
        if embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding count must match chunk count.")
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for top-k most similar chunks.
        Returns list of (chunk_dict, score) sorted by descending score.
        """
        if self.index.ntotal == 0:
            return []
        # query_embedding shape: (1, dim)
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append((self.chunks[idx], float(score)))
        return results

    def size(self) -> int:
        return self.index.ntotal
