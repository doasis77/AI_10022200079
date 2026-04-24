# src/retrieval/scoring.py
# Author: [Your Name] | Index: [Your Index Number]
#
# INNOVATION: Domain-Specific Scoring Function
#
# Instead of relying purely on cosine similarity, this module combines
# multiple signals into a final relevance score:
#
#   final_score = (
#       w_vec   * vector_score_norm     +
#       w_bm25  * bm25_score_norm       +
#       source_match_bonus              +
#       keyword_overlap_bonus           +
#       year_numeric_bonus
#   )
#
# Justification:
#   - Vector similarity captures semantic meaning but misses exact keyword matches
#   - BM25 captures exact keyword relevance but misses paraphrased concepts
#   - Source bonus rewards chunks from the right domain (election vs budget)
#   - Keyword overlap boosts chunks containing query terms directly
#   - Year/numeric bonus rewards chunks mentioning specific figures/years in the query

import re
from typing import List, Dict, Tuple


# Scoring weights
W_VECTOR = 0.45
W_BM25   = 0.35
W_SOURCE = 0.10
W_KW     = 0.07
W_YEAR   = 0.03


def _normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    mn, mx = min(scores), max(scores)
    if mx == mn:
        return [1.0 for _ in scores]
    return [(s - mn) / (mx - mn) for s in scores]


def _query_type(query: str) -> str:
    """
    Classify query as 'election', 'budget', or 'mixed'
    based on keyword presence.
    """
    query_lower = query.lower()
    election_kws = {"election", "vote", "votes", "candidate", "party", "parliament",
                    "npp", "ndc", "polling", "constituency", "winner", "result"}
    budget_kws   = {"budget", "expenditure", "revenue", "fiscal", "tax", "gdp",
                    "allocation", "spending", "government", "finance", "debt", "2025"}
    e_hits = sum(1 for k in election_kws if k in query_lower)
    b_hits = sum(1 for k in budget_kws   if k in query_lower)
    if e_hits > 0 and b_hits > 0:
        return "mixed"
    if e_hits >= b_hits:
        return "election"
    return "budget"


def _source_match_bonus(chunk: Dict, query_type: str) -> float:
    """Reward chunks from the domain matching the query type."""
    src = chunk.get("source", "")
    if query_type == "mixed":
        return W_SOURCE * 0.5
    if (query_type == "election" and src == "election") or \
       (query_type == "budget"   and src == "budget"):
        return W_SOURCE
    return 0.0


def _keyword_overlap_bonus(chunk: Dict, query: str) -> float:
    """Reward chunks whose keywords overlap with query tokens."""
    query_tokens = set(re.findall(r'\b[a-z]{3,}\b', query.lower()))
    chunk_kws    = set(chunk.get("keywords", []))
    overlap = len(query_tokens & chunk_kws)
    # Max bonus when overlap >= 3
    return W_KW * min(overlap / 3.0, 1.0)


def _year_numeric_bonus(chunk: Dict, query: str) -> float:
    """Reward chunks that share years or specific numbers with the query."""
    query_nums = set(re.findall(r'\b\d{4}\b', query))
    chunk_text_nums = set(re.findall(r'\b\d{4}\b', chunk.get("text", "")))
    if query_nums & chunk_text_nums:
        return W_YEAR
    return 0.0


def compute_final_scores(
    vector_results: List[Tuple[Dict, float]],
    bm25_results:   List[Tuple[Dict, float]],
    query: str
) -> List[Tuple[Dict, Dict]]:
    """
    Merge vector and BM25 results, deduplicate by chunk_id,
    and compute a final domain-specific score for each chunk.

    Returns a list of (chunk, score_breakdown) sorted by final_score descending.
    """
    query_type = _query_type(query)

    # Collect all unique chunks
    seen: Dict[str, Dict] = {}
    vec_map:  Dict[str, float] = {}
    bm25_map: Dict[str, float] = {}

    for chunk, score in vector_results:
        cid = chunk["chunk_id"]
        seen[cid] = chunk
        vec_map[cid] = score

    for chunk, score in bm25_results:
        cid = chunk["chunk_id"]
        seen[cid] = chunk
        bm25_map[cid] = score

    all_ids = list(seen.keys())

    # Normalize scores across the merged candidate pool
    raw_vec  = [vec_map.get(cid, 0.0)  for cid in all_ids]
    raw_bm25 = [bm25_map.get(cid, 0.0) for cid in all_ids]
    norm_vec  = _normalize_scores(raw_vec)
    norm_bm25 = _normalize_scores(raw_bm25)

    scored = []
    for i, cid in enumerate(all_ids):
        chunk = seen[cid]
        nv   = norm_vec[i]
        nb   = norm_bm25[i]
        src  = _source_match_bonus(chunk, query_type)
        kw   = _keyword_overlap_bonus(chunk, query)
        yr   = _year_numeric_bonus(chunk, query)
        final = W_VECTOR * nv + W_BM25 * nb + src + kw + yr

        scored.append((chunk, {
            "final_score":    round(final, 4),
            "vector_score":   round(raw_vec[i],  4),
            "bm25_score":     round(raw_bm25[i], 4),
            "norm_vector":    round(nv,   4),
            "norm_bm25":      round(nb,   4),
            "source_bonus":   round(src,  4),
            "keyword_bonus":  round(kw,   4),
            "year_bonus":     round(yr,   4),
            "query_type":     query_type
        }))

    scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
    return scored
