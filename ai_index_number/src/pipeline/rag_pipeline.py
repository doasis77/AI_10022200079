# src/pipeline/rag_pipeline.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Full RAG pipeline:
#   User Query → Retrieval → Context Selection → Prompt → LLM → Response
#
# This module is the central orchestrator. It:
#   1. Accepts a query
#   2. Runs hybrid retrieval
#   3. Selects and filters context chunks
#   4. Builds a versioned prompt
#   5. Calls the LLM
#   6. Logs every step
#   7. Returns the full pipeline result

import json
import os
from typing import List, Dict, Any, Tuple

from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt_builder import build_prompt
from src.generation.llm_client import generate_response, generate_pure_llm
from src.utils.logger import log_query
from src.retrieval.scoring import _query_type

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../../outputs/chunks.json")


def load_chunks() -> List[Dict]:
    """Load all chunks from the saved JSON file."""
    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"chunks.json not found at {CHUNKS_PATH}. "
            "Run the ingestion script (app.py setup) first."
        )
    with open(CHUNKS_PATH) as f:
        return json.load(f)


def _filter_weak_chunks(
    scored: List[Tuple[Dict, Dict]],
    min_score: float = 0.05
) -> List[Tuple[Dict, Dict]]:
    """Remove chunks whose final score is below the threshold."""
    return [(c, s) for c, s in scored if s["final_score"] >= min_score]


def run_pipeline(
    query: str,
    retriever: HybridRetriever,
    top_k: int = 4,
    prompt_version: str = "v3",
    compare_pure_llm: bool = False
) -> Dict[str, Any]:
    """
    Execute the full RAG pipeline for a query.

    Returns a dict with:
      - query
      - query_type
      - retrieved_chunks (with scores)
      - selected_context (top_k after filtering)
      - final_prompt
      - response (RAG answer)
      - pure_llm_response (if compare_pure_llm=True)
    """
    # ── Step 1: Classify query ────────────────────────────────────────────────
    query_type = _query_type(query)

    # ── Step 2: Hybrid retrieval ──────────────────────────────────────────────
    retrieved = retriever.retrieve(query, top_k=top_k * 3)  # Over-retrieve first

    # ── Step 3: Context selection — filter weak, deduplicate, keep top_k ─────
    filtered = _filter_weak_chunks(retrieved, min_score=0.05)
    selected = filtered[:top_k]

    # ── Step 4: Build prompt ──────────────────────────────────────────────────
    prompt = build_prompt(query, selected, version=prompt_version)

    # ── Step 5: LLM generation ────────────────────────────────────────────────
    response = generate_response(prompt)

    # ── Step 6: Optional pure LLM comparison ─────────────────────────────────
    pure_llm_response = None
    if compare_pure_llm:
        pure_llm_response = generate_pure_llm(query)

    # ── Step 7: Log everything ────────────────────────────────────────────────
    log_entry = {
        "query": query,
        "query_type": query_type,
        "prompt_version": prompt_version,
        "top_k": top_k,
        "retrieved_chunks": [
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "text_preview": c["text"][:200],
                "scores": s
            }
            for c, s in retrieved
        ],
        "selected_context": [
            {
                "chunk_id": c["chunk_id"],
                "source": c["source"],
                "text_preview": c["text"][:200],
                "scores": s
            }
            for c, s in selected
        ],
        "final_prompt": prompt,
        "response": response,
        "pure_llm_response": pure_llm_response,
    }
    log_query(log_entry)

    return {
        "query": query,
        "query_type": query_type,
        "retrieved": retrieved,
        "selected": selected,
        "prompt": prompt,
        "response": response,
        "pure_llm_response": pure_llm_response,
    }
