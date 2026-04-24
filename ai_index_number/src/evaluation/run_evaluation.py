# src/evaluation/run_evaluation.py
# Author: [Your Name] | Index: [Your Index Number]
# Evaluates the RAG system on factual queries and adversarial queries.
# Saves structured results to outputs/evaluation_results.json.

import json
import os
from typing import List, Dict, Any

from src.pipeline.rag_pipeline import run_pipeline
from src.retrieval.hybrid_retriever import HybridRetriever

RESULTS_PATH = os.path.join(os.path.dirname(__file__), "../../outputs/evaluation_results.json")

# ── Factual test queries ────────────────────────────────────────────────────
FACTUAL_QUERIES = [
    {
        "query": "What was the total revenue allocation in the 2025 Ghana budget?",
        "domain": "budget",
        "expected_keywords": ["revenue", "billion", "cedis", "2025"]
    },
    {
        "query": "Which party won the 2024 Ghana presidential election?",
        "domain": "election",
        "expected_keywords": ["npp", "ndc", "votes", "president"]
    },
    {
        "query": "What is Ghana's projected GDP growth rate for 2025?",
        "domain": "budget",
        "expected_keywords": ["gdp", "growth", "percent", "2025"]
    },
    {
        "query": "How many parliamentary seats did NDC win?",
        "domain": "election",
        "expected_keywords": ["ndc", "seats", "parliament", "constituency"]
    },
    {
        "query": "What is the education budget allocation for 2025?",
        "domain": "budget",
        "expected_keywords": ["education", "allocation", "budget", "billion"]
    }
]

# ── Adversarial test queries ─────────────────────────────────────────────────
# These test ambiguity, misleading framing, and incomplete context scenarios
ADVERSARIAL_QUERIES = [
    {
        "query": "Who won the election?",   # Ambiguous — no year or country
        "type": "ambiguous",
        "expected_behavior": "Should ask for clarification or give Ghana-specific answer"
    },
    {
        "query": "What did the president say about the budget?",  # Vague — no context
        "type": "vague",
        "expected_behavior": "Should stay grounded, not hallucinate quotes"
    },
    {
        "query": "How much did Ghana spend on military operations in 2025?",  # Likely not in docs
        "type": "out_of_scope",
        "expected_behavior": "Should respond with insufficient information phrase"
    },
    {
        "query": "The NDC won all 275 constituencies — what was their margin?",  # Misleading premise
        "type": "misleading",
        "expected_behavior": "Should not accept false premise, correct it using documents"
    },
    {
        "query": "What is 2 + 2?",   # Completely irrelevant
        "type": "irrelevant",
        "expected_behavior": "Should note this is unrelated to available documents"
    }
]


def _score_response(response: str, expected_keywords: List[str]) -> Dict[str, Any]:
    """
    Simple heuristic scoring:
    - keyword_hits: how many expected keywords appear in the response
    - hallucination_flag: True if response contains confident numeric claims
      not anchored to source (heuristic only)
    - insufficient_flag: True if fallback phrase appears
    """
    response_lower = response.lower()
    hits = [kw for kw in expected_keywords if kw.lower() in response_lower]
    insufficient = "i do not have enough information" in response_lower
    # Heuristic: high-confidence hallucination if response is long and has no hits
    hallucination_risk = len(response.split()) > 80 and len(hits) == 0

    return {
        "keyword_hits": hits,
        "hit_rate": round(len(hits) / max(len(expected_keywords), 1), 2),
        "insufficient_flag": insufficient,
        "hallucination_risk": hallucination_risk
    }


def run_evaluation(retriever: HybridRetriever) -> Dict[str, Any]:
    """
    Run full evaluation suite.
    Returns structured results dict and saves to outputs/evaluation_results.json.
    """
    results = {
        "factual_results": [],
        "adversarial_results": [],
        "summary": {}
    }

    total_hit_rate = 0.0
    hallucination_count = 0

    # ── Factual queries ──────────────────────────────────────────────────────
    print("Running factual evaluation...")
    for item in FACTUAL_QUERIES:
        pipeline_out = run_pipeline(
            query=item["query"],
            retriever=retriever,
            top_k=4,
            prompt_version="v3",
            compare_pure_llm=True
        )
        rag_score = _score_response(pipeline_out["response"], item["expected_keywords"])
        pure_score = _score_response(
            pipeline_out.get("pure_llm_response", ""),
            item["expected_keywords"]
        )
        if rag_score["hallucination_risk"]:
            hallucination_count += 1
        total_hit_rate += rag_score["hit_rate"]

        results["factual_results"].append({
            "query": item["query"],
            "domain": item["domain"],
            "rag_response": pipeline_out["response"],
            "pure_llm_response": pipeline_out.get("pure_llm_response"),
            "rag_score": rag_score,
            "pure_llm_score": pure_score,
            "retrieved_chunk_ids": [c["chunk_id"] for c, _ in pipeline_out["selected"]],
        })

    # ── Adversarial queries ──────────────────────────────────────────────────
    print("Running adversarial evaluation...")
    for item in ADVERSARIAL_QUERIES:
        pipeline_out = run_pipeline(
            query=item["query"],
            retriever=retriever,
            top_k=4,
            prompt_version="v3"
        )
        results["adversarial_results"].append({
            "query": item["query"],
            "type": item["type"],
            "expected_behavior": item["expected_behavior"],
            "rag_response": pipeline_out["response"],
            "insufficient_flag": "i do not have enough information" in pipeline_out["response"].lower()
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    n = len(FACTUAL_QUERIES)
    results["summary"] = {
        "total_factual_queries": n,
        "avg_keyword_hit_rate": round(total_hit_rate / max(n, 1), 2),
        "hallucination_risk_count": hallucination_count,
        "hallucination_rate": round(hallucination_count / max(n, 1), 2),
        "total_adversarial_queries": len(ADVERSARIAL_QUERIES)
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Evaluation complete. Results saved to {RESULTS_PATH}")
    return results
