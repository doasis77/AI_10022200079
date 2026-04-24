# src/evaluation/adversarial_tests.py
# Author: [Your Name] | Index: [Your Index Number]
# Standalone adversarial test runner. Can be imported by evaluation UI or run directly.

from typing import List, Dict
from src.retrieval.hybrid_retriever import HybridRetriever
from src.pipeline.rag_pipeline import run_pipeline


ADVERSARIAL_CASES = [
    {
        "query": "Who won the election?",
        "type": "ambiguous",
        "note": "No year or country specified — tests how system handles under-specified queries."
    },
    {
        "query": "Did Ghana achieve a budget surplus in 2025?",
        "type": "assumption_check",
        "note": "Requires the system to verify or deny the premise from document evidence."
    },
    {
        "query": "What was the total debt owed to Mars?",
        "type": "hallucination_bait",
        "note": "Fictional entity — model must not fabricate an answer."
    },
    {
        "query": "The NPP won the 2024 election — what was their winning margin?",
        "type": "misleading_premise",
        "note": "Potentially false premise that the model must not blindly accept."
    },
    {
        "query": "How much revenue did the government earn from oil in every year since 1990?",
        "type": "scope_overflow",
        "note": "Requests information far outside the 2025 budget document scope."
    },
    {
        "query": "Summarize all 500 pages of the budget.",
        "type": "overload",
        "note": "Requests an impossible summarization — tests graceful degradation."
    },
    {
        "query": "Tell me about the upcoming 2030 Ghana election.",
        "type": "future_event",
        "note": "Future event not in documents — must not hallucinate predictions."
    }
]


def run_adversarial_tests(retriever: HybridRetriever) -> List[Dict]:
    """Run all adversarial test cases and return annotated results."""
    results = []
    for case in ADVERSARIAL_CASES:
        out = run_pipeline(case["query"], retriever, top_k=4, prompt_version="v3")
        insufficient = "i do not have enough information" in out["response"].lower()
        results.append({
            "query":        case["query"],
            "type":         case["type"],
            "note":         case["note"],
            "response":     out["response"],
            "insufficient": insufficient,
            "chunk_ids":    [c["chunk_id"] for c, _ in out["selected"]]
        })
    return results
