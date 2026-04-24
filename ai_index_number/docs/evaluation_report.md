# Evaluation report — RAG vs pure LLM

**Student:** Amegah Sewenam Kwame Bill | **Index:** 10022200079

## Goals

- Measure **grounding** (numbers in answers vs context) as a simple proxy for factual alignment.
- Track **refusal/hedging** when context is insufficient.
- Compare **RAG** vs **pure LLM** (same configured OpenAI-compatible model, no documents).
- Run **adversarial** queries (wrong year, vague prompts).

## How to run

```bash
cd ai_index_number
python -m src.evaluation.run_evaluation
```

Results: `outputs/evaluation_results.json`.

## Metrics (heuristic)

| Metric | Meaning |
|--------|---------|
| `numeric_overlap_*` | Share of numeric tokens in the answer that also appear in retrieved context (RAG) or empty string baseline (pure). |
| `hedge_*` | 1 if answer contains refusal-style phrasing. |
| `consistency_jaccard_tokens` | Token-set similarity between two RAG runs on the same query. |

**These are coursework aids, not certified benchmarks.** Complement with manual grading.

## Adversarial examples

See `src/evaluation/adversarial_tests.py`. Wrong-year questions should ideally yield **refusal** or clear correction when using **v2/v3** prompts.

## Evidence-based RAG vs pure

- **RAG:** When chunks contain the answer, numeric overlap should be **higher** and answers **shorter** and **document-grounded**.
- **Pure LLM:** May still sound plausible but **numeric_overlap_pure** often diverges from CSV ground truth; higher hallucination risk.

## Hallucination rate

Define a simple manual rubric per answer: 0 = grounded, 1 = partial, 2 = fabricated. Average per system. Log in your video walkthrough.

## Retrieval quality

Inspect `outputs/logs.json`: compare `vector_raw` vs `bm25_raw` and final `final_score` for each chunk.
