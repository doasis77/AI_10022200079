# Video walkthrough notes

**Student:** Amegah Sewenam Kwame Bill | **Index:** 10022200079

## Suggested structure (5–8 minutes)

1. **Problem** — Academic City assistant for Ghana election CSV + 2025 budget PDF.
2. **No frameworks** — Show `src/` tree; point out manual chunking, FAISS, BM25, hybrid scoring.
3. **Live demo** — `streamlit run app.py`: rebuild index once, ask a **factual** election question; show retrieved chunks and scores.
4. **Prompts** — Toggle V1/V2/V3 comparison; explain grounding + refusal string in V2.
5. **Innovation** — Open `scoring.py`: weights for vector, BM25, source, keywords, year.
6. **Evaluation** — Run `python -m src.evaluation.run_evaluation`; open `evaluation_results.json`.
7. **Failure case** — Describe cross-domain confusion; cite hybrid + source bonus fix.
8. **Deployment** — configure `.env` (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL`); mention Streamlit Cloud secrets management.

## Checklist before recording

- [ ] API key configured (`OPENAI_API_KEY` or `GROQ_API_KEY`).
- [ ] `OPENAI_BASE_URL` configured for Groq (`https://api.groq.com/openai/v1`) or left empty for OpenAI.
- [ ] `OPENAI_MODEL` set to an available model for your provider.
- [ ] `.env` configured.
- [ ] `data/Ghana_Election_Result.csv` and `data/2025_Budget_Statement.pdf` present.
- [ ] Index built (sidebar **Rebuild**).

## Talking points

- **Chunk overlap:** reduces boundary errors; trade-off with redundancy.
- **Hybrid:** dense retrieval + keyword gaps filled by BM25.
- **Pure LLM baseline:** shows why retrieval matters for tabular PDF/CSV facts.
