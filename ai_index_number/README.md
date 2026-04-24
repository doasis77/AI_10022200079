# Academic City RAG — Ghana Elections & Budget

**Student:** Amegah Sewenam Kwame Bill  
**Index:** 10022200079  
**Course:** CS4241 Introduction to Artificial Intelligence  
**Institution:** Academic City University — assistant for public datasets (election results + national budget excerpt).

## Overview

The system implements a **manual RAG pipeline** (no LangChain/LlamaIndex): documents are ingested, chunked, embedded with `sentence-transformers`, indexed in **FAISS**, augmented with **BM25**, merged through **hybrid retrieval** and **domain-specific scoring**, then passed to an **OpenAI-compatible API** (Groq/OpenAI).

## Technology Stack

Python · Streamlit · pandas · PyMuPDF · sentence-transformers · FAISS · NumPy · scikit-learn · rank-bm25 · python-dotenv · openai · pytest

## Architecture summary

1. **Ingest** `data/Ghana_Election_Result.csv` and `data/2025_Budget_Statement.pdf`.  
2. **Chunk:** CSV → one natural-language row per chunk; PDF → fixed (400 words, 80 overlap) *or* paragraph-aware (300–500 words).  
3. **Embed** all chunks; build **FAISS** index (inner product on L2-normalized vectors = cosine similarity).  
4. **BM25** index over chunk texts.  
5. **Query:** classify (election / budget / mixed) → retrieve top-K from both → normalize scores → **weighted domain score** → context-window management (**rank + filter + truncate**) with configurable chunk count and word budgets.  
6. **Prompt** (V1 / V2 / V3) → **OpenAI-compatible chat API** → answer.  
7. **Log** structured JSON to `outputs/logs.json`.

See [docs/architecture.md](docs/architecture.md) for diagrams and data flow.

## Innovation: domain-specific scoring

Final ranking combines normalized **vector** and **BM25** scores with bonuses for **source alignment**, **keyword overlap** (Jaccard), and **year / numeric** overlap when applicable (`src/retrieval/scoring.py`).

## Setup

### 1. Clone / copy this folder

Replace the root folder name `ai_index_number` with your own naming if required.

### 2. Python environment

```bash
cd ai_index_number
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3. LLM provider setup (Groq / OpenAI)

Use an OpenAI-compatible endpoint. For Groq, create an API key from the [Groq Console](https://console.groq.com/keys).

### 4. Environment variables

Edit `.env`:

- `OPENAI_API_KEY` — your API key (`gsk_...` for Groq, `sk-...` for OpenAI)
- `OPENAI_BASE_URL` — set `https://api.groq.com/openai/v1` for Groq, or leave empty for OpenAI
- `OPENAI_MODEL` — e.g. `llama-3.3-70b-versatile` (Groq) or `gpt-4o-mini` (OpenAI)

### 5. Data files

- `data/Ghana_Election_Result.csv` — included sample / your full file.
- `data/2025_Budget_Statement.pdf` — add your official PDF, **or** generate the bundled teaching sample:

```bash
python scripts/create_sample_budget_pdf.py
```

## Run locally

```bash
python -m streamlit run app.py
```

First launch auto-builds/loads the index. You can still use sidebar **Rebuild Index** manually.

## How retrieval works

- **Dense:** query embedding vs chunk embeddings in FAISS (cosine via normalized inner product).  
- **Sparse:** BM25 over tokenized chunk text.  
- **Hybrid:** union of candidates → min–max normalize each signal → **domain score** → sort → **context selection** (score threshold + near-duplicate removal).
- **Context window management:** truncate selected chunks with `per_chunk_max_words` and `total_context_max_words`, and cap number of chunks (`max_context_chunks`).

## UI controls for context selection

Sidebar settings include:

- `Top-K retrieval`
- `Max chunks per chat`
- `Per-chunk word cap`
- `Total context word cap`

This directly supports exam requirements around context window control.

## Image generation command (UI feature)

In chat input, use:

- `/image your prompt here`
- `image: your prompt here`
- `generate image: your prompt here`

The app returns an AI-generated image URL and renders it inline in chat history.

## Evaluation

Heuristic batch evaluation (RAG vs pure LLM):

```bash
python run_evaluation.py
```

Output: `outputs/evaluation_results.json`. See [docs/evaluation_report.md](docs/evaluation_report.md).

## Tests

```bash
pytest -q
```

## Documentation map

| File | Purpose |
|------|---------|
| [docs/architecture.md](docs/architecture.md) | System design & flow |
| [docs/experiment_logs.md](docs/experiment_logs.md) | Chunking experiments |
| [docs/evaluation_report.md](docs/evaluation_report.md) | Metrics & adversarial discussion |
| [docs/walkthrough_notes.md](docs/walkthrough_notes.md) | Video script outline |

## Deployability

- **Streamlit:** deploy to Streamlit Community Cloud or any host that can run Python; configure API secrets (`OPENAI_API_KEY`, optional `OPENAI_BASE_URL`, `OPENAI_MODEL`) in host secret management.  
- **Secrets:** never commit `.env`; use host secret management for production.

## Evaluation summary (fill after runs)

- Mean numeric overlap (RAG vs pure): _TBD_  
- Hallucination rubric (manual): _TBD_  
- Notes: _TBD_

---


