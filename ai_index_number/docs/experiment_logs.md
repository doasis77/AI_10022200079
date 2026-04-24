# Experiment logs — Chunking & retrieval

**Student:** Amegah Sewenam Kwame Bill | **Index:** 10022200079

## Chunking strategy design (with justification)

### 1) CSV chunking

- **Strategy:** one CSV row is converted into one natural-language chunk.
- **Why this design:** each row is already an atomic fact unit (year, region, candidate, party, votes, vote share). Combining rows can blur entities and reduce retrieval precision.
- **Implementation:** `build_csv_row_chunks()` in `src/preprocessing/chunking.py`.

### 2) PDF fixed-size chunking

- **Chunk size:** **400 words**
- **Overlap:** **80 words** (20%)
- **Why this design:**
  - 400 words keeps enough fiscal context for a complete idea.
  - 80-word overlap reduces boundary loss where key facts cross chunk edges.
  - Predictable size helps stable embedding behavior.
- **Implementation:** `chunk_pdf_fixed_words()` in `src/preprocessing/chunking.py`.

### 3) PDF paragraph-aware chunking

- **Target size range:** **300 to 500 words**
- **Overlap method:** sentence-tail carryover (last 1–2 sentences, capped).
- **Why this design:**
  - Budget PDFs are section/paragraph oriented; paragraph-aware chunks preserve semantic flow.
  - A bounded range prevents very small noisy chunks and very large mixed-topic chunks.
- **Implementation:** `chunk_pdf_paragraph_aware()` in `src/preprocessing/chunking.py`.

## Chunk size and overlap (PDF fixed strategy)

- **400 words / 80 overlap:** chosen to fit comfortably under typical embedding context while **overlap** reducing the chance that a fact sits on a chunk boundary. Smaller chunks increase granularity but fragment tables/lists; larger chunks mix unrelated themes.
- **Paragraph-aware (300–500 words):** respects blank-line structure in budget PDFs, improving topical coherence versus arbitrary fixed windows.

## Comparing strategies (qualitative)

Run the **Chunking strategy lab** expander in the Streamlit app or:

```python
from pathlib import Path
from src.ingestion.load_pdf import load_pdf_text
from src.preprocessing.chunking import compare_chunk_strategies_sample
raw, _ = load_pdf_text(Path("data/2025_Budget_Statement.pdf"))
print(compare_chunk_strategies_sample(raw))
```

Expect **fewer, longer** chunks for paragraph mode vs **more uniform** lengths for fixed mode. Retrieval quality: paragraph chunks often improve **precision** on thematic budget questions; fixed mode can be stronger when **exact phrases** repeat across pages.

## Implementation checklist completed

- [x] Designed and justified chunk sizes and overlap
- [x] Implemented CSV row chunking
- [x] Implemented PDF fixed chunking (400/80)
- [x] Implemented PDF paragraph-aware chunking (300–500 + sentence overlap)
- [x] Added metadata: `chunk_id`, `source`, `chunk_type`, `text`, `year`, `keywords`

## Hybrid retrieval failure case (example)

- **Symptom:** Query uses budget vocabulary but the top vector hits are election rows because of shared tokens (e.g. “Ghana”, “2020”).
- **Mitigation implemented:** `classify_query` + **source_match_bonus** in `scoring.py`, plus **BM25** to surface keyword-heavy budget chunks, and **deduplication** in context selection.

_Record dated runs and observations below as you experiment._

| Date | PDF strategy | Notes |
|------|----------------|-------|
| YYYY-MM-DD | paragraph | ... |
