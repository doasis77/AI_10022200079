# src/preprocessing/chunking.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Two PDF chunking strategies are implemented:
#
# A) Fixed-size chunking (400 words, 80-word overlap)
#    - Pros: Uniform chunk sizes; predictable for embedding
#    - Cons: May split mid-sentence or mid-concept
#    - Best for: Densely formatted documents like budget statements
#
# B) Paragraph-aware chunking (300–500 words, 1-sentence overlap)
#    - Pros: Preserves semantic boundaries; more readable context
#    - Cons: Variable chunk size; some sections may be too short/long
#    - Best for: Narrative sections with clear paragraph structure
#
# Chunk size justification:
#   400 words ≈ 512 tokens, which fits well within sentence-transformer
#   context windows and keeps semantic focus tight enough for retrieval.
#   80-word overlap ensures continuity across chunk boundaries.

import re
from typing import List, Dict
from src.utils.helpers import extract_keywords


# ─── Strategy A: Fixed-size chunking ──────────────────────────────────────────

def fixed_size_chunks(
    pages: List[Dict],
    chunk_size: int = 400,
    overlap: int = 80
) -> List[Dict]:
    """
    Concatenate all page text and split into overlapping fixed-size word chunks.
    """
    # Concatenate all cleaned page text
    full_text = " ".join(p["cleaned_text"] for p in pages)
    words = full_text.split()

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        text = " ".join(chunk_words)
        years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', text)]

        chunks.append({
            "chunk_id": f"pdf_fixed_{chunk_idx}",
            "source": "budget",
            "chunk_type": "fixed_size",
            "text": text,
            "section_title": None,
            "year": years[0] if years else None,
            "keywords": extract_keywords(text),
            "word_count": len(chunk_words)
        })
        start += chunk_size - overlap
        chunk_idx += 1

    return chunks


# ─── Strategy B: Paragraph-aware chunking ─────────────────────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using basic punctuation rules."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def paragraph_aware_chunks(
    pages: List[Dict],
    min_words: int = 300,
    max_words: int = 500
) -> List[Dict]:
    """
    Split text by double-newline paragraphs, then group paragraphs into
    chunks within the target word range. Adjacent chunks share one sentence
    of overlap for continuity.
    """
    full_text = " ".join(p["cleaned_text"] for p in pages)

    # Split on double newlines (or sentence-ending punctuation followed by caps)
    paragraphs = re.split(r'\n{2,}|\.\s{2,}(?=[A-Z])', full_text)
    paragraphs = [p.strip() for p in paragraphs if len(p.split()) > 10]

    chunks = []
    chunk_idx = 0
    current_words: List[str] = []
    last_sentence = ""

    for para in paragraphs:
        para_words = para.split()

        # If adding this paragraph would exceed max_words, flush first
        if len(current_words) + len(para_words) > max_words and len(current_words) >= min_words:
            text = " ".join(current_words)
            years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', text)]
            chunks.append({
                "chunk_id": f"pdf_para_{chunk_idx}",
                "source": "budget",
                "chunk_type": "paragraph_aware",
                "text": text,
                "section_title": _detect_section_title(text),
                "year": years[0] if years else None,
                "keywords": extract_keywords(text),
                "word_count": len(current_words)
            })
            chunk_idx += 1
            # Carry over last sentence as overlap
            overlap_words = last_sentence.split() if last_sentence else []
            current_words = overlap_words + para_words
        else:
            current_words += para_words

        # Track last sentence for overlap
        sentences = _split_sentences(para)
        if sentences:
            last_sentence = sentences[-1]

    # Flush remaining words
    if len(current_words) > 50:
        text = " ".join(current_words)
        years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', text)]
        chunks.append({
            "chunk_id": f"pdf_para_{chunk_idx}",
            "source": "budget",
            "chunk_type": "paragraph_aware",
            "text": text,
            "section_title": _detect_section_title(text),
            "year": years[0] if years else None,
            "keywords": extract_keywords(text),
            "word_count": len(current_words)
        })

    return chunks


def _detect_section_title(text: str) -> str | None:
    """
    Heuristic: if the first sentence is short and title-cased, treat it as a section header.
    """
    first_line = text.split(".")[0].strip()
    if 3 <= len(first_line.split()) <= 10 and first_line.istitle():
        return first_line
    return None
