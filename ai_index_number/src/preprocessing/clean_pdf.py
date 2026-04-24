# src/preprocessing/clean_pdf.py
# Author: [Your Name] | Index: [Your Index Number]
# Cleans raw PDF text extracted page-by-page

import re
from typing import List, Dict
from src.utils.helpers import normalize_text


def clean_page(raw_text: str) -> str:
    """
    Clean a single page of extracted PDF text:
    - Remove page headers/footers (short repeated lines)
    - Normalize whitespace
    - Remove control characters
    """
    # Remove lines that are very short (likely headers/footers/page numbers)
    lines = raw_text.split("\n")
    filtered = [l for l in lines if len(l.strip()) > 10]
    text = " ".join(filtered)
    # Remove non-printable characters
    text = re.sub(r'[^\x20-\x7E\n]', ' ', text)
    return normalize_text(text)


def clean_pages(pages: List[Dict]) -> List[Dict]:
    """
    Apply clean_page to all extracted pages.
    Returns pages with cleaned text, skipping near-empty pages.
    """
    cleaned = []
    for page in pages:
        text = clean_page(page["raw_text"])
        if len(text.split()) > 20:  # skip pages with barely any content
            cleaned.append({**page, "cleaned_text": text})
    return cleaned
