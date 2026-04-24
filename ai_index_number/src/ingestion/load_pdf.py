# src/ingestion/load_pdf.py
# Author: [Your Name] | Index: [Your Index Number]
# Loads and extracts raw text from the 2025 Ghana Budget Statement PDF using PyMuPDF

import fitz  # PyMuPDF
from typing import List, Dict


def load_pdf(path: str) -> List[Dict]:
    """
    Extract text page by page from a PDF.
    Returns a list of dicts with page_number and raw_text.
    """
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page_number": i + 1,
                "raw_text": text,
                "source": path
            })
    doc.close()
    return pages
