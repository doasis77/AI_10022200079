# src/utils/helpers.py
# Author: [Your Name] | Index: [Your Index Number]
# Shared utility functions used across modules

import re
from typing import List


def normalize_text(text: str) -> str:
    """Remove excessive whitespace and normalize newlines."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_years(text: str) -> List[int]:
    """Extract 4-digit years from text."""
    return [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', text)]


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Simple keyword extraction: lowercase tokens, remove stopwords."""
    stopwords = {
        "the","a","an","and","or","is","in","on","at","to","of","for",
        "with","this","that","are","was","were","be","been","has","have",
        "had","by","from","it","its","as","not","but","so","than","if"
    }
    tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
    filtered = [t for t in tokens if t not in stopwords]
    # Return most frequent tokens (simple frequency rank)
    from collections import Counter
    freq = Counter(filtered)
    return [w for w, _ in freq.most_common(top_n)]


def truncate_text(text: str, max_words: int) -> str:
    """Truncate text to a maximum number of words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."
