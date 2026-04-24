# src/preprocessing/clean_csv.py
# Author: [Your Name] | Index: [Your Index Number]
# Cleans election CSV and converts rows to natural-language text chunks

import pandas as pd
from typing import List, Dict
from src.utils.helpers import normalize_text, extract_keywords


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the election results DataFrame:
    - Strip whitespace from string columns
    - Drop fully empty rows
    - Normalize column names
    """
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df = df.dropna(how="all")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def rows_to_chunks(df: pd.DataFrame, source: str) -> List[Dict]:
    """
    Convert each row of election data into a natural-language text chunk.
    Each chunk represents one election result record.
    """
    chunks = []
    for idx, row in df.iterrows():
        # Build readable sentence from row fields
        parts = []
        for col, val in row.items():
            if pd.isna(val):
                continue
            s = str(val).strip()
            if not s or s.lower() in ("nan", "none"):
                continue
            parts.append(f"{col.replace('_', ' ').title()}: {s}")
        text = ". ".join(parts) + "."
        text = normalize_text(text)

        # Extract any years mentioned
        import re
        years = [int(y) for y in re.findall(r'\b(19|20)\d{2}\b', text)]

        chunks.append({
            "chunk_id": f"csv_{idx}",
            "source": "election",
            "chunk_type": "csv_row",
            "text": text,
            "section_title": None,
            "year": years[0] if years else None,
            "keywords": extract_keywords(text),
            "row_index": int(idx)
        })
    return chunks
