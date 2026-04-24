# src/ingestion/load_csv.py
# Author: [Your Name] | Index: [Your Index Number]
# Loads Ghana election results from CSV using pandas

import pandas as pd
from typing import Tuple


def load_csv(path: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load election results CSV.
    Returns the raw DataFrame and basic metadata.
    """
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    metadata = {
        "source": path,
        "rows": len(df),
        "columns": list(df.columns),
        "type": "election_results"
    }
    return df, metadata
