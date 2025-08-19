"""
Convert raw LendingClub and CFPB CSVs to Parquet.

Behavior preserved:
- Reads from: <project-root>/data/raw
    - accepted_2007_to_2018Q4.csv
    - cfpb_complaints.csv
- Writes to:  <project-root>/data/processed
    - lendingclub.parquet
    - cfpb.parquet
- Prints: "Raw CSVs converted to Parquet."
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import polars as pl


# ─────────────────────────── Paths ───────────────────────────
BASE = Path(__file__).resolve().parents[1]
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

LC_CSV = RAW / "accepted_2007_to_2018Q4.csv"
CFPB_CSV = RAW / "cfpb_complaints.csv"

LC_PARQUET = PROC / "lendingclub.parquet"
CFPB_PARQUET = PROC / "cfpb.parquet"


# ─────────────────────── Helper utilities ─────────────────────
def _require(path: Path, label: Optional[str] = None) -> None:
    """Raise a clear error if a required file is missing."""
    if not path.exists():
        name = f"{label or 'file'}"
        raise FileNotFoundError(
            f"Expected {name} not found:\n  {path}\n"
            "Make sure the raw dataset is present in data/raw."
        )


# ───────────────────────── Converters ─────────────────────────
def load_lendingclub() -> None:
    """Read LendingClub CSV with pandas and write Parquet."""
    _require(LC_CSV, "LendingClub CSV")
    df = pd.read_csv(LC_CSV, low_memory=False)
    try:
        df.to_parquet(LC_PARQUET)  # keep defaults to preserve behavior
    except Exception as e:
        # Common case: pyarrow/fastparquet not installed
        raise RuntimeError(
            "Failed to write lendingclub.parquet. "
            "Ensure 'pyarrow' or 'fastparquet' is installed."
        ) from e


def load_cfpb() -> None:
    """Read CFPB CSV with polars and write Parquet."""
    _require(CFPB_CSV, "CFPB CSV")
    df = pl.read_csv(CFPB_CSV)
    df.write_parquet(CFPB_PARQUET)  # keep defaults to preserve behavior


# ──────────────────────────── CLI ─────────────────────────────
if __name__ == "__main__":
    load_lendingclub()
    load_cfpb()
    print("Raw CSVs converted to Parquet.")

















