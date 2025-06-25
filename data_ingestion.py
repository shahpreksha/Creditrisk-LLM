from pathlib import Path
import pandas as pd
import polars as pl

BASE = Path(__file__).resolve().parents[1]   
RAW  = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

def load_lendingclub():
    df = pd.read_csv(RAW / "accepted_2007_to_2018Q4.csv",
                 low_memory=False)
    df.to_parquet(PROC / "lendingclub.parquet")


def load_cfpb():
    df = pl.read_csv(RAW / "cfpb_complaints.csv")
    df.write_parquet(PROC / "cfpb.parquet")

if __name__ == "__main__":
    load_lendingclub()
    load_cfpb()
    print("Raw CSVs converted to Parquet.")
