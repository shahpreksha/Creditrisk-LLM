"""
add_cfpb_cohorts.py
──────────────────────────────────────────────────────────────
Adds ZIP-3 × Year CFPB complaint aggregates to the engineered
LendingClub numeric dataset.

Output → lc_numeric_fe_plus_cfpb.parquet
"""

from pathlib import Path
import pandas as pd


# ───────── Paths ─────────
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"


# ───────── Utilities ─────────
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {path}")


def _find_col(cols, *needles):
    """Return the first column whose lowercase name contains all needle substrings."""
    low = {c: c.lower() for c in cols}
    for c, lc in low.items():
        if all(n in lc for n in needles):
            return c
    raise KeyError(f"Could not find a column matching: {needles}")


# ───────── 1 · CFPB structured table ─────────
def load_cfpb_with_keys() -> pd.DataFrame:
    _require(DATA / "cfpb.parquet")
    cfpb = pd.read_parquet(DATA / "cfpb.parquet")

    # detect key columns
    id_col   = _find_col(cfpb.columns, "complaint", "id")
    zip_col  = _find_col(cfpb.columns, "zip")
    date_col = _find_col(cfpb.columns, "date")

    cfpb = cfpb.rename(columns={id_col: "complaint_id",
                                zip_col: "zip_code",
                                date_col: "date_col"})
    return cfpb


# ───────── 2 · attach sentiment from text-feature parquet ─────────
def attach_sentiment(cfpb: pd.DataFrame) -> pd.DataFrame:
    _require(DATA / "cfpb_text_feats.parquet")
    feats = (
        pd.read_parquet(DATA / "cfpb_text_feats.parquet")
          .rename(columns={"row_id": "complaint_id"})
          .loc[:, ["complaint_id", "sentiment"]]
    )
    out = cfpb.merge(feats, on="complaint_id", how="left", validate="1:1")
    print("CFPB rows after merge:", out.shape)
    return out


# ───────── 3 · build ZIP-3 × year aggregates ─────────
def build_zip3_year_aggregates(cfpb: pd.DataFrame) -> pd.DataFrame:
    cfpb = cfpb.copy()

    # keep clean 5-digit ZIPs only
    cfpb["zip_code"] = cfpb["zip_code"].astype(str)
    mask = cfpb["zip_code"].str.match(r"^\d{5}$", na=False)
    cfpb = cfpb[mask].copy()

    # derive zip3 / year
    cfpb["zip3"] = cfpb["zip_code"].str[:3]
    cfpb["year"] = pd.to_datetime(cfpb["date_col"], errors="coerce").dt.year

    agg = (
        cfpb.groupby(["zip3", "year"])
            .agg(
                cfpb_cnt=("complaint_id", "nunique"),
                cfpb_sentiment_mean=("sentiment", "mean"),
            )
            .reset_index()
    )
    print("Aggregates:", agg.shape)
    return agg


# ───────── 4 · load engineered LendingClub data ─────────
def load_loans_with_issue_date() -> pd.DataFrame:
    fe_path = DATA / "lc_numeric_fe.parquet"
    raw_path = DATA / "lc_numeric_clean.parquet"
    _require(fe_path)
    loans = pd.read_parquet(fe_path)

    # recover issue_d if dropped during FE
    if "issue_d" not in loans.columns:
        _require(raw_path)
        if "id" not in loans.columns:
            raise KeyError("Engineered loans parquet lacks 'id' needed to merge 'issue_d'.")
        raw_dates = pd.read_parquet(raw_path, columns=["id", "issue_d"])
        loans = loans.merge(raw_dates, on="id", how="left")
        if loans["issue_d"].isna().any():
            raise ValueError("Could not recover issue_d for some loans.")

    # ensure ZIP present and normalized
    if "zip_code" not in loans.columns:
        raise KeyError("Expected 'zip_code' column not found in engineered loans file.")
    loans["zip_code"] = loans["zip_code"].astype(str)
    loans["zip3"] = loans["zip_code"].str[:3]
    loans["year"] = pd.to_datetime(loans["issue_d"], errors="coerce").dt.year
    return loans


# ───────── 5 · merge cohort features onto loans ─────────
def merge_cohorts(loans: pd.DataFrame, agg: pd.DataFrame) -> pd.DataFrame:
    out = loans.merge(agg, on=["zip3", "year"], how="left")
    out["cfpb_cnt"] = out["cfpb_cnt"].fillna(0)
    out["cfpb_sentiment_mean"] = out["cfpb_sentiment_mean"].fillna(0.0)
    print("\nCFPB feature summary:")
    print(out[["cfpb_cnt", "cfpb_sentiment_mean"]].describe().T)
    return out


def main() -> None:
    cfpb = load_cfpb_with_keys()
    cfpb = attach_sentiment(cfpb)
    agg = build_zip3_year_aggregates(cfpb)
    loans = load_loans_with_issue_date()
    merged = merge_cohorts(loans, agg)

    out_path = DATA / "lc_numeric_fe_plus_cfpb.parquet"
    merged.to_parquet(out_path, compression="zstd")
    print(f"\nSaved file -> {out_path.name}")


if __name__ == "__main__":
    main()




















    