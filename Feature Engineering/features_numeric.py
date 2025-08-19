"""
Feature Engineering – Numeric LendingClub Data
------------------------------------------------
• Reads the cleaned parquet produced by eda_numeric.py
• Applies a handful of high-impact, low-effort transforms:
      1) log-scale annual_inc
      2) cap / flag DTI outliers
      3) ordinal-encode loan grade
      4) one-hot (but de-rare) loan purpose
      5) extract issue_year + issue_quarter
• Writes a train-ready parquet: lc_numeric_fe.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ─────────────────── project paths ──────────────────────────────────────────
BASE_DIR       = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed"
INPUT_FILE     = DATA_PROCESSED / "lc_numeric_clean.parquet"
OUTPUT_FILE    = DATA_PROCESSED / "lc_numeric_fe.parquet"


# ─────────────────── helpers ────────────────────────────────────────────────
def add_log_income(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p(income). Keeps/overwrites 'annual_inc_log' if already present."""
    if "annual_inc" in df.columns:
        df["annual_inc_log"] = np.log1p(df["annual_inc"])
    return df


def tidy_dti(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace impossible DTI values (>60), and create a flag before imputation.
    After capping: set DTI>60 → NaN → median-impute.
    """
    if "dti" not in df.columns:
        return df
    df["dti_flag_missing"] = df["dti"] > 60
    df.loc[df["dti"] > 60, "dti"] = np.nan
    df["dti"] = df["dti"].fillna(df["dti"].median())
    return df


def encode_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Map letter grade to an ordered integer 1…7 (A→1, …, G→7)."""
    if "grade" not in df.columns:
        return df
    mapping = {g: i for i, g in enumerate("ABCDEFG", start=1)}
    df["grade_num"] = df["grade"].map(mapping).astype("Int8")
    return df


def one_hot_purpose(df: pd.DataFrame, rare_threshold: float = 0.01) -> pd.DataFrame:
    """
    One-hot encode 'purpose', collapsing rare categories (< rare_threshold share) into 'Other'.
    Leaves dataframe unchanged if 'purpose' not present.
    """
    if "purpose" not in df.columns:
        return df
    freq = df["purpose"].value_counts(normalize=True)
    common = freq[freq >= rare_threshold].index
    df["purpose_mod"] = np.where(df["purpose"].isin(common), df["purpose"], "Other")
    purpose_dummies = pd.get_dummies(df["purpose_mod"], prefix="purpose")
    df = df.join(purpose_dummies).drop(columns=["purpose_mod"])
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract year and quarter from issue_d (expects datetime; attempts safe cast if needed)."""
    if "issue_d" not in df.columns:
        return df
    if not np.issubdtype(df["issue_d"].dtype, np.datetime64):
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce", format="%b-%Y")
    df["issue_year"] = df["issue_d"].dt.year.astype("int16")
    df["issue_quarter"] = df["issue_d"].dt.quarter.astype("int8")
    return df


# ─────────────────── main pipeline ──────────────────────────────────────────
def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean parquet not found: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} cols")

    # Apply each transformation in place (order preserved)
    df = (
        df.pipe(add_log_income)
          .pipe(tidy_dti)
          .pipe(encode_grade)
          .pipe(one_hot_purpose)
          .pipe(add_time_features)
    )

    # Optional: drop columns no longer needed for modelling
    drop_cols = ["issue_d", "purpose"]  # keep raw 'grade' if desired for SHAP
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")

    df.to_parquet(OUTPUT_FILE)
    print(f"Feature-engineered parquet written -> {OUTPUT_FILE.name}")
    print(f"Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")


if __name__ == "__main__":
    main()





































