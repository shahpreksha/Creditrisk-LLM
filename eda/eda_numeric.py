"""
Numeric EDA for LendingClub accepted-loan data
Converted directly from eda_lendingclub.ipynb

This script loads the cleaned LendingClub parquet, creates a binary target,
does basic missingness and distribution checks, and saves the updated parquet.
Behavior is preserved exactly (plots, transforms, and save path).
"""

# ────────────────── Imports & file paths ─────────────────────────────────────
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Resolve <project-root>/data/processed  (one level above this script)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PROCESSED = BASE_DIR / "data" / "processed"

# Name of the parquet file you saved earlier
PARQUET_FILE = DATA_PROCESSED / "lc_numeric_clean.parquet"


def _require_file(path: Path) -> None:
    """Ensure required parquet exists; raise with a clear message otherwise."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected file not found:\n  {path}\n"
            "Make sure you have run the numeric cleaning pipeline and saved the parquet."
        )


def _plot_default_rate(df: pd.DataFrame, cat_col: str, top_n: int = 10) -> None:
    """Bar plot of default rate by a categorical column (top N levels by rate)."""
    rate = (
        df.groupby(cat_col)["target_default"]
        .mean()
        .sort_values(ascending=False)[:top_n]
    )
    rate.plot.bar()
    plt.ylabel("Default rate")
    plt.title(f"Default rate by {cat_col}")
    plt.show()


def main() -> None:
    # ────────────────── Load once ────────────────────────────────────────────
    _require_file(PARQUET_FILE)
    df = pd.read_parquet(PARQUET_FILE)
    print(f"Loaded {len(df):,} rows  |  {df.shape[1]} columns\n")

    # ────────────────── Quick sample for eyeballing ─────────────────────────
    # (Kept for parity with original script; not used downstream.)
    sample_df = df.sample(100_000, random_state=42)  # noqa: F841

    # ────────────────── Basic structure & memory check ──────────────────────
    print("DataFrame info:")
    df.info(memory_usage="deep")
    print()

    # ────────────────── Create binary target (0 = good, 1 = bad) ────────────
    good_status = ["Fully Paid", "Current"]
    bad_status = [
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    ]

    df = df[df["loan_status"].isin(good_status + bad_status)].copy()
    df["target_default"] = df["loan_status"].isin(bad_status).astype(int)

    df["target_default"].value_counts(normalize=True).plot.bar(
        title="Default vs Good (%)"
    )
    plt.show()

    # ────────────────── Missing-value heat-map & prune sparse cols ──────────
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isna(), cbar=False)
    plt.title("Missing-value map")
    plt.show()

    missing_pct = df.isna().mean().sort_values(ascending=False)
    drop_cols = missing_pct[missing_pct > 0.80].index
    df.drop(columns=drop_cols, inplace=True)
    print(f"Dropped {len(drop_cols)} columns with >80% missing values.\n")

    # ────────────────── Numeric / categorical splits ────────────────────────
    num_cols: List[str] = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols: List[str] = df.select_dtypes(include="object").columns.tolist()
    print(f"{len(num_cols)} numeric  |  {len(cat_cols)} categorical/text\n")

    # ────────────────── Convert % strings to floats ─────────────────────────
    pct_candidates = ["int_rate", "revol_util", "annual_inc_joint"]
    pct_cols = [c for c in pct_candidates if c in df.columns]

    for col in pct_cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.rstrip("%")
            .replace({"": np.nan, "nan": np.nan})
            .astype(float)
        )
    print("Converted columns:", pct_cols, "\n")

    # ────────────────── Initial histograms ──────────────────────────────────
    plot_cols = ["loan_amnt", "annual_inc", "dti", "int_rate"]
    # Keep original assumption that these exist; will only plot existing subset.
    existing_plot_cols = [c for c in plot_cols if c in df.columns]
    if existing_plot_cols:
        df[existing_plot_cols].hist(figsize=(10, 6))
        plt.tight_layout()
        plt.show()

    # ────────────────── Outlier handling & log income ───────────────────────
    upper_inc = df["annual_inc"].quantile(0.99)  # cap 99th percentile
    df["annual_inc"] = df["annual_inc"].clip(upper=upper_inc)
    df["annual_inc_log"] = np.log1p(df["annual_inc"])

    df.loc[df["dti"] > 60, "dti"] = np.nan
    df["dti"].fillna(df["dti"].median(), inplace=True)

    # Histograms after cleanup (same selected columns as original)
    if existing_plot_cols:
        df[existing_plot_cols].hist(figsize=(10, 6))
        plt.tight_layout()
        plt.show()

    # ────────────────── Correlation heat-map ────────────────────────────────
    # Preserve original behavior: correlation uses num_cols captured earlier,
    # so 'annual_inc_log' is intentionally excluded from this heatmap.
    if num_cols:
        corr = df[num_cols].corr(method="spearman")
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, cmap="vlag", center=0)
        plt.title("Spearman correlations")
        plt.show()

    # ────────────────── Default rate by key categoricals ────────────────────
    if "grade" in df.columns:
        _plot_default_rate(df, "grade")
    if "purpose" in df.columns:
        _plot_default_rate(df, "purpose")
    if "emp_length" in df.columns:
        _plot_default_rate(df, "emp_length")

    # ────────────────── Quarterly default trend ─────────────────────────────
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y")
        quarterly = (
            df.set_index("issue_d").resample("Q")["target_default"].mean()
        )
        quarterly.plot(marker="o", figsize=(10, 4))
        plt.ylabel("Quarterly default rate")
        plt.title("Default rate over time")
        plt.show()

    # ────────────────── Save cleaned numeric parquet ────────────────────────
    # Preserve original: overwrite same file name.
    df.to_parquet(PARQUET_FILE)
    print(f"Clean numeric dataset saved to {PARQUET_FILE.name}")


if __name__ == "__main__":
    main()



























