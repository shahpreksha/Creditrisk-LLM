"""
Feature engineering pipeline
Combines:
    • Clean numeric LendingClub data   (lc_numeric_clean.parquet)
    • Clean text-feature parquet       (cfpb_text_feats.parquet)
Outputs:
    • features.parquet  (X matrix)
    • target.parquet    (y vector)

Run:  python src/feature_engineering.py
"""

from pathlib import Path
import numpy as np
import polars as pl


# ────────────────── Paths ───────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parents[1]
DATA_PROC = BASE_DIR / "data" / "processed"
NUMERIC_PQ = DATA_PROC / "lc_numeric_clean.parquet"
TEXT_PQ    = DATA_PROC / "cfpb_text_feats.parquet"
X_OUT      = DATA_PROC / "features.parquet"
Y_OUT      = DATA_PROC / "target.parquet"


# ────────────────── Helpers ─────────────────────────────────────────────────
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {path}")


def add_log_longtails(df: pl.DataFrame) -> pl.DataFrame:
    """Add log1p transforms for long-tailed numeric columns, if present."""
    exprs = []
    if "loan_amnt" in df.columns:
        exprs.append(pl.col("loan_amnt").log1p().alias("loan_amnt_log"))
    if "annual_inc" in df.columns:
        exprs.append(pl.col("annual_inc").log1p().alias("annual_inc_log"))
    return df.with_columns(exprs) if exprs else df


def add_simple_ratios(df: pl.DataFrame) -> pl.DataFrame:
    """Add installment-to-income ratio when columns exist."""
    if {"installment", "annual_inc"} <= set(df.columns):
        return df.with_columns(
            (pl.col("installment") / (pl.col("annual_inc") + 1)).alias("inst_to_inc")
        )
    return df


def encode_grade(df: pl.DataFrame) -> pl.DataFrame:
    """Ordinal + one-hot encode loan grade (A…G) when present."""
    if "grade" not in df.columns:
        return df
    grade_map = {g: i for i, g in enumerate("ABCDEFG", 1)}
    out = df.with_columns(pl.col("grade").replace(grade_map).alias("grade_num"))
    # One-hot as grade_A … grade_G
    dummies = out.select("grade").to_dummies()
    return out.drop("grade").hstack(dummies)


def derare_and_ohe_purpose(df: pl.DataFrame, min_count: int = 5_000) -> pl.DataFrame:
    """Bucket rare 'purpose' values to 'Other' and one-hot encode."""
    if "purpose" not in df.columns:
        return df

    vc = df["purpose"].value_counts()
    # Polars version compatibility: counts column may be 'count' or 'counts'
    cnt_col = "count" if "count" in vc.columns else "counts"
    val_col = "purpose" if "purpose" in vc.columns else vc.columns[0]

    rare_values = vc.filter(pl.col(cnt_col) < min_count)[val_col].to_list()

    tmp_col = "purpose_tmp"
    out = df.with_columns(
        pl.when(pl.col("purpose").is_in(rare_values))
          .then(pl.lit("Other"))
          .otherwise(pl.col("purpose"))
          .alias(tmp_col)
    )
    dummies = (
        out.select(tmp_col)
           .to_dummies()
           .rename(lambda c: c.replace(f"{tmp_col}_", "purpose_"))
    )
    return out.drop(["purpose", tmp_col]).hstack(dummies)


def attach_text_features(num_df: pl.DataFrame, text_df: pl.DataFrame) -> pl.DataFrame:
    """
    Demo merge: align by positional key (tmp_key).
    Advanced projects should link via a real key (e.g., ZIP+year, lender, etc.).
    """
    text_df = text_df.rename({"row_id": "text_row_id", "sentiment": "cfpb_sentiment"})
    num_df  = num_df.with_columns(pl.Series("tmp_key", np.arange(num_df.height)))
    text_df = text_df.with_columns(pl.Series("tmp_key", np.arange(text_df.height)))
    merged  = num_df.join(text_df, on="tmp_key", how="left").drop("tmp_key")
    # Neutral sentiment for missing narratives
    return merged.with_columns(pl.col("cfpb_sentiment").fill_null(pl.lit(0.0)))


# ────────────────── Main ────────────────────────────────────────────────────
def main() -> None:
    _require(NUMERIC_PQ)
    _require(TEXT_PQ)

    num_df  = pl.read_parquet(NUMERIC_PQ)
    text_df = pl.read_parquet(TEXT_PQ)

    print(f"Numeric rows : {num_df.height:,}")
    print(f"Text rows    : {text_df.height:,}")

    # 1) Numeric feature construction
    num_df = (
        num_df
        .pipe(add_log_longtails)
        .pipe(add_simple_ratios)
        .pipe(encode_grade)
        .pipe(derare_and_ohe_purpose)
    )

    # 2) Attach text features (demo merge by tmp_key)
    merged = attach_text_features(num_df, text_df)
    print(f"Merged shape : {merged.shape}")

    # 3) Separate X and y
    if "target_default" not in merged.columns:
        raise KeyError("Expected column 'target_default' not found after merge.")
    y = merged.select("target_default")
    X = merged.drop("target_default")

    # 4) Save artifacts
    X.write_parquet(X_OUT, compression="zstd")
    y.write_parquet(Y_OUT, compression="zstd")
    print(f"Features saved -> {X_OUT.name}   ({X.width} cols)")
    print(f"Target   saved -> {Y_OUT.name}")


if __name__ == "__main__":
    main()




















































