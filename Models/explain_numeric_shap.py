"""
explain_numeric_shap.py
──────────────────────────────────────────────────────────────
Retrains the engineered-plus-CFPB XGBoost model (200k rows),
computes SHAP values on a 5k-row sample, and saves:

- reports/figures/shap_summary_bar.png
- data/processed/shap_sample.csv
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# ── Paths & Params ───────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
FIGS = BASE / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

FE_PQ = DATA / "lc_numeric_fe_plus_cfpb.parquet"
RAW_PQ = DATA / "lc_numeric_clean.parquet"

SAMPLE_ROWS = 200_000
SHAP_ROWS = 5_000

XGB_PARAMS = dict(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.10,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method="hist",
    predictor="cpu_predictor",
    n_jobs=-1,
    random_state=42,
    eval_metric="auc",
)

LEAK_PATTERNS = [
    "pymnt", "payment", "recover", "collection",
    "out_prncp", "total_rec", "last_fico", "next_pymnt",
    "last_pymnt", "last_credit_pull", "mths_since",
    "settlement", "hardship",
]
LEAK_EXACT = ["loan_status"]

# Keep encoder at module scope to mirror original behavior.
# (Note: 'sparse_output' requires scikit-learn >=1.2. If using <=1.1, switch to 'sparse=True'.)
enc = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=True, dtype=np.int8)


# ── Helpers ─────────────────────────────────────────────────
def _require(path: Path) -> None:
    """Raise a clear error if a required parquet file is missing."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {path}")


def strip_leaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that leak target information:
    - Any column whose lowercase name starts with or contains a pattern in LEAK_PATTERNS
    - Any exact match in LEAK_EXACT
    """
    patt = tuple(LEAK_PATTERNS)
    drop_cols = [
        c for c in df.columns
        if (c.lower().startswith(patt) or any(p in c.lower() for p in patt)) or (c in LEAK_EXACT)
    ]
    return df.drop(columns=drop_cols, errors="ignore")


def prep(df: pd.DataFrame, fit: bool) -> Tuple[sparse.csr_matrix, np.ndarray, List[str]]:
    """
    Prepare features for modeling:
    - Removes leak columns
    - Splits numeric vs categorical
    - One-hot encodes categoricals (fit controls encoder fitting)
    - Returns (X_sparse, y, feature_names)
    """
    # Require target and date columns; mirror original assumptions
    for col in ("target_default", "issue_d"):
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in dataframe.")

    X = strip_leaks(df).drop(columns=["target_default", "issue_d"])
    y = df["target_default"].to_numpy()

    cat_df = X.select_dtypes("object")
    num_df = X.drop(columns=cat_df.columns)
    num_cols = num_df.columns.tolist()

    # Numeric block — float32 for memory and XGB speed
    num = num_df.to_numpy(np.float32)

    # Categorical block — sparse one-hot
    X_cat = enc.fit_transform(cat_df) if fit else enc.transform(cat_df)
    cat_cols = enc.get_feature_names_out().tolist()

    feat_names = num_cols + cat_cols
    X_all = sparse.hstack([num, X_cat]).tocsr()
    return X_all, y, feat_names


def main() -> None:
    # ── Load engineered data; ensure issue_d present ─────────
    _require(FE_PQ)
    fe = pd.read_parquet(FE_PQ)

    if "issue_d" not in fe.columns:
        _require(RAW_PQ)
        # Only bring in what's required to keep memory in check
        raw_dates = pd.read_parquet(RAW_PQ, columns=["id", "issue_d"])
        fe = fe.merge(raw_dates, on="id", how="left")

    fe_sample = fe.sample(SAMPLE_ROWS, random_state=42).reset_index(drop=True)
    print("Training rows:", len(fe_sample))

    # ── Train model ──────────────────────────────────────────
    t0 = time.perf_counter()
    X_train, y_train, feat_names = prep(fe_sample, fit=True)

    model = XGBClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train)
    print(f"Model trained in {(time.perf_counter() - t0):.1f}s  |  features: {X_train.shape[1]:,}")

    # ── SHAP on 5k-row sample ────────────────────────────────
    shap_sample = fe_sample.sample(SHAP_ROWS, random_state=1).reset_index(drop=True)
    X_shap, y_shap, _ = prep(shap_sample, fit=False)

    # For tree models, TreeExplainer is the right choice; keep feature_names for plotting
    explainer = shap.TreeExplainer(model, feature_names=feat_names)
    shap_values = explainer.shap_values(X_shap)

    # Save raw SHAP values (same path/name as original)
    pd.DataFrame(shap_values, columns=feat_names).to_csv(DATA / "shap_sample.csv", index=False)
    print("Saved to data/processed/shap_sample.csv")

    # Summary bar plot (top 20)
    plt.figure(figsize=(6, 4))
    shap.summary_plot(
        shap_values, X_shap, feature_names=feat_names,
        plot_type="bar", max_display=20, show=False
    )
    plt.tight_layout()
    plt.savefig(FIGS / "shap_summary_bar.png", dpi=150)
    plt.close()
    print("Saved to reports/figures/shap_summary_bar.png")


if __name__ == "__main__":
    main()








































