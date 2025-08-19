"""
train_numeric.py

Trains two XGBoost baselines on LendingClub numeric data:
  1) Raw numeric (lc_numeric_clean.parquet)
  2) Engineered + CFPB features (lc_numeric_fe_plus_cfpb.parquet)
Time-based split:
  - train:  issue_d <= 2016-12-31
  - valid:  2017-01-01 ... 2017-12-31
  - test:   issue_d >= 2018-01-01

Logs metrics to MLflow and prints a compact summary.
"""

from pathlib import Path
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, brier_score_loss,
)
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost  # keep import to mirror original environment expectations


# ─── File paths ─────────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1] / "data" / "processed"
RAW_PQ = BASE / "lc_numeric_clean.parquet"          # has issue_d
FE_PQ  = BASE / "lc_numeric_fe_plus_cfpb.parquet"   # engineered, may lack issue_d

# ─── Time windows & sample size ─────────────────────────────
TRAIN_END = "2016-12-31"
VALID_END = "2017-12-31"
SAMPLE_ROWS = 100_000  # keep runtime & RAM modest

# ─── XGBoost hyper-params ──────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=250, max_depth=6, learning_rate=0.10,
    subsample=0.80, colsample_bytree=0.80,
    tree_method="hist", predictor="cpu_predictor",
    n_jobs=-1, random_state=42, eval_metric="auc",
)

# ─── Leakage guards ────────────────────────────────────────
LEAK_PATTERNS = [
    "pymnt", "payment", "recover", "collection",
    "out_prncp", "total_rec", "last_fico", "next_pymnt",
    "last_pymnt", "last_credit_pull", "mths_since",
    "settlement", "hardship",
]
LEAK_EXACT = ["loan_status"]

# ─── One-hot encoder (sparse) ──────────────────────────────
# NOTE: 'sparse_output' requires scikit-learn >= 1.2. If using <=1.1, use sparse=True.
enc = OneHotEncoder(drop="first", handle_unknown="ignore",
                    sparse_output=True, dtype=np.int8)


# ─── Helpers ───────────────────────────────────────────────
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {path}")


def _ensure_datetime_issue_d(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 'issue_d' is datetime[ns] for robust comparisons."""
    if "issue_d" not in df.columns:
        raise KeyError("Expected column 'issue_d' not found.")
    if not np.issubdtype(df["issue_d"].dtype, np.datetime64):
        df = df.copy()
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    return df


def chrono_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _ensure_datetime_issue_d(df)
    tr = df[df.issue_d <= TRAIN_END]
    va = df[(df.issue_d > TRAIN_END) & (df.issue_d <= VALID_END)]
    te = df[df.issue_d > VALID_END]
    return tr, va, te


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    yhat = (p >= 0.5).astype(int)
    return dict(
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
        accuracy=accuracy_score(y, yhat),
        f1=f1_score(y, yhat),
        brier=brier_score_loss(y, p),
    )


def strip_leaks(df: pd.DataFrame) -> pd.DataFrame:
    patt = tuple(LEAK_PATTERNS)
    cols = [
        c for c in df.columns
        if (c.lower().startswith(patt) or any(p in c.lower() for p in patt))
        or c in LEAK_EXACT
    ]
    return df.drop(columns=cols, errors="ignore")


def prep(df: pd.DataFrame, fit_encoder: bool):
    X = strip_leaks(df).drop(columns=["target_default"])
    y = df["target_default"].to_numpy()

    if "issue_d" in X:
        X = X.drop(columns=["issue_d"])

    cat = X.select_dtypes("object")
    num = X.drop(columns=cat.columns).to_numpy(np.float32)
    X_cat = enc.fit_transform(cat) if fit_encoder else enc.transform(cat)
    return sparse.hstack([num, X_cat]).tocsr(), y


# ─── Ensure engineered frame has issue_d ───────────────────
def load_engineered_with_dates() -> pd.DataFrame:
    _require(FE_PQ)
    fe = pd.read_parquet(FE_PQ)
    if "issue_d" in fe.columns:
        return fe

    _require(RAW_PQ)
    raw_cols = pd.read_parquet(RAW_PQ, nrows=0).columns
    id_col = next((c for c in fe.columns if c in raw_cols and c.lower().endswith("id")), None)
    if id_col is None:
        raise ValueError("Could not find a common ID column to merge 'issue_d'.")

    dates = pd.read_parquet(RAW_PQ, columns=[id_col, "issue_d"])
    merged = fe.merge(dates, on=id_col, how="left")
    missing = merged["issue_d"].isna().sum()
    if missing:
        print(f"Warning: {missing:,} rows missing 'issue_d' after merge.")
    return merged


# ─── Training wrapper ───────────────────────────────────────
def run_model(tag: str, df: pd.DataFrame) -> None:
    if SAMPLE_ROWS:
        df = df.sample(SAMPLE_ROWS, random_state=42)

    mlflow.set_experiment("numeric_models")
    with mlflow.start_run(run_name=tag):
        t0 = time.perf_counter()
        tr, va, te = chrono_split(df)

        Xtr, ytr = prep(tr, True)
        Xva, yva = prep(va, False)
        Xte, yte = prep(te, False)

        print(f"[{tag}] train {Xtr.shape}  valid {Xva.shape}  test {Xte.shape}  "
              f"prep={time.perf_counter()-t0:.1f}s")

        mdl = XGBClassifier(**XGB_PARAMS)
        mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)

        prob = mdl.predict_proba(Xte)[:, 1]
        metrics = metric_dict(yte, prob)
        mlflow.log_metrics(metrics)

        print(f"── {tag} ──")
        for k, v in metrics.items():
            print(f"{k:8}: {v:.4f}")
        print("────────────\n")


# ─── Main ───────────────────────────────────────────────────
if __name__ == "__main__":
    _require(RAW_PQ)
    raw_df = pd.read_parquet(RAW_PQ)
    run_model("N-Base_raw 100k", raw_df)

    fe_df = load_engineered_with_dates()
    run_model("N-FE_engineered+CFPB 100k", fe_df)



















