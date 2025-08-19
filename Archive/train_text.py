"""
train_text.py  -  sentiment & MiniLM baselines
────────────────────────────────────────────────────────────────
Trains two logistic-regression models on CFPB complaint features:

    • T-Sent  - VADER compound sentiment only
    • T-Emb  - 384-d MiniLM sentence embeddings

Logs metrics to MLflow under experiment "text_models".
"""

from pathlib import Path
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, brier_score_loss,
)

# ───────── Paths ─────────
BASE        = Path(__file__).resolve().parents[1] / "data" / "processed"
TEXT_PQ     = BASE / "cfpb_text_feats.parquet"     # sentiment + emb_0 … emb_383
NUMERIC_PQ  = BASE / "lc_numeric_clean.parquet"    # for issue_d & target_default

KEY_COL = "id"            # numeric file’s unique loan ID
SAMPLE_ROWS = 100_000     # set to None to use all rows


# ───────── Utilities ─────────
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {path}")


def random_split(frame: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 stratified split."""
    train, temp = train_test_split(
        frame, test_size=0.30, random_state=seed, stratify=frame["target_default"]
    )
    valid, test = train_test_split(
        temp, test_size=0.50, random_state=seed, stratify=temp["target_default"]
    )
    return train, valid, test


def metric_dict(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    yhat = (p >= 0.5).astype(int)
    return dict(
        roc_auc=roc_auc_score(y, p),
        pr_auc=average_precision_score(y, p),
        accuracy=accuracy_score(y, yhat),
        f1=f1_score(y, yhat),
        brier=brier_score_loss(y, p),
    )


def build_matrix(df_slice: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    X = df_slice[cols].to_numpy(np.float32)
    y = df_slice["target_default"].to_numpy()
    return X, y


def load_joined_frame() -> pd.DataFrame:
    """Load text feats and numeric target, join on KEY_COL (string-typed)."""
    _require(TEXT_PQ)
    _require(NUMERIC_PQ)

    text_df = (
        pd.read_parquet(TEXT_PQ)
          .rename(columns={"row_id": KEY_COL})      # row_id → id
          .assign(**{KEY_COL: lambda d: d[KEY_COL].astype(str)})
    )

    numeric_cols = [KEY_COL, "issue_d", "target_default"]
    num_df = (
        pd.read_parquet(NUMERIC_PQ, columns=numeric_cols)
          .assign(**{KEY_COL: lambda d: d[KEY_COL].astype(str)})
    )

    df = text_df.merge(num_df, on=KEY_COL, how="left").dropna(subset=["target_default"])
    return df


def run_model(tag: str, df: pd.DataFrame, cols: List[str]) -> None:
    """Fit StandardScaler→LogReg(saga) pipeline; log metrics to MLflow; print summary."""
    mlflow.set_experiment("text_models")
    with mlflow.start_run(run_name=tag):
        t0 = time.perf_counter()

        train, valid, test = random_split(df, seed=42)
        X_tr, y_tr = build_matrix(train, cols)
        X_va, y_va = build_matrix(valid, cols)
        X_te, y_te = build_matrix(test,  cols)

        pipe = make_pipeline(
            # with_mean=False keeps behavior if inputs are/ever become sparse
            StandardScaler(with_mean=False),
            LogisticRegression(
                penalty="l2", C=1.0, solver="saga",
                max_iter=1000, n_jobs=-1, random_state=42,
            ),
        )

        # Fit on train+valid exactly like your original intent
        pipe.fit(np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va]))

        proba = pipe.predict_proba(X_te)[:, 1]
        m = metric_dict(y_te, proba)
        mlflow.log_metrics(m)

        dur = (time.perf_counter() - t0) / 60.0
        print(f"── {tag} ({dur:.1f} min) ──")
        for k, v in m.items():
            print(f"{k:8}: {v:.4f}")
        print("──────────────\n")


# ───────── Main ─────────
if __name__ == "__main__":
    df = load_joined_frame()

    # Optional downsample for speed/ram
    if SAMPLE_ROWS and SAMPLE_ROWS < len(df):
        df = df.sample(SAMPLE_ROWS, random_state=42)

    print("Loaded text dataframe :", df.shape)

    # Columns for each model
    if "sentiment" not in df.columns:
        raise KeyError("Expected 'sentiment' column not found in text features parquet.")
    sent_cols = ["sentiment"]

    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise KeyError("No embedding columns found (expected columns named like 'emb_0'...'emb_383').")

    run_model("T-Sent (VADER)",  sent_cols)
    run_model("T-Emb  (MiniLM)", emb_cols)




























