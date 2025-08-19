"""
visualize_metrics.py
──────────────────────────────────────────────────────────────
Plots ROC, PR, confusion-matrix, and calibration curves for
the tuned numeric + CFPB XGBoost model.

Run:
    python src/visualize_metrics.py
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    roc_auc_score, average_precision_score,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from xgboost import XGBClassifier


# ── Paths & params ─────────────────────────────────────────
BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "processed"
FIGS = BASE / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

FE_PQ  = DATA / "lc_numeric_fe_plus_cfpb.parquet"
RAW_PQ = DATA / "lc_numeric_clean.parquet"

TRAIN_END, VALID_END = "2016-12-31", "2017-12-31"
SAMPLE_ROWS = 150_000          # subsample for plotting

TUNED_PARAMS = dict(
    n_estimators      = 272,
    max_depth         = 6,
    learning_rate     = 0.0999,
    subsample         = 0.9950,
    colsample_bytree  = 0.8989,
    min_child_weight  = 4,
    gamma             = 4.2310,
    tree_method       = "hist",
    predictor         = "cpu_predictor",
    n_jobs            = -1,
    random_state      = 42,
    eval_metric       = "auc",
)

LEAK_PATTERNS = [
    "pymnt","payment","recover","collection","out_prncp","total_rec",
    "last_fico","next_pymnt","last_pymnt","last_credit_pull",
    "mths_since","settlement","hardship",
]
LEAK_EXACT = ["loan_status"]

# NOTE: 'sparse_output' requires scikit-learn >=1.2. If on <=1.1, use sparse=True.
enc = OneHotEncoder(drop="first", handle_unknown="ignore",
                    sparse_output=True, dtype=np.int8)


# ── Helpers ────────────────────────────────────────────────
def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {path}")


def strip_leaks(df: pd.DataFrame) -> pd.DataFrame:
    patt = tuple(LEAK_PATTERNS)
    bad  = [c for c in df.columns
            if (c.lower().startswith(patt) or any(p in c.lower() for p in patt))
               or c in LEAK_EXACT]
    return df.drop(columns=bad, errors="ignore")


def _ensure_datetime_issue_d(df: pd.DataFrame) -> pd.DataFrame:
    if "issue_d" not in df.columns:
        raise KeyError("Expected column 'issue_d' not found.")
    if not np.issubdtype(df["issue_d"].dtype, np.datetime64):
        df = df.copy()
        df["issue_d"] = pd.to_datetime(df["issue_d"], errors="coerce")
    return df


def prep(df: pd.DataFrame, fit: bool = False) -> Tuple[sparse.csr_matrix, np.ndarray]:
    X = strip_leaks(df).drop(columns=["target_default", "issue_d"])
    y = df["target_default"].to_numpy()

    cats = X.select_dtypes("object")
    nums = X.drop(columns=cats.columns).to_numpy(np.float32)

    if cats.shape[1] == 0:
        X_cat = csr_matrix((len(X), 0), dtype=np.int8)
    else:
        X_cat = enc.fit_transform(cats) if fit else enc.transform(cats)

    return sparse.hstack([nums, X_cat]).tocsr(), y


def chrono_split(df: pd.DataFrame):
    df = _ensure_datetime_issue_d(df)
    tr = df[df.issue_d <= TRAIN_END]
    va = df[(df.issue_d > TRAIN_END) & (df.issue_d <= VALID_END)]
    te = df[df.issue_d > VALID_END]
    return tr, va, te


# ── Main flow ──────────────────────────────────────────────
def main() -> None:
    _require(FE_PQ)
    df = pd.read_parquet(FE_PQ)

    if "issue_d" not in df.columns:
        _require(RAW_PQ)
        if "id" not in df.columns:
            raise KeyError("Engineered parquet lacks 'id' needed to merge 'issue_d'.")
        df = df.merge(pd.read_parquet(RAW_PQ, columns=["id", "issue_d"]),
                      on="id", how="left")

    # Subsample before splitting (for speed/clarity of plots)
    if SAMPLE_ROWS and SAMPLE_ROWS < len(df):
        df = df.sample(SAMPLE_ROWS, random_state=42).reset_index(drop=True)

    train_df, _, test_df = chrono_split(df)
    X_tr, y_tr = prep(train_df, fit=True)
    X_te, y_te = prep(test_df,  fit=False)

    # Train tuned model
    model = XGBClassifier(**TUNED_PARAMS)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    ap    = average_precision_score(y_te, proba)
    print(f"ROC-AUC {auc:.4f} | PR-AUC {ap:.4f}")

    # 1) ROC
    fpr, tpr, _ = roc_curve(y_te, proba)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", lw=1, c="grey")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC curve"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGS / "roc_curve.png", dpi=150); plt.close()

    # 2) Precision–Recall
    prec, rec, _ = precision_recall_curve(y_te, proba)
    plt.figure(figsize=(4, 4))
    plt.plot(rec, prec, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall curve"); plt.legend()
    plt.tight_layout(); plt.savefig(FIGS / "pr_curve.png", dpi=150); plt.close()

    # 3) Confusion matrix @ 0.50
    y_pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_te, y_pred)

    plt.figure(figsize=(3, 3))
    plt.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            plt.text(j, i, cm[i, j], ha="center", va="center", color=color, fontsize=11)
    plt.xticks([0, 1], ["Good", "Bad"])   # Predicted
    plt.yticks([0, 1], ["Good", "Bad"])   # True
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.title("Confusion matrix (0.50 cut-off)")
    plt.tight_layout(); plt.savefig(FIGS / "conf_matrix.png", dpi=150); plt.close()

    # 4) Calibration curve (20 bins)
    prob_true, prob_pred = calibration_curve(y_te, proba, n_bins=20)
    plt.figure(figsize=(4, 4))
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], "--", c="grey")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve (20 bins)")
    plt.tight_layout(); plt.savefig(FIGS / "calibration_curve.png", dpi=150); plt.close()

    print("ROC, PR, confusion-matrix & calibration plots saved to reports/figures/")


if __name__ == "__main__":
    main()









































