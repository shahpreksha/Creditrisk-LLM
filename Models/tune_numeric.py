"""
tune_numeric.py  -  Optuna tuning + Platt calibration with pruning
──────────────────────────────────────────────────────────────────
Default: 100 k rows for tuning, 20 k for hold-out, 20 trials.
Adjust SAMPLE_ROWS, TEST_ROWS, N_TRIALS near the top if needed.
"""

# ── Imports ────────────────────────────────────────────────
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import mlflow

from scipy import sparse
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from optuna.integration import XGBoostPruningCallback
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)

# ── Tuning sample sizes & trials ───────────────────────────
SAMPLE_ROWS = 100_000    # training+valid rows used in tuning
TEST_ROWS   = 20_000     # hold-out rows for final metric
N_TRIALS    = 20         # Optuna trials

# ── Paths ─────────────────────────────────────────────────
DATA   = Path(__file__).resolve().parents[1] / "data" / "processed"
FE_PQ  = DATA / "lc_numeric_fe_plus_cfpb.parquet"
RAW_PQ = DATA / "lc_numeric_clean.parquet"

# ── Leakage guards & encoder ──────────────────────────────
LEAK_PATTERNS = [
    "pymnt","payment","recover","collection","out_prncp","total_rec",
    "last_fico","next_pymnt","last_pymnt","last_credit_pull",
    "mths_since","settlement","hardship",
]
LEAK_EXACT = ["loan_status"]

# NOTE: 'sparse_output' requires scikit-learn >=1.2. If on <=1.1, use sparse=True.
enc = OneHotEncoder(drop="first", handle_unknown="ignore",
                    sparse_output=True, dtype=np.int8)

def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected file not found:\n  {path}")

def strip_leaks(df: pd.DataFrame) -> pd.DataFrame:
    patt = tuple(LEAK_PATTERNS)
    bad  = [c for c in df.columns
            if (c.lower().startswith(patt) or any(p in c.lower() for p in patt))
            or c in LEAK_EXACT]
    return df.drop(columns=bad, errors="ignore")

def make_matrix(df: pd.DataFrame, fit_enc: bool):
    X = strip_leaks(df).drop(columns=["target_default", "issue_d"])
    y = df["target_default"].to_numpy()
    cat = X.select_dtypes("object")
    num = X.drop(columns=cat.columns).to_numpy(np.float32)
    X_cat = enc.fit_transform(cat) if fit_enc else enc.transform(cat)
    return sparse.hstack([num, X_cat]).tocsr(), y

# ── Load engineered + ensure issue_d ──────────────────────
_require(FE_PQ)
fe = pd.read_parquet(FE_PQ)
if "issue_d" not in fe.columns:
    _require(RAW_PQ)
    if "id" not in fe.columns:
        raise KeyError("Engineered parquet lacks 'id' needed to merge 'issue_d'.")
    fe = fe.merge(pd.read_parquet(RAW_PQ, columns=["id", "issue_d"]),
                  on="id", how="left")

# Sub-sample for tuning
fe_sm = fe.sample(SAMPLE_ROWS + TEST_ROWS, random_state=42)
train_df, test_df = train_test_split(
    fe_sm, test_size=TEST_ROWS, stratify=fe_sm["target_default"], random_state=42
)

X_train, y_train = make_matrix(train_df, fit_enc=True)
X_test,  y_test  = make_matrix(test_df,  fit_enc=False)

# ── Optuna objective with pruning ─────────────────────────
def objective(trial: optuna.Trial):
    params = dict(
        n_estimators      = trial.suggest_int( "n_estimators",     200, 400),
        max_depth         = trial.suggest_int( "max_depth",        4,   8),
        learning_rate     = trial.suggest_float("lr",              0.05, 0.20),
        subsample         = trial.suggest_float("subsample",       0.60, 1.00),
        colsample_bytree  = trial.suggest_float("colsample",       0.60, 1.00),
        min_child_weight  = trial.suggest_int( "min_child_weight", 1,   8),
        gamma             = trial.suggest_float("gamma",           0.0, 5.0),
        tree_method       = "hist",
        predictor         = "cpu_predictor",
        n_jobs            = -1,
        random_state      = 42,
        eval_metric       = "auc",
    )
    mdl = XGBClassifier(**params)
    mdl.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
        callbacks=[XGBoostPruningCallback(trial, "validation_0-auc")],
    )
    return roc_auc_score(y_test, mdl.predict_proba(X_test)[:, 1])

pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
study = optuna.create_study(direction="maximize", pruner=pruner)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
best_params = study.best_params
print("Best params:", best_params)

# ── Retrain on full tuning sample with best params ────────
# Map Optuna suggestion names back to XGBoost kwargs
fixed_params = dict(best_params)
if "lr" in fixed_params:
    fixed_params["learning_rate"] = fixed_params.pop("lr")
if "colsample" in fixed_params:
    fixed_params["colsample_bytree"] = fixed_params.pop("colsample")

best = XGBClassifier(
    **fixed_params,
    tree_method="hist",
    predictor="cpu_predictor",
    n_jobs=-1,
    random_state=42,
    eval_metric="auc",
)
t0 = time.perf_counter()
best.fit(X_train, y_train)
train_secs = time.perf_counter() - t0

prob = best.predict_proba(X_test)[:, 1]
auc  = roc_auc_score(y_test, prob)
pra  = average_precision_score(y_test, prob)
print(f"\nRaw model  AUC {auc:.4f} | PR-AUC {pra:.4f}  (train {train_secs:.1f}s)")

# ── Manual Platt scaling ──────────────────────────────────
eps   = 1e-6
logit = np.log(prob.clip(eps, 1 - eps) / (1 - prob.clip(eps, 1 - eps))).reshape(-1, 1)

platt = LogisticRegression(solver="lbfgs")
platt.fit(logit, y_test)

prob_cal = platt.predict_proba(logit)[:, 1]
auc_cal  = roc_auc_score(y_test, prob_cal)
pra_cal  = average_precision_score(y_test, prob_cal)
print(f"Platt-calib AUC {auc_cal:.4f} | PR-AUC {pra_cal:.4f}")

# ── Log to MLflow ─────────────────────────────────────────
mlflow.set_experiment("numeric_tuning")
with mlflow.start_run(run_name=f"optuna_{N_TRIALS}_trials"):
    # Log both Optuna and XGB params (after mapping)
    mlflow.log_params(best_params)
    mlflow.log_params({"learning_rate": fixed_params.get("learning_rate"),
                       "colsample_bytree": fixed_params.get("colsample_bytree")})
    mlflow.log_metric("auc",        auc)
    mlflow.log_metric("pr_auc",     pra)
    mlflow.log_metric("auc_cal",    auc_cal)
    mlflow.log_metric("pr_auc_cal", pra_cal)
    mlflow.log_metric("study_best_value_auc", study.best_value)

    # minimal example + signature for the Platt layer
    import mlflow.sklearn
    from mlflow.models.signature import infer_signature
    ex = logit[:1]
    sig = infer_signature(ex, platt.predict_proba(ex))
    mlflow.sklearn.log_model(platt, "platt_layer", input_example=ex, signature=sig)

print("Logged to MLflow: experiment=numeric_tuning")


























