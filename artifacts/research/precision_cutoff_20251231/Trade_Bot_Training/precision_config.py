from __future__ import annotations

import os
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = SYSTEM_ROOT / "artifacts"

RAW_DATA_DIR = ARTIFACTS_DIR / "datasets" / "direction_m5_raw"
DATASET_DIR = ARTIFACTS_DIR / "datasets" / "precision_m5"
MODEL_ROOT = ARTIFACTS_DIR / "models" / "precision_m5"
REPORT_ROOT = ARTIFACTS_DIR / "reports" / "precision_m5"

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCNH"]

M5_MINUTES = 5
HORIZON_BARS = 6
HORIZON_MINUTES = 30

# The classifier is deliberately three-class.  Tiny 30-minute moves are not
# forced into UP/DOWN labels; they become HOLD and the model learns abstention.
MIN_MEANINGFUL_MOVE_ATR = 0.60

CLASS_DOWN = 0
CLASS_HOLD = 1
CLASS_UP = 2
CLASS_NAMES = {0: "DOWN", 1: "HOLD", 2: "UP"}

# This is a fixed architecture-level action gate, not selected on the final test.
ACTION_CONFIDENCE = 0.55

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
PURGE_BARS = HORIZON_BARS

RANDOM_STATE = 42
N_JOBS = max(1, min(4, (os.cpu_count() or 4) - 1))

MODEL_PARAMS = {
    "n_estimators": 1400,
    "learning_rate": 0.035,
    "max_depth": 5,
    "min_child_weight": 20,
    "subsample": 0.88,
    "colsample_bytree": 0.82,
    "reg_alpha": 0.20,
    "reg_lambda": 12.0,
    "gamma": 0.02,
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_bin": 128,
    "random_state": RANDOM_STATE,
    "n_jobs": N_JOBS,
}

EARLY_STOPPING_ROUNDS = 80

# A model is never labelled production-ready merely because training completed.
# These gates are evaluated only on the final chronological holdout.
PRODUCTION_MIN_DIRECTIONAL_ACCURACY = 0.78
PRODUCTION_MIN_ACTION_ROWS = 250
PRODUCTION_MIN_MEAN_SIGNED_MOVE_ATR = 0.50
PRODUCTION_MIN_SIDE_ACCURACY = 0.70
PRODUCTION_MIN_SIDE_ROWS = 50
PRODUCTION_MIN_GOOD_MONTH_FRACTION = 0.70
PRODUCTION_MONTH_MIN_ROWS = 10
PRODUCTION_MONTH_MIN_ACCURACY = 0.65


def raw_path(symbol: str) -> Path:
    return RAW_DATA_DIR / f"{symbol.upper()}_M5.csv.gz"


def dataset_path(symbol: str) -> Path:
    return DATASET_DIR / f"{symbol.upper()}.csv.gz"


def model_dir(symbol: str) -> Path:
    return MODEL_ROOT / symbol.upper()


def report_dir(symbol: str) -> Path:
    return REPORT_ROOT / symbol.upper()


def ensure_directories() -> None:
    for p in (DATASET_DIR, MODEL_ROOT, REPORT_ROOT):
        p.mkdir(parents=True, exist_ok=True)
