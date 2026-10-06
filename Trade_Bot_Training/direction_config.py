from __future__ import annotations

import os
from pathlib import Path


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = SYSTEM_ROOT / "artifacts"

RAW_DATA_DIR = ARTIFACTS_DIR / "datasets" / "direction_m5_raw"
LEGACY_RAW_DATA_DIR = ARTIFACTS_DIR / "datasets" / "direction_m5_v1_raw"
PROCESSED_DATA_DIR = ARTIFACTS_DIR / "datasets" / "direction_m5"
MODEL_ROOT = ARTIFACTS_DIR / "models" / "direction_m5"
REPORT_ROOT = ARTIFACTS_DIR / "reports" / "direction_m5"
DATASET_METADATA_PATH = PROCESSED_DATA_DIR / "dataset_metadata.json"

MODEL_VERSION = "tradeai_direction_m5"
DATASET_VERSION = "tradeai_direction_dataset"

PRIMARY_TIMEFRAME = "M5"
CONTEXT_TIMEFRAME = "H1"
M5_MINUTES = 5

HORIZON_BARS = 6
HORIZON_MINUTES = HORIZON_BARS * M5_MINUTES

PREDICTION_CONTRACT = (
    "After a fully CLOSED M5 candle, forecast the executable market move "
    "from the immediately following contiguous M5 OPEN to the CLOSE "
    "30 minutes later."
)

TARGET_UP = 1
TARGET_DOWN = 0

SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCNH",
]

HISTORY_START_UTC = "2020-01-01"
M5_CHUNK_DAYS = 90
H1_CHUNK_DAYS = 365
MIN_RAW_M5_ROWS = 50_000
MIN_RAW_H1_ROWS = 4_000
MAX_FEATURE_LOOKBACK = 96

# Strict chronological production split.
#
# TRAIN       : fit model parameters
# VALIDATION  : early stopping only
# CALIBRATION : probability calibration only
# META        : train trade-selection meta model later
# TEST        : final untouched economic evaluation
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.15
CALIBRATION_FRACTION = 0.10
META_FRACTION = 0.05
TEST_FRACTION = 0.10
PURGE_MINUTES = HORIZON_MINUTES

RANDOM_STATE = 42
CPU_COUNT = os.cpu_count() or 4
N_JOBS = max(1, CPU_COUNT - 1)
XGB_DEVICE = os.environ.get(
    "TRADEAI_XGB_DEVICE",
    "cpu",
).strip().lower()
EARLY_STOPPING_ROUNDS = 120

# Fixed diverse ensemble. No grid search against holdout data.
CLASSIFIER_ENSEMBLE = [
    {
        "name": "shallow",
        "n_estimators": 2200,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 10,
        "subsample": 0.90,
        "colsample_bytree": 0.88,
        "reg_alpha": 0.20,
        "reg_lambda": 9.0,
        "gamma": 0.01,
        "random_state_offset": 0,
    },
    {
        "name": "balanced",
        "n_estimators": 2200,
        "learning_rate": 0.03,
        "max_depth": 5,
        "min_child_weight": 14,
        "subsample": 0.88,
        "colsample_bytree": 0.84,
        "reg_alpha": 0.30,
        "reg_lambda": 12.0,
        "gamma": 0.015,
        "random_state_offset": 17,
    },
    {
        "name": "selective",
        "n_estimators": 2200,
        "learning_rate": 0.025,
        "max_depth": 6,
        "min_child_weight": 22,
        "subsample": 0.86,
        "colsample_bytree": 0.80,
        "reg_alpha": 0.40,
        "reg_lambda": 16.0,
        "gamma": 0.02,
        "random_state_offset": 31,
    },
]

REGRESSOR_PARAMS = {
    "n_estimators": 2200,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 18,
    "subsample": 0.88,
    "colsample_bytree": 0.84,
    "reg_alpha": 0.25,
    "reg_lambda": 12.0,
    "gamma": 0.0,
}

REGRESSION_TARGET_CLIP_ATR = 6.0

# Diagnostics only. Never used as production trade thresholds.
CONFIDENCE_LEVELS = [
    0.50,
    0.52,
    0.55,
    0.58,
    0.60,
    0.65,
    0.70,
    0.75,
]


def ensure_directories() -> None:
    for path in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODEL_ROOT,
        REPORT_ROOT,
    ):
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def raw_path(
    symbol: str,
    timeframe: str,
) -> Path:
    symbol = str(symbol).upper()
    timeframe = str(timeframe).upper()
    filename = f"{symbol}_{timeframe}.csv.gz"

    canonical = RAW_DATA_DIR / filename
    if canonical.exists():
        return canonical

    legacy = LEGACY_RAW_DATA_DIR / filename
    if legacy.exists():
        return legacy

    return canonical


def processed_path(
    symbol: str,
) -> Path:
    return (
        PROCESSED_DATA_DIR
        / f"{str(symbol).upper()}.csv.gz"
    )


def symbol_model_dir(
    symbol: str,
) -> Path:
    return (
        MODEL_ROOT
        / str(symbol).upper()
    )


def symbol_report_dir(
    symbol: str,
) -> Path:
    return (
        REPORT_ROOT
        / str(symbol).upper()
    )
