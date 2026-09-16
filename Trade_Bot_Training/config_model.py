# filename: Trade_Bot_Traning/config_model.py

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = PROJECT_DIR.parent

ARTIFACTS_DIR = SYSTEM_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
DATASET_DIR = ARTIFACTS_DIR / "datasets"
REPORT_DIR = ARTIFACTS_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATASET_DIR / "market_dataset.csv"
DATASET_METADATA_PATH = DATASET_DIR / "market_dataset.metadata.json"
MODEL_PATH = MODEL_DIR / "trading_model.pkl"

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCNH"]
TIMEFRAME_NAME = "M5"

# Provisional production gates. Stage 4 calibrates these from out-of-sample P/L.
SIGNAL_THRESHOLD = 0.10
MIN_CONFIDENCE = 0.45

# One canonical multiclass XGBoost specification for trainer + holdout +
# feature-importance analysis.
MODEL_PARAMS = dict(
    n_estimators=550,
    learning_rate=0.025,
    max_depth=5,
    subsample=0.80,
    colsample_bytree=0.80,
    min_child_weight=4,
    reg_alpha=0.15,
    reg_lambda=1.25,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3,
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)


# =====================================================
# STAGE 4 PRODUCTION VALIDATION
# =====================================================

CALIBRATION_TRAIN_FRACTION = 0.70
CALIBRATION_END_FRACTION = 0.85

CALIBRATION_CONFIDENCE_GRID = [0.42, 0.46, 0.50, 0.54, 0.58, 0.62, 0.66]
CALIBRATION_EDGE_GRID = [0.06, 0.10, 0.14, 0.18, 0.22]

CALIBRATION_MIN_TRADES = 300
CALIBRATION_MIN_PROFIT_FACTOR = 1.05
CALIBRATION_MAX_COVERAGE = 0.25

VALIDATION_BALANCES = [10.0, 20.0, 50.0, 100.0]
PRIMARY_ACCEPTANCE_BALANCES = [20.0, 50.0]

ACCEPT_NORMAL_MIN_PROFIT_FACTOR = 1.15
ACCEPT_NORMAL_MAX_DRAWDOWN_PERCENT = 12.0
ACCEPT_NORMAL_MIN_TRADES = 100

ACCEPT_STRESS_MIN_PROFIT_FACTOR = 1.00
ACCEPT_STRESS_MAX_DRAWDOWN_PERCENT = 15.0
ACCEPT_STRESS_MIN_TRADES = 75

# =====================================================
# STAGE 5 RISK CALIBRATION
# =====================================================

# Risk is selected ONLY from the Stage 5 calibration slice. The final test
# cannot influence this choice.
STAGE5_RISK_GRID = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75, 1.00]

# Calibration uses stricter drawdown limits than the final acceptance gates.
# This creates a margin for regime deterioration before promotion.
STAGE5_CAL_NORMAL_MAX_DRAWDOWN_PERCENT = 8.0
STAGE5_CAL_STRESS_MAX_DRAWDOWN_PERCENT = 10.0
STAGE5_CAL_NORMAL_MIN_PROFIT_FACTOR = ACCEPT_NORMAL_MIN_PROFIT_FACTOR
STAGE5_CAL_STRESS_MIN_PROFIT_FACTOR = ACCEPT_STRESS_MIN_PROFIT_FACTOR
STAGE5_CAL_MIN_TRADES = 100
