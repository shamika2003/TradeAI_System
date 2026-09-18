import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
SYSTEM_ROOT = PROJECT_DIR.parent
ARTIFACTS_DIR = SYSTEM_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
DATASET_DIR = ARTIFACTS_DIR / "datasets"
REPORT_DIR = ARTIFACTS_DIR / "reports"
for _p in (MODEL_DIR, DATASET_DIR, REPORT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

DATA_PATH = DATASET_DIR / "market_dataset.csv"
DATASET_METADATA_PATH = DATASET_DIR / "market_dataset.metadata.json"
MODEL_PATH = MODEL_DIR / "trading_model.pkl"

SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCNH"]
TIMEFRAME_NAME = "M5"
TRAINING_CUTOFF_DATE = os.environ.get("TRADEAI_TRAINING_CUTOFF_DATE", "").strip() or None

# Runtime fallbacks only. Production requires a calibrated structural-meta policy.
MIN_CONFIDENCE = 0.55
SIGNAL_THRESHOLD = 0.10
HOLD_MARGIN = 0.0

# Directional opportunity ensemble: each symbol owns two binary TP-hit models
# and two expected-net-R regressors. Separate direction models avoid forcing a
# BUY/HOLD/SELL competition when the real question is whether either side has
# positive executable expectancy.
CLASSIFIER_PARAMS = dict(
    n_estimators=700,
    learning_rate=0.018,
    max_depth=4,
    min_child_weight=10,
    subsample=0.82,
    colsample_bytree=0.82,
    reg_alpha=0.25,
    reg_lambda=2.5,
    gamma=0.05,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)
REGRESSOR_PARAMS = dict(
    n_estimators=650,
    learning_rate=0.018,
    max_depth=4,
    min_child_weight=12,
    subsample=0.82,
    colsample_bytree=0.82,
    reg_alpha=0.30,
    reg_lambda=3.0,
    gamma=0.05,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=43,
    n_jobs=-1,
)
MODEL_PARAMS = {
    "architecture": "structural_meta_probability_plus_expected_r",
    "classifier": CLASSIFIER_PARAMS,
    "regressor": REGRESSOR_PARAMS,
}

# Recent observations matter more but old regimes are retained. Half-life is in
# M5 rows for a single symbol; ~55k bars is roughly one trading year.
RECENCY_HALF_LIFE_ROWS = 55_000
RECENCY_MIN_WEIGHT = 0.20
CLASS_BALANCE_MAX_MULTIPLIER = 4.0

# Stage 11 uses wider, separated chronological zones.  Stage 10's policy
# selection window was too short for a deliberately sparse strategy: a handful
# of zero-trade blocks could disable a genuinely selective symbol.  We now give
# policy selection roughly four times more history while keeping the final
# pre-June slice completely untouched.
QUALIFICATION_TRAIN_FRACTION = 0.64
QUALIFICATION_SCORE_CAL_END_FRACTION = 0.70
POLICY_SELECTION_END_FRACTION = 0.90
DEPLOYMENT_CALIBRATION_END_FRACTION = 0.95
# Backward-compatible aliases used by older imports/tests.
CALIBRATION_TRAIN_FRACTION = QUALIFICATION_TRAIN_FRACTION
CALIBRATION_END_FRACTION = POLICY_SELECTION_END_FRACTION

# Structural meta-label training is intentionally broad.  The model learns only
# from bars that already resemble a plausible trend/breakout/pullback setup,
# rather than trying to discover direction from every random M5 candle.
META_TRAIN_SETUP_MIN_SCORE = 0.40
META_TRAIN_SETUP_MIN_GAP = 0.00
META_MIN_TRAIN_ROWS = 2500

# Quality-first policy grids.  Setup thresholds are calibrated jointly with ML
# probability/EV thresholds, but coverage remains tightly capped.
OPPORTUNITY_SETUP_SCORE_GRID = [0.50, 0.58, 0.66]
OPPORTUNITY_SETUP_GAP_GRID = [0.03, 0.08]
OPPORTUNITY_DIRECTION_MODES = ["BOTH", "BUY", "SELL"]
OPPORTUNITY_PROBABILITY_GRID = [0.48, 0.54, 0.60, 0.66]
OPPORTUNITY_EXPECTED_R_GRID = [0.05, 0.15, 0.30, 0.45]
OPPORTUNITY_EV_GAP_GRID = [0.03, 0.08, 0.15]
OPPORTUNITY_PROB_GAP_GRID = [0.00, 0.04]
CALIBRATION_MAX_COVERAGE = 0.025

# Per-symbol money gate.  A symbol can remain silent for days; evidence is
# accumulated over a much longer selection window and only active blocks count
# toward sparse-signal stability.
POLICY_CALIBRATION_BALANCE = 150.0
POLICY_CALIBRATION_RISK_PERCENT = 0.45
POLICY_CALIBRATION_MIN_TRADES = 12
# Broker minimum-lot rejection must not force the selector toward larger risk.
# Independent episode evidence remains >=12 setups, while money validation only requires a
# smaller number of actually executable trades at the $150 calibration balance.
POLICY_CALIBRATION_MIN_EXECUTED_TRADES = 6
POLICY_CALIBRATION_NORMAL_MIN_PROFIT_FACTOR = 1.30
POLICY_CALIBRATION_STRESS_MIN_PROFIT_FACTOR = 1.10
POLICY_CALIBRATION_MIN_PAYOFF_RATIO = 1.25
POLICY_CALIBRATION_BLOCKS = 10
POLICY_CALIBRATION_MIN_ACTIVE_BLOCKS = 4
POLICY_CALIBRATION_MIN_TRADES_PER_ACTIVE_BLOCK = 1
POLICY_CALIBRATION_MIN_POSITIVE_BLOCK_FRACTION = 0.60
POLICY_CALIBRATION_NORMAL_MAX_DRAWDOWN_PERCENT = 8.0
POLICY_CALIBRATION_STRESS_MAX_DRAWDOWN_PERCENT = 10.0

VALIDATION_BALANCES = [50.0, 100.0, 150.0]
PRIMARY_ACCEPTANCE_BALANCES = [150.0]
FINAL_BACKTEST_BALANCE = float(os.environ.get("TRADEAI_FINAL_BACKTEST_BALANCE", "150"))
FINAL_BACKTEST_START_DATE = os.environ.get("TRADEAI_FINAL_BACKTEST_START_DATE", "2026-06-01").strip()
FINAL_BACKTEST_END_DATE = os.environ.get("TRADEAI_FINAL_BACKTEST_END_DATE", "2026-06-30").strip()
REQUIRE_BROKER_PROFILE_FOR_FINAL_VALIDATION = True

# Portfolio acceptance is quality-first. We still require enough trades to
# avoid promoting a result based on one lucky position, but no gate pressures
# the model toward dozens of low-quality trades per symbol.
ACCEPT_NORMAL_MIN_PROFIT_FACTOR = 1.25
ACCEPT_NORMAL_MAX_DRAWDOWN_PERCENT = 10.0
ACCEPT_NORMAL_MIN_TRADES = 6
ACCEPT_STRESS_MIN_PROFIT_FACTOR = 1.05
ACCEPT_STRESS_MAX_DRAWDOWN_PERCENT = 12.0
ACCEPT_STRESS_MIN_TRADES = 5

STAGE5_RISK_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
STAGE5_CAL_NORMAL_MAX_DRAWDOWN_PERCENT = 7.0
STAGE5_CAL_STRESS_MAX_DRAWDOWN_PERCENT = 9.0
STAGE5_CAL_NORMAL_MIN_PROFIT_FACTOR = ACCEPT_NORMAL_MIN_PROFIT_FACTOR
STAGE5_CAL_STRESS_MIN_PROFIT_FACTOR = ACCEPT_STRESS_MIN_PROFIT_FACTOR
STAGE5_CAL_MIN_TRADES = 0

# Legacy Stage-4 import aliases. The final pipeline no longer uses the old
# multiclass threshold search, but keeping these names lets the shared execution
# simulator remain import-compatible with older diagnostics/tests.
CALIBRATION_CONFIDENCE_GRID = [0.55]
CALIBRATION_EDGE_GRID = [0.10]
CALIBRATION_HOLD_MARGIN_GRID = [0.0]
CALIBRATION_MIN_TRADES = 30
CALIBRATION_MIN_PROFIT_FACTOR = 1.0
