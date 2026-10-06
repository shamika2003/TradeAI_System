from __future__ import annotations
from pathlib import Path
from Trade_Bot_Training.direction_config import ARTIFACTS_DIR

BROKER_PROFILE_PATH = ARTIFACTS_DIR / "reports" / "broker_profile.json"
POLICY_ROOT = ARTIFACTS_DIR / "strategy" / "signal_policy"
SIGNAL_REPORT_ROOT = ARTIFACTS_DIR / "reports" / "signal_policy"
POLICY_VERSION = "tradeai_signal_policy"

META_SEARCH_FRACTION = 0.60

# Candidate trade gates are learned from META quantiles, not fixed probability numbers.
CONFIDENCE_QUANTILES = [
    0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
    0.80, 0.84, 0.88, 0.91, 0.94, 0.96,
    0.975, 0.985, 0.992, 0.996,
]
EDGE_QUANTILES = [0.20, 0.30, 0.40, 0.50, 0.60, 0.68, 0.75, 0.82, 0.88, 0.92, 0.95]

# Evidence requirements, not individual-trade thresholds.
MIN_SEARCH_TRADES = 30
MIN_CONFIRM_TRADES = 25
MIN_PROFIT_FACTOR = 1.0

SPREAD_FALLBACK_LOOKBACK_DAYS = 90
SPREAD_FALLBACK_MIN_ROWS = 500
SPREAD_REGIME_MAX_RATIO = 8.0
SPREAD_REGIME_MIN_RATIO = 1.0 / SPREAD_REGIME_MAX_RATIO

def ensure_strategy_directories() -> None:
    POLICY_ROOT.mkdir(parents=True, exist_ok=True)
    SIGNAL_REPORT_ROOT.mkdir(parents=True, exist_ok=True)

def symbol_policy_dir(symbol: str) -> Path:
    return POLICY_ROOT / str(symbol).upper()

def symbol_signal_report_dir(symbol: str) -> Path:
    return SIGNAL_REPORT_ROOT / str(symbol).upper()
