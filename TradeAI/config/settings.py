# filename: TradeAI/config/settings.py

import os
from pathlib import Path


# =====================================================
# PROJECT PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent.parent          # TradeAI/
SYSTEM_ROOT = BASE_DIR.parent                             # TradeAI_System/
ARTIFACTS_DIR = SYSTEM_ROOT / "artifacts"

MODEL_DIR = ARTIFACTS_DIR / "models"
DATASET_DIR = ARTIFACTS_DIR / "datasets"
REPORT_DIR = ARTIFACTS_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# MODE
# =====================================================

# Keep BACKTEST during the rebuild/validation stages.
# Change to LIVE only after the new contract model passes the later gates.
MODE = "DEMO_FORWARD"
MODE = os.environ.get("TRADEAI_MODE_OVERRIDE", MODE).strip().upper()

USE_MT5 = MODE in ("LIVE", "DEMO_FORWARD")
USE_PAPER = MODE == "PAPER"
USE_BACKTEST = MODE == "BACKTEST"
USE_OFFLINE = USE_PAPER or USE_BACKTEST


# =====================================================
# DEMO FORWARD GATE
# =====================================================

# DEMO_FORWARD uses live MT5 market data and broker execution,
# but it is hard-blocked from running on a real-money account.
DEMO_FORWARD = MODE == "DEMO_FORWARD"
DEMO_FORWARD_REQUIRE_DEMO_ACCOUNT = True
DEMO_FORWARD_CAPITAL = 50.0
DEMO_FORWARD_MIN_TRADES = 100
DEMO_FORWARD_MAX_DD_PERCENT = 12.0
DEMO_FORWARD_MIN_PROFIT_FACTOR = 1.20
DEMO_FORWARD_MIN_RETURN_PERCENT = 0.0
DEMO_FORWARD_STATE_PATH = REPORT_DIR / "demo_forward_state.json"



# =====================================================
# MODEL / DATA — ONE CANONICAL LOCATION
# =====================================================

MODEL_PATH = MODEL_DIR / "trading_model.pkl"
DATA_PATH = DATASET_DIR / "market_dataset.csv"


# =====================================================
# SYMBOLS
# =====================================================

SYMBOLS = [
    "EURUSD",
    "GBPUSD",
    "USDJPY",
    "USDCNH",
]

_symbol_override = os.environ.get("TRADEAI_SYMBOLS_OVERRIDE", "").strip()
if _symbol_override:
    parsed_symbols = [part.strip().upper() for part in _symbol_override.split(",") if part.strip()]
    if parsed_symbols:
        SYMBOLS = parsed_symbols


# =====================================================
# TIMEFRAME
# =====================================================

TIMEFRAME = 5
HISTORY_SIZE = 2000


# =====================================================
# LOOP
# =====================================================

LIVE_INTERVAL = 5
BACKTEST_DELAY = 0


# =====================================================
# AI / SIGNAL
# =====================================================

# Stage 3 multiclass probability-edge gates.
# These are provisional; Stage 4 calibrates them on out-of-sample money P/L.
SIGNAL_THRESHOLD = 0.14
MIN_CONFIDENCE = 0.55
HOLD_MARGIN = 0.05

# Stage 4: production startup requires a calibrated per-symbol decision policy
# embedded in the model artifact. The validator writes it only after acceptance.
REQUIRE_CALIBRATED_POLICY = True
REQUIRE_CALIBRATED_RISK_POLICY = True


# =====================================================
# CAPITAL
# =====================================================

START_BALANCE = 10.0
try:
    START_BALANCE = float(os.environ.get("TRADEAI_START_BALANCE", START_BALANCE))
except (TypeError, ValueError):
    pass
DEFAULT_CAPITAL = START_BALANCE


# =====================================================
# LOW-BALANCE ACCOUNT PROFILE
# =====================================================

LOW_BALANCE_MODE = True
LOW_BALANCE_MIN = 10.0
LOW_BALANCE_MAX = 50.0


# =====================================================
# RISK
# =====================================================

RISK_PERCENT = 1.0
MAX_ACTUAL_RISK_PERCENT = 1.25


# =====================================================
# POSITION LIMITS
# =====================================================

MAX_OPEN_POSITIONS = 2
MAX_TOTAL_POSITIONS = 2

# Cap aggregate initial loss-at-stop across all open positions. With the
# calibrated per-trade grid below, two simultaneous trades can coexist without
# silently multiplying account risk.
MAX_PORTFOLIO_RISK_PERCENT = 0.80

MAX_OPEN_TRADES = MAX_OPEN_POSITIONS
MAX_TOTAL_TRADES = MAX_TOTAL_POSITIONS


# =====================================================
# COOLDOWN
# =====================================================

COOLDOWN_SECONDS = 300


# =====================================================
# LOT LIMITS
# =====================================================

# PAPER/BACKTEST defaults. LIVE reads the broker's actual lot rules.
MIN_LOT = 0.001
MAX_LOT = 1.0
LOT_STEP = 0.001
DEFAULT_LOT = 0.001
TRADE_LOT = DEFAULT_LOT


# =====================================================
# ATR STOP MANAGEMENT
# =====================================================

USE_ATR_STOPS = True
ATR_SL_MULTIPLIER = 0.75
ATR_TP_MULTIPLIER = 1.875


# =====================================================
# TRADE MANAGEMENT
# =====================================================

# Management is expressed in R (initial stop distance), not fixed pips.
# This keeps behaviour comparable across EURUSD, GBPUSD, USDJPY and USDCNH.
USE_BREAK_EVEN = True
BREAK_EVEN_TRIGGER_R = 1.20
BREAK_EVEN_BUFFER_PIPS = 0.20
# A break-even exit should not round back to $0.00 after commission.
BREAK_EVEN_MIN_PROFIT_MONEY = 0.02

# Once a trade proves itself, lock a meaningful fraction of the initial risk
# before the late runner trail starts. This is intentionally later than old
# Stage-3 management so winners are not clipped around 1R.
USE_PROFIT_LOCK = True
PROFIT_LOCK_TRIGGER_R = 1.75
PROFIT_LOCK_R = 0.55

USE_TRAILING_STOP = False
TRAILING_TRIGGER_R = 2.40
TRAILING_ATR_MULTIPLIER = 0.90

# Defensive AI exit: only cut a trade early when it is already meaningfully
# adverse AND the model produces a calibrated, confident signal in the
# opposite direction. Stage 5 will recalibrate these thresholds after the
# execution/backtest rebuild is complete.
USE_AI_DEFENSIVE_EXIT = True
AI_DEFENSIVE_EXIT_MIN_BARS = 2
AI_DEFENSIVE_EXIT_ADVERSE_R = 0.12
AI_DEFENSIVE_EXIT_SIGNAL_MULTIPLIER = 1.00

# Opportunity-thesis management. These rules require model evidence before an
# early exit; they do not cut ordinary candle noise.
USE_OPPORTUNITY_THESIS_EXIT = True
THESIS_EXIT_MIN_BARS = 3
THESIS_EXIT_ADVERSE_R = 0.12
THESIS_STALE_BARS = 18
THESIS_STALE_MAX_R = 0.20
THESIS_OWN_PROBABILITY_FRACTION = 0.70
THESIS_OWN_EV_FLOOR = -0.05
THESIS_OPPOSITE_EV_MARGIN = 0.15
THESIS_OPPOSITE_PROBABILITY_MARGIN = 0.05
# Structural setup must fall materially below its entry gate before it can
# contribute to an early-exit decision. It never exits a healthy trade alone.
THESIS_SETUP_FRACTION = 0.70

# Legacy names retained for compatibility with external/UI code. They are no
# longer used by TradeManager for production management decisions.
BREAK_EVEN_TRIGGER_PIPS = 10.0
TRAILING_TRIGGER_PIPS = 20.0

# Align production trade lifecycle with the 48-bar quality/payoff horizon.
USE_MAX_HOLD = True
MAX_HOLD_BARS = 48
MAX_HOLD_MINUTES = 240


# =====================================================
# ACCOUNT PROTECTION
# =====================================================

MAX_DRAWDOWN_PERCENT = 12.0
MAX_DAILY_LOSS_PERCENT = 5.0


# =====================================================
# MT5
# =====================================================

MT5_MAGIC = 123456
DEVIATION = 20


# =====================================================
# BACKTEST DATE RANGE
# =====================================================

BACKTEST_START_DATE = "2026-06-01"
BACKTEST_END_DATE = "2026-06-30"
BACKTEST_START_DATE = os.environ.get("TRADEAI_BACKTEST_START_DATE", BACKTEST_START_DATE).strip()
BACKTEST_END_DATE = os.environ.get("TRADEAI_BACKTEST_END_DATE", BACKTEST_END_DATE).strip()


# =====================================================
# EXECUTION MODEL / BROKER PARITY
# =====================================================

# MT5 copy_rates* OHLC bars are treated as BID-side prices.
# PAPER/BACKTEST therefore executes BUY entries on ASK (BID + spread)
# and SELL entries on BID, matching MT5 FX execution semantics.
HISTORICAL_OHLC_SIDE = "BID"

# When a DEMO_FORWARD/LIVE MT5 executor starts, it snapshots the broker's
# symbol rules here. PAPER/BACKTEST consumes the same profile when present.
BROKER_PROFILE_PATH = REPORT_DIR / "broker_profile.json"
USE_BROKER_PROFILE = True

# Fallbacks are used only when no captured broker value is available.
DEFAULT_SPREAD_PIPS = 1.2
SPREAD_PIPS = DEFAULT_SPREAD_PIPS
SIMULATED_SLIPPAGE_PIPS = 0.2
COMMISSION_PER_LOT = 7.0

# Stage 4 stress scenario used by production validation.
STRESS_SPREAD_PIPS = 1.8
STRESS_SLIPPAGE_PIPS = 0.5
STRESS_COMMISSION_PER_LOT = 9.0

USE_SPREAD_EXECUTION = True


# =====================================================
# LOGGING
# =====================================================

# Fixed: BASE_DIR already points at TradeAI, so do not append TradeAI again.
LOG_FOLDER = BASE_DIR / "logs"
LOG_FOLDER.mkdir(parents=True, exist_ok=True)

TRADE_LOG = LOG_FOLDER / "trades.csv"
BOT_LOG = LOG_FOLDER / "bot.log"
