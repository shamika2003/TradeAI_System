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
SIGNAL_THRESHOLD = 0.10
MIN_CONFIDENCE = 0.45

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

MAX_OPEN_POSITIONS = 1
MAX_TOTAL_POSITIONS = 1

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
ATR_SL_MULTIPLIER = 1.5
ATR_TP_MULTIPLIER = 3.0


# =====================================================
# TRADE MANAGEMENT
# =====================================================

USE_BREAK_EVEN = True
BREAK_EVEN_TRIGGER_PIPS = 10.0

USE_TRAILING_STOP = True
TRAILING_TRIGGER_PIPS = 20.0

# Align production trade lifecycle with the Stage 3 barrier target.
USE_MAX_HOLD = True
MAX_HOLD_BARS = 24
MAX_HOLD_MINUTES = 120


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
# EXECUTION MODEL
# =====================================================

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
