# filename: shared/tradeai_core/target_definition.py

from __future__ import annotations

import numpy as np
import pandas as pd


# Stage 7: payoff-first, execution-aware target contract.
TARGET_VERSION = "tradeai_payoff_target_v4_20260917"
PREDICTION_TYPE = "barrier_multiclass"

# Runtime and labels intentionally share the same geometry.
# 1.25 ATR stop / 3.75 ATR target = 3.0R gross reward:risk.
STOP_ATR_MULTIPLIER = 1.25
TAKE_ATR_MULTIPLIER = 3.75
MAX_HOLD_BARS = 36          # 3 hours on M5
MAX_TARGET_HORIZON_BARS = MAX_HOLD_BARS
PRIMARY_HORIZON_BARS = MAX_HOLD_BARS
LONG_HORIZON_BARS = MAX_HOLD_BARS

SELL_CLASS = 0
HOLD_CLASS = 1
BUY_CLASS = 2
CLASS_MAP = {
    "SELL": SELL_CLASS,
    "HOLD": HOLD_CLASS,
    "BUY": BUY_CLASS,
}

# Directional labels now require a more meaningful net edge. Tiny moves are
# intentionally HOLD so the classifier spends capacity on trades with room to
# pay spread/commission and still deliver asymmetric payoff.
MIN_EDGE_R = 0.25

# Label economics mirror the production/BACKTEST execution convention:
# MT5 OHLC is BID-side; BUY enters on ASK, SELL enters on BID; SELL exits are
# triggered from ASK. Commission is the current broker profile's round-trip
# model used by PaperExecutor/validation ($7 per standard lot).
LABEL_SLIPPAGE_PIPS = 0.20
LABEL_COMMISSION_PER_LOT = 7.0
LABEL_OHLC_SIDE = "BID"

_SYMBOL_POINT_SIZE = {
    "EURUSD": 0.00001,
    "GBPUSD": 0.00001,
    "USDJPY": 0.001,
    "USDCNH": 0.00001,
}
_SYMBOL_PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "USDCNH": 0.0001,
}
CONTRACT_SIZE = 100_000.0


def target_contract() -> dict:
    return {
        "target_version": TARGET_VERSION,
        "prediction_type": PREDICTION_TYPE,
        "stop_atr_multiplier": STOP_ATR_MULTIPLIER,
        "take_atr_multiplier": TAKE_ATR_MULTIPLIER,
        "gross_reward_to_risk": TAKE_ATR_MULTIPLIER / STOP_ATR_MULTIPLIER,
        "max_hold_bars": MAX_HOLD_BARS,
        "min_edge_r": MIN_EDGE_R,
        "label_slippage_pips": LABEL_SLIPPAGE_PIPS,
        "label_commission_per_lot": LABEL_COMMISSION_PER_LOT,
        "historical_ohlc_side": LABEL_OHLC_SIDE,
        "class_map": dict(CLASS_MAP),
    }


def _resolve_symbol(df: pd.DataFrame, symbol: str | None) -> str:
    if symbol:
        return str(symbol).upper()

    if "symbol" in df.columns:
        values = [str(x).upper() for x in df["symbol"].dropna().unique()]
        if len(values) == 1:
            return values[0]

    raise ValueError("add_training_targets requires one symbol at a time")


def _pip_value_per_lot(symbol: str, prices: np.ndarray) -> np.ndarray:
    """Approximate USD-account pip value per standard lot, vectorized.

    This mirrors production_economics.fallback_pip_value_per_lot and is used
    only to convert the configured commission into R units during relabelling.
    """
    symbol = str(symbol).upper()
    pip = _SYMBOL_PIP_SIZE.get(symbol, 0.0001)
    quote_value = CONTRACT_SIZE * pip
    out = np.full(len(prices), quote_value, dtype=np.float64)

    if symbol.endswith("USD"):
        return out

    if symbol.startswith("USD"):
        valid = np.isfinite(prices) & (prices > 0)
        out[valid] = quote_value / prices[valid]
        return out

    return out


def _label_one_symbol(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    result = df.copy().reset_index(drop=True)
    n = len(result)

    if n == 0:
        result["target_class"] = np.nan
        result["target_buy_r"] = np.nan
        result["target_sell_r"] = np.nan
        result["target_best_r"] = np.nan
        return result

    required = ["close", "high", "low", "atr"]
    missing = [c for c in required if c not in result.columns]
    if missing:
        raise ValueError(f"Barrier target missing columns: {missing}")

    close = result["close"].to_numpy(dtype=np.float64)
    high = result["high"].to_numpy(dtype=np.float64)
    low = result["low"].to_numpy(dtype=np.float64)
    atr = result["atr"].to_numpy(dtype=np.float64)

    sl_distance = atr * STOP_ATR_MULTIPLIER
    tp_distance = atr * TAKE_ATR_MULTIPLIER

    valid_base = (
        np.isfinite(close)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(sl_distance)
        & (close > 0)
        & (sl_distance > 0)
        & np.isfinite(tp_distance)
        & (tp_distance > 0)
    )

    point = _SYMBOL_POINT_SIZE.get(symbol, 0.00001)
    pip = _SYMBOL_PIP_SIZE.get(symbol, 0.0001)

    spread_points = (
        result["spread"].to_numpy(dtype=np.float64)
        if "spread" in result.columns
        else np.zeros(n, dtype=np.float64)
    )
    spread_points = np.nan_to_num(spread_points, nan=0.0, posinf=0.0, neginf=0.0)
    spread_points = np.maximum(spread_points, 0.0)
    spread_price = spread_points * point
    slippage_price = LABEL_SLIPPAGE_PIPS * pip

    # Production parity:
    # BUY entry = BID close + spread + adverse slippage.
    # SELL entry = BID close - adverse slippage.
    buy_entry = close + spread_price + slippage_price
    sell_entry = close - slippage_price

    buy_sl = close - sl_distance
    buy_tp = close + tp_distance
    sell_sl = close + sl_distance
    sell_tp = close - tp_distance

    stop_pips = np.divide(
        sl_distance,
        pip,
        out=np.zeros(n, dtype=np.float64),
        where=sl_distance > 0,
    )
    pip_value = _pip_value_per_lot(symbol, close)
    price_risk_per_lot = stop_pips * pip_value
    commission_r = np.divide(
        LABEL_COMMISSION_PER_LOT,
        price_risk_per_lot,
        out=np.zeros(n, dtype=np.float64),
        where=price_risk_per_lot > 0,
    )

    buy_stop_reward = np.divide(
        buy_sl - buy_entry,
        sl_distance,
        out=np.full(n, np.nan, dtype=np.float64),
        where=sl_distance > 0,
    ) - commission_r
    buy_tp_reward = np.divide(
        buy_tp - buy_entry,
        sl_distance,
        out=np.full(n, np.nan, dtype=np.float64),
        where=sl_distance > 0,
    ) - commission_r
    sell_stop_reward = np.divide(
        sell_entry - sell_sl,
        sl_distance,
        out=np.full(n, np.nan, dtype=np.float64),
        where=sl_distance > 0,
    ) - commission_r
    sell_tp_reward = np.divide(
        sell_entry - sell_tp,
        sl_distance,
        out=np.full(n, np.nan, dtype=np.float64),
        where=sl_distance > 0,
    ) - commission_r

    buy_r = np.full(n, np.nan, dtype=np.float64)
    sell_r = np.full(n, np.nan, dtype=np.float64)
    buy_exit_bar = np.full(n, -1, dtype=np.int16)
    sell_exit_bar = np.full(n, -1, dtype=np.int16)

    full_horizon = np.zeros(n, dtype=bool)
    if n > MAX_HOLD_BARS:
        full_horizon[: n - MAX_HOLD_BARS] = True

    eligible = valid_base & full_horizon
    buy_active = eligible.copy()
    sell_active = eligible.copy()

    # Earliest future candle resolves each direction. BUY closes on BID.
    # SELL closes/triggers on ASK = BID + that future candle's spread.
    # Same-candle ambiguity stays conservative: stop wins.
    for step in range(1, MAX_HOLD_BARS + 1):
        count = n - step
        if count <= 0:
            break

        idx = np.arange(count)
        fut_high_bid = high[step:]
        fut_low_bid = low[step:]
        fut_spread = spread_price[step:]
        fut_high_ask = fut_high_bid + fut_spread
        fut_low_ask = fut_low_bid + fut_spread

        active_b = buy_active[:count]
        if active_b.any():
            sl_hit = fut_low_bid <= buy_sl[:count]
            tp_hit = fut_high_bid >= buy_tp[:count]
            resolve_sl = active_b & sl_hit
            resolve_tp = active_b & (~sl_hit) & tp_hit

            buy_r[idx[resolve_sl]] = buy_stop_reward[:count][resolve_sl]
            buy_r[idx[resolve_tp]] = buy_tp_reward[:count][resolve_tp]
            buy_exit_bar[idx[resolve_sl | resolve_tp]] = step
            buy_active[idx[resolve_sl | resolve_tp]] = False

        active_s = sell_active[:count]
        if active_s.any():
            sl_hit = fut_high_ask >= sell_sl[:count]
            tp_hit = fut_low_ask <= sell_tp[:count]
            resolve_sl = active_s & sl_hit
            resolve_tp = active_s & (~sl_hit) & tp_hit

            sell_r[idx[resolve_sl]] = sell_stop_reward[:count][resolve_sl]
            sell_r[idx[resolve_tp]] = sell_tp_reward[:count][resolve_tp]
            sell_exit_bar[idx[resolve_sl | resolve_tp]] = step
            sell_active[idx[resolve_sl | resolve_tp]] = False

    # Timeout at the maximum horizon using production quote sides.
    base_count = max(0, n - MAX_HOLD_BARS)
    if base_count:
        idx = np.arange(base_count)
        terminal_bid = close[MAX_HOLD_BARS:]
        terminal_ask = terminal_bid + spread_price[MAX_HOLD_BARS:]

        buy_timeout = (
            (terminal_bid - buy_entry[:base_count]) / sl_distance[:base_count]
            - commission_r[:base_count]
        )
        sell_timeout = (
            (sell_entry[:base_count] - terminal_ask) / sl_distance[:base_count]
            - commission_r[:base_count]
        )

        # Keep timeout rewards within the same executable barrier envelope.
        buy_timeout = np.minimum(
            np.maximum(buy_timeout, buy_stop_reward[:base_count]),
            buy_tp_reward[:base_count],
        )
        sell_timeout = np.minimum(
            np.maximum(sell_timeout, sell_stop_reward[:base_count]),
            sell_tp_reward[:base_count],
        )

        unresolved_b = buy_active[:base_count]
        unresolved_s = sell_active[:base_count]

        buy_r[idx[unresolved_b]] = buy_timeout[unresolved_b]
        sell_r[idx[unresolved_s]] = sell_timeout[unresolved_s]
        buy_exit_bar[idx[unresolved_b]] = MAX_HOLD_BARS
        sell_exit_bar[idx[unresolved_s]] = MAX_HOLD_BARS

    target = np.full(n, np.nan, dtype=np.float64)
    best_r = np.full(n, np.nan, dtype=np.float64)

    valid_rewards = eligible & np.isfinite(buy_r) & np.isfinite(sell_r)
    if valid_rewards.any():
        best_r[valid_rewards] = np.maximum(buy_r[valid_rewards], sell_r[valid_rewards])

        buy_choice = (
            valid_rewards
            & (buy_r >= MIN_EDGE_R)
            & (buy_r > sell_r)
        )
        sell_choice = (
            valid_rewards
            & (sell_r >= MIN_EDGE_R)
            & (sell_r > buy_r)
        )
        hold_choice = valid_rewards & ~(buy_choice | sell_choice)

        target[buy_choice] = BUY_CLASS
        target[sell_choice] = SELL_CLASS
        target[hold_choice] = HOLD_CLASS

    result["target_class"] = target
    result["target_buy_r"] = buy_r
    result["target_sell_r"] = sell_r
    result["target_best_r"] = best_r
    result["target_buy_exit_bar"] = buy_exit_bar
    result["target_sell_exit_bar"] = sell_exit_bar

    result.drop(
        columns=["future_return", "target_short", "target_long", "target"],
        inplace=True,
        errors="ignore",
    )

    return result


def add_training_targets(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Create execution-aware BUY/HOLD/SELL labels for one symbol."""
    resolved_symbol = _resolve_symbol(df, symbol)
    return _label_one_symbol(df, resolved_symbol)
