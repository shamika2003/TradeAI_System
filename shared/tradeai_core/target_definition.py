# filename: shared/tradeai_core/target_definition.py

from __future__ import annotations

import numpy as np
import pandas as pd


TARGET_VERSION = "tradeai_barrier_target_v3_20260907"
PREDICTION_TYPE = "barrier_multiclass"

# Must stay aligned with TradeAI/config/settings.py initial ATR stop profile.
STOP_ATR_MULTIPLIER = 1.5
TAKE_ATR_MULTIPLIER = 3.0
MAX_HOLD_BARS = 24          # 2 hours on M5
MAX_TARGET_HORIZON_BARS = MAX_HOLD_BARS
PRIMARY_HORIZON_BARS = MAX_HOLD_BARS

# Backward-compatibility alias retained for the Stage 1 package __init__.py.
# Stage 3 has one execution-aligned barrier horizon, so the legacy long
# horizon name points to the same maximum holding horizon.
LONG_HORIZON_BARS = MAX_HOLD_BARS

# XGBoost multiclass IDs. Keep stable: they are part of the model contract.
SELL_CLASS = 0
HOLD_CLASS = 1
BUY_CLASS = 2
CLASS_MAP = {
    "SELL": SELL_CLASS,
    "HOLD": HOLD_CLASS,
    "BUY": BUY_CLASS,
}

# A small positive realized edge is required before a row is labelled as a
# directional opportunity. This prevents tiny/noisy timeout moves from being
# forced into BUY or SELL.
MIN_EDGE_R = 0.10

# Cost proxy used ONLY for label construction. Production backtest remains the
# authority for actual money P/L. It mirrors the existing PaperExecutor entry
# model: half spread + 0.2 pip adverse slippage.
LABEL_SLIPPAGE_PIPS = 0.20

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


def target_contract() -> dict:
    return {
        "target_version": TARGET_VERSION,
        "prediction_type": PREDICTION_TYPE,
        "stop_atr_multiplier": STOP_ATR_MULTIPLIER,
        "take_atr_multiplier": TAKE_ATR_MULTIPLIER,
        "max_hold_bars": MAX_HOLD_BARS,
        "min_edge_r": MIN_EDGE_R,
        "label_slippage_pips": LABEL_SLIPPAGE_PIPS,
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

    buy_r = np.full(n, np.nan, dtype=np.float64)
    sell_r = np.full(n, np.nan, dtype=np.float64)
    buy_exit_bar = np.full(n, -1, dtype=np.int16)
    sell_exit_bar = np.full(n, -1, dtype=np.int16)

    # Rows that do not have a complete future horizon are intentionally left
    # NaN. The dataset builder/relabeler removes them explicitly.
    full_horizon = np.zeros(n, dtype=bool)
    if n > MAX_HOLD_BARS:
        full_horizon[: n - MAX_HOLD_BARS] = True

    eligible = valid_base & full_horizon
    buy_active = eligible.copy()
    sell_active = eligible.copy()

    buy_sl = close - sl_distance
    buy_tp = close + tp_distance
    sell_sl = close + sl_distance
    sell_tp = close - tp_distance

    # Vectorized horizon scan. Earliest future candle resolves each simulated
    # direction. If SL and TP are touched inside the same OHLC candle, SL wins,
    # matching TradeAI PaperExecutor's conservative ambiguity rule.
    for step in range(1, MAX_HOLD_BARS + 1):
        count = n - step
        if count <= 0:
            break

        idx = np.arange(count)
        fut_high = high[step:]
        fut_low = low[step:]

        active_b = buy_active[:count]
        if active_b.any():
            sl_hit = fut_low <= buy_sl[:count]
            tp_hit = fut_high >= buy_tp[:count]
            resolve_sl = active_b & sl_hit
            resolve_tp = active_b & (~sl_hit) & tp_hit

            buy_r[idx[resolve_sl]] = -1.0
            buy_r[idx[resolve_tp]] = TAKE_ATR_MULTIPLIER / STOP_ATR_MULTIPLIER
            buy_exit_bar[idx[resolve_sl | resolve_tp]] = step
            buy_active[idx[resolve_sl | resolve_tp]] = False

        active_s = sell_active[:count]
        if active_s.any():
            sl_hit = fut_high >= sell_sl[:count]
            tp_hit = fut_low <= sell_tp[:count]
            resolve_sl = active_s & sl_hit
            resolve_tp = active_s & (~sl_hit) & tp_hit

            sell_r[idx[resolve_sl]] = -1.0
            sell_r[idx[resolve_tp]] = TAKE_ATR_MULTIPLIER / STOP_ATR_MULTIPLIER
            sell_exit_bar[idx[resolve_sl | resolve_tp]] = step
            sell_active[idx[resolve_sl | resolve_tp]] = False

    # Timeout rows exit at the close of the maximum holding bar. This prevents
    # unresolved samples from being discarded and allows HOLD labels to form.
    base_count = max(0, n - MAX_HOLD_BARS)
    if base_count:
        idx = np.arange(base_count)
        terminal = close[MAX_HOLD_BARS:]
        directional_r = (terminal - close[:base_count]) / sl_distance[:base_count]
        directional_r = np.clip(directional_r, -1.0, TAKE_ATR_MULTIPLIER / STOP_ATR_MULTIPLIER)

        unresolved_b = buy_active[:base_count]
        unresolved_s = sell_active[:base_count]

        buy_r[idx[unresolved_b]] = directional_r[unresolved_b]
        sell_r[idx[unresolved_s]] = -directional_r[unresolved_s]
        buy_exit_bar[idx[unresolved_b]] = MAX_HOLD_BARS
        sell_exit_bar[idx[unresolved_s]] = MAX_HOLD_BARS

    # Approximate entry friction in R units. Spread from MT5 OHLC is in points.
    spread_points = (
        result["spread"].to_numpy(dtype=np.float64)
        if "spread" in result.columns
        else np.zeros(n, dtype=np.float64)
    )
    spread_points = np.nan_to_num(spread_points, nan=0.0, posinf=0.0, neginf=0.0)
    spread_points = np.maximum(spread_points, 0.0)

    point = _SYMBOL_POINT_SIZE.get(symbol, 0.00001)
    pip = _SYMBOL_PIP_SIZE.get(symbol, 0.0001)
    entry_cost_price = (0.5 * spread_points * point) + (LABEL_SLIPPAGE_PIPS * pip)
    cost_r = np.divide(
        entry_cost_price,
        sl_distance,
        out=np.zeros(n, dtype=np.float64),
        where=sl_distance > 0,
    )

    buy_r[eligible] -= cost_r[eligible]
    sell_r[eligible] -= cost_r[eligible]

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

    # Retire old return-regression labels if they are present. Keeping them in
    # the same production dataset invites accidental reuse.
    result.drop(
        columns=["future_return", "target_short", "target_long", "target"],
        inplace=True,
        errors="ignore",
    )

    return result


def add_training_targets(df: pd.DataFrame, symbol: str | None = None) -> pd.DataFrame:
    """Create execution-aligned, causal BUY/HOLD/SELL labels for one symbol."""
    resolved_symbol = _resolve_symbol(df, symbol)
    return _label_one_symbol(df, resolved_symbol)
