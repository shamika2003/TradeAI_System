# filename: shared/tradeai_core/target_definition.py

from __future__ import annotations

import numpy as np
import pandas as pd


# Stage 9: execution-aware directional opportunity target contract.
TARGET_VERSION = "tradeai_compact_quality_target_v8_20260918"
PREDICTION_TYPE = "directional_quality_barrier_plus_expected_r"

# Runtime and labels intentionally share the same geometry.
# 0.75 ATR stop / 1.875 ATR target = 2.5R gross reward:risk.
# Stage 11.2 uses a compact invalidation stop because the live broker has a
# 0.01 minimum lot and a $150 account cannot safely execute many 1.25 ATR M5
# stops. The 2.5R payoff ratio is preserved; only the price distance is tighter.
# Labels and runtime use the exact same geometry.
STOP_ATR_MULTIPLIER = 0.75
TAKE_ATR_MULTIPLIER = 1.875
MAX_HOLD_BARS = 48          # 4 hours on M5
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
MIN_EDGE_R = 0.20
# Classifier target: can price achieve +1.0 net R before the original -1R stop?
# Runtime is still allowed to hold winners toward the larger 2.5R target.
QUALITY_SUCCESS_R = 1.00
MIN_TP_SUCCESS_R = 1.50

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
        "quality_success_r": QUALITY_SUCCESS_R,
        "min_tp_success_r": MIN_TP_SUCCESS_R,
        "directional_label_rule": "quality_barrier_hit_before_stop_within_horizon",
        "model_objective": "separate_buy_sell_quality_probability_plus_expected_net_r",
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

    pip_value = _pip_value_per_lot(symbol, close)

    # R is normalized by the *actual executable net loss at the original stop*,
    # including spread/slippage displacement and round-trip commission. This is
    # the same risk concept used by Stage 2 runtime sizing.
    buy_risk_money = ((buy_entry - buy_sl) / pip) * pip_value + LABEL_COMMISSION_PER_LOT
    sell_risk_money = ((sell_sl - sell_entry) / pip) * pip_value + LABEL_COMMISSION_PER_LOT

    def _buy_reward(exit_price):
        pnl = ((exit_price - buy_entry) / pip) * pip_value - LABEL_COMMISSION_PER_LOT
        return np.divide(
            pnl, buy_risk_money, out=np.full(n, np.nan, dtype=np.float64), where=buy_risk_money > 0
        )

    def _sell_reward(exit_price):
        pnl = ((sell_entry - exit_price) / pip) * pip_value - LABEL_COMMISSION_PER_LOT
        return np.divide(
            pnl, sell_risk_money, out=np.full(n, np.nan, dtype=np.float64), where=sell_risk_money > 0
        )

    buy_stop_reward = _buy_reward(buy_sl)
    buy_tp_reward = _buy_reward(buy_tp)
    sell_stop_reward = _sell_reward(sell_sl)
    sell_tp_reward = _sell_reward(sell_tp)

    # A +1R quality barrier is intentionally easier to learn than the final
    # +2.5R take-profit. Entry quality and exit ambition are separate jobs.
    buy_quality_price = buy_entry + ((QUALITY_SUCCESS_R * buy_risk_money + LABEL_COMMISSION_PER_LOT) / (pip_value + 1e-12)) * pip
    sell_quality_price = sell_entry - ((QUALITY_SUCCESS_R * sell_risk_money + LABEL_COMMISSION_PER_LOT) / (pip_value + 1e-12)) * pip

    buy_r = np.full(n, np.nan, dtype=np.float64)
    sell_r = np.full(n, np.nan, dtype=np.float64)
    buy_exit_bar = np.full(n, -1, dtype=np.int16)
    sell_exit_bar = np.full(n, -1, dtype=np.int16)
    buy_tp_success = np.zeros(n, dtype=bool)
    sell_tp_success = np.zeros(n, dtype=bool)
    buy_quality_success = np.zeros(n, dtype=bool)
    sell_quality_success = np.zeros(n, dtype=bool)
    buy_quality_bar = np.full(n, -1, dtype=np.int16)
    sell_quality_bar = np.full(n, -1, dtype=np.int16)

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
            quality_hit = fut_high_bid >= buy_quality_price[:count]
            first_quality = active_b & (~sl_hit) & quality_hit & (~buy_quality_success[:count])
            buy_quality_success[idx[first_quality]] = True
            buy_quality_bar[idx[first_quality]] = step
            tp_hit = fut_high_bid >= buy_tp[:count]
            resolve_sl = active_b & sl_hit
            resolve_tp = active_b & (~sl_hit) & tp_hit

            buy_r[idx[resolve_sl]] = buy_stop_reward[:count][resolve_sl]
            buy_r[idx[resolve_tp]] = buy_tp_reward[:count][resolve_tp]
            buy_tp_success[idx[resolve_tp]] = True
            buy_exit_bar[idx[resolve_sl | resolve_tp]] = step
            buy_active[idx[resolve_sl | resolve_tp]] = False

        active_s = sell_active[:count]
        if active_s.any():
            sl_hit = fut_high_ask >= sell_sl[:count]
            quality_hit = fut_low_ask <= sell_quality_price[:count]
            first_quality = active_s & (~sl_hit) & quality_hit & (~sell_quality_success[:count])
            sell_quality_success[idx[first_quality]] = True
            sell_quality_bar[idx[first_quality]] = step
            tp_hit = fut_low_ask <= sell_tp[:count]
            resolve_sl = active_s & sl_hit
            resolve_tp = active_s & (~sl_hit) & tp_hit

            sell_r[idx[resolve_sl]] = sell_stop_reward[:count][resolve_sl]
            sell_r[idx[resolve_tp]] = sell_tp_reward[:count][resolve_tp]
            sell_tp_success[idx[resolve_tp]] = True
            sell_exit_bar[idx[resolve_sl | resolve_tp]] = step
            sell_active[idx[resolve_sl | resolve_tp]] = False

    # Timeout at the maximum horizon using production quote sides.
    base_count = max(0, n - MAX_HOLD_BARS)
    if base_count:
        idx = np.arange(base_count)
        terminal_bid = close[MAX_HOLD_BARS:]
        terminal_ask = terminal_bid + spread_price[MAX_HOLD_BARS:]

        buy_timeout_money = (
            ((terminal_bid - buy_entry[:base_count]) / pip) * pip_value[:base_count]
            - LABEL_COMMISSION_PER_LOT
        )
        sell_timeout_money = (
            ((sell_entry[:base_count] - terminal_ask) / pip) * pip_value[:base_count]
            - LABEL_COMMISSION_PER_LOT
        )
        buy_timeout = np.divide(
            buy_timeout_money,
            buy_risk_money[:base_count],
            out=np.full(base_count, np.nan, dtype=np.float64),
            where=buy_risk_money[:base_count] > 0,
        )
        sell_timeout = np.divide(
            sell_timeout_money,
            sell_risk_money[:base_count],
            out=np.full(base_count, np.nan, dtype=np.float64),
            where=sell_risk_money[:base_count] > 0,
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

        # Entry labels answer a narrower, more learnable question than the final
        # exit target: which side reaches +1.0 net R before its stop?  This keeps
        # quality-over-quantity while letting runtime management hold exceptional
        # trades toward the larger 2.5R take-profit.
        buy_only = valid_rewards & buy_quality_success & ~sell_quality_success
        sell_only = valid_rewards & sell_quality_success & ~buy_quality_success
        both = valid_rewards & buy_quality_success & sell_quality_success

        buy_first = both & (buy_quality_bar < sell_quality_bar)
        sell_first = both & (sell_quality_bar < buy_quality_bar)
        same_bar = both & (buy_quality_bar == sell_quality_bar)
        buy_tie = same_bar & (buy_r >= sell_r)
        sell_tie = same_bar & ~buy_tie

        buy_choice = buy_only | buy_first | buy_tie
        sell_choice = sell_only | sell_first | sell_tie
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
    result["target_buy_tp_hit"] = buy_tp_success.astype(np.int8)
    result["target_sell_tp_hit"] = sell_tp_success.astype(np.int8)
    result["target_buy_quality_hit"] = buy_quality_success.astype(np.int8)
    result["target_sell_quality_hit"] = sell_quality_success.astype(np.int8)
    result["target_buy_quality_bar"] = buy_quality_bar
    result["target_sell_quality_bar"] = sell_quality_bar

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
