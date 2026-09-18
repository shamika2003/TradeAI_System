from __future__ import annotations

import numpy as np
import pandas as pd


SETUP_ENGINE_VERSION = "tradeai_structural_setup_v1_20260918"


def _clip01(x):
    return np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)


def _signed_unit(x, scale=1.0):
    x = np.asarray(x, dtype=np.float64) / max(float(scale), 1e-9)
    return np.tanh(x)


def _positive_unit(x, scale=1.0):
    return 0.5 + 0.5 * _signed_unit(x, scale)


def add_structural_setup_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal, interpretable setup scores used by the meta-label model.

    The scores deliberately do not inspect future candles. They summarize three
    broad FX setup families:
      * trend continuation,
      * breakout continuation,
      * pullback/recovery inside a higher-timeframe trend.

    They are not trading signals by themselves. The ML meta-model still decides
    whether a structurally valid setup has enough probability/expected value to
    risk money.
    """
    out = df.copy()

    def col(name, default=0.0):
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").fillna(default).to_numpy(dtype=np.float64)
        return np.full(len(out), float(default), dtype=np.float64)

    ema_m5 = col("ema21_55_atr")
    ema_fast = col("ema8_21_atr")
    slope_m5 = col("trend_slope_12_atr")
    slope_fast = col("ema21_slope_6_atr")
    ema_h1 = col("h1_ema21_55_atr")
    slope_h1 = col("h1_trend_slope_12_atr")
    mom3 = col("momentum_3_atr")
    mom12 = col("momentum_12_atr")
    mom24 = col("momentum_24_atr")
    h1_mom = col("h1_momentum_12_atr")
    di = col("di_spread14")
    adx = col("adx14")
    h1_adx = col("h1_adx14")
    eff12 = col("trend_efficiency_12")
    eff48 = col("trend_efficiency_48")
    range_exp = col("range_expansion_20", 1.0)
    volume_z = col("volume_z20")
    breakout_up20 = col("breakout_up_20_atr")
    breakout_down20 = col("breakout_down_20_atr")
    breakout_up50 = col("breakout_up_50_atr")
    breakout_down50 = col("breakout_down_50_atr")
    range_pos20 = col("range_position_20", 0.5)
    range_pos50 = col("range_position_50", 0.5)
    close_loc = col("close_location", 0.5)
    signed_body = col("signed_body_atr")
    wick_imb = col("wick_imbalance")
    rsi = col("rsi14", 50.0)
    rsi_slope = col("rsi_slope_3")
    ema8_gap = col("ema8_gap_atr")
    vol_ratio = col("volatility_ratio", 1.0)
    vol_term = col("volatility_term_5_60", 1.0)
    atr_rank = col("atr_rank_200", 0.5)
    spread_rel = col("spread_relative_100", 1.0)

    # Higher/lower timeframe directional consensus.
    trend_raw = (
        0.95 * ema_m5
        + 0.55 * ema_fast
        + 0.55 * slope_m5
        + 0.30 * slope_fast
        + 0.95 * ema_h1
        + 0.55 * slope_h1
        + 0.35 * h1_mom
        + 0.25 * di
    )
    trend_buy = _positive_unit(trend_raw, 1.75)
    trend_sell = _positive_unit(-trend_raw, 1.75)

    momentum_raw = (
        0.30 * mom3
        + 0.45 * mom12
        + 0.20 * mom24
        + 0.25 * h1_mom
        + 0.20 * signed_body
        + 0.15 * rsi_slope * 4.0
    )
    momentum_buy = _positive_unit(momentum_raw, 1.50)
    momentum_sell = _positive_unit(-momentum_raw, 1.50)

    breakout_buy_raw = (
        0.65 * np.maximum(breakout_up20, 0.0)
        + 0.35 * np.maximum(breakout_up50, 0.0)
        + 0.30 * np.maximum(range_pos20 - 0.70, 0.0) * 3.0
        + 0.20 * np.maximum(range_pos50 - 0.65, 0.0) * 2.5
        + 0.18 * np.maximum(range_exp - 1.0, 0.0)
        + 0.10 * np.maximum(volume_z, 0.0)
    )
    breakout_sell_raw = (
        0.65 * np.maximum(breakout_down20, 0.0)
        + 0.35 * np.maximum(breakout_down50, 0.0)
        + 0.30 * np.maximum(0.30 - range_pos20, 0.0) * 3.0
        + 0.20 * np.maximum(0.35 - range_pos50, 0.0) * 2.5
        + 0.18 * np.maximum(range_exp - 1.0, 0.0)
        + 0.10 * np.maximum(volume_z, 0.0)
    )
    breakout_buy = _positive_unit(breakout_buy_raw - 0.20, 0.75)
    breakout_sell = _positive_unit(breakout_sell_raw - 0.20, 0.75)

    # Pullback/recovery: trend is already present, price has retraced toward the
    # fast EMA, and the current candle/momentum begins to re-assert direction.
    pullback_near_ema = np.exp(-np.square(ema8_gap / 0.85))
    buy_recovery = _clip01(
        0.35 * _positive_unit(signed_body, 0.60)
        + 0.25 * _positive_unit(mom3, 0.70)
        + 0.20 * _positive_unit(rsi_slope * 5.0, 0.60)
        + 0.10 * _clip01(close_loc)
        + 0.10 * _positive_unit(wick_imb, 0.70)
    )
    sell_recovery = _clip01(
        0.35 * _positive_unit(-signed_body, 0.60)
        + 0.25 * _positive_unit(-mom3, 0.70)
        + 0.20 * _positive_unit(-rsi_slope * 5.0, 0.60)
        + 0.10 * _clip01(1.0 - close_loc)
        + 0.10 * _positive_unit(-wick_imb, 0.70)
    )
    pullback_buy = _clip01(trend_buy * pullback_near_ema * buy_recovery)
    pullback_sell = _clip01(trend_sell * pullback_near_ema * sell_recovery)

    # Regime quality penalizes dead/chaotic/spread-heavy conditions without
    # forcing a particular direction.
    trend_quality = _clip01(
        0.35 * np.clip(adx / 0.35, 0.0, 1.0)
        + 0.20 * np.clip(h1_adx / 0.35, 0.0, 1.0)
        + 0.25 * np.clip((eff12 + eff48) / 1.10, 0.0, 1.0)
        + 0.20 * np.clip(range_exp / 1.60, 0.0, 1.0)
    )
    vol_penalty = np.clip(np.abs(vol_ratio - 1.0) / 2.0, 0.0, 0.65)
    term_penalty = np.clip(np.maximum(vol_term - 2.5, 0.0) / 3.0, 0.0, 0.35)
    spread_penalty = np.clip(np.maximum(spread_rel - 1.5, 0.0) / 4.0, 0.0, 0.60)
    extreme_atr_penalty = np.clip(np.maximum(atr_rank - 0.95, 0.0) * 4.0, 0.0, 0.20)
    regime_quality = _clip01(
        0.35 + 0.65 * trend_quality
        - vol_penalty
        - term_penalty
        - spread_penalty
        - extreme_atr_penalty
    )

    continuation_buy = _clip01(0.48 * trend_buy + 0.34 * momentum_buy + 0.18 * breakout_buy)
    continuation_sell = _clip01(0.48 * trend_sell + 0.34 * momentum_sell + 0.18 * breakout_sell)
    pullback_path_buy = _clip01(0.62 * trend_buy + 0.38 * pullback_buy)
    pullback_path_sell = _clip01(0.62 * trend_sell + 0.38 * pullback_sell)

    buy_setup = _clip01(regime_quality * np.maximum(continuation_buy, pullback_path_buy))
    sell_setup = _clip01(regime_quality * np.maximum(continuation_sell, pullback_path_sell))

    out["setup_trend_buy"] = trend_buy
    out["setup_trend_sell"] = trend_sell
    out["setup_momentum_buy"] = momentum_buy
    out["setup_momentum_sell"] = momentum_sell
    out["setup_breakout_buy"] = breakout_buy
    out["setup_breakout_sell"] = breakout_sell
    out["setup_pullback_buy"] = pullback_buy
    out["setup_pullback_sell"] = pullback_sell
    out["setup_regime_quality"] = regime_quality
    out["setup_buy_score"] = buy_setup
    out["setup_sell_score"] = sell_setup
    out["setup_score_gap"] = buy_setup - sell_setup

    return out


def setup_candidate_masks(
    frame: pd.DataFrame,
    *,
    min_score: float,
    min_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    buy = np.asarray(frame["setup_buy_score"], dtype=np.float64)
    sell = np.asarray(frame["setup_sell_score"], dtype=np.float64)
    gap = buy - sell
    regime = np.asarray(frame.get("setup_regime_quality", 1.0), dtype=np.float64)

    quality_ok = regime >= 0.30
    buy_mask = quality_ok & (buy >= float(min_score)) & (gap >= float(min_gap))
    sell_mask = quality_ok & (sell >= float(min_score)) & (gap <= -float(min_gap))
    return buy_mask, sell_mask
