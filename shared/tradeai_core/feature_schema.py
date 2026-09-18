from __future__ import annotations

import hashlib
import json


FEATURE_SCHEMA_VERSION = "tradeai_features_v7_structural_meta_20260918"
M5_BAR_MINUTES = 5
H1_BAR_MINUTES = 60

# Stationary / scale-normalized market-state features. V7 adds structural setup
# meta-features on top of the directional
# candle pressure, multi-horizon momentum/volatility and richer H1 regime state
# so the opportunity model can rank rare setups rather than memorize price.
FEATURE_NAMES = [
    "return_1", "return_3", "return_6", "return_12", "return_24", "return_48", "return_96",
    "range_pct", "body_pct", "signed_body_atr", "wick_imbalance", "upper_wick_pct", "lower_wick_pct",
    "close_location", "close_location_mean_6", "range_expansion_20",
    "rsi14", "rsi_slope_3",
    "atr", "atr_pct", "atr_ratio", "atr_rank_200",
    "ema8_gap_atr", "ema21_gap_atr", "ema55_gap_atr", "ema8_21_atr", "ema21_55_atr",
    "ema21_slope_6_atr", "ema55_slope_24_atr", "trend_slope_12_atr",
    "momentum_3_atr", "momentum_12_atr", "momentum_24_atr", "momentum_48_atr",
    "trend_efficiency_12", "trend_efficiency_48", "up_fraction_12", "up_fraction_48",
    "volatility_5", "volatility_20", "volatility_60", "volatility_ratio", "volatility_term_5_60",
    "range_position_20", "range_position_50", "range_position_100",
    "breakout_up_20_atr", "breakout_down_20_atr", "breakout_up_50_atr", "breakout_down_50_atr",
    "bb_z20", "bb_width20",
    "adx14", "plus_di14", "minus_di14", "di_spread14",
    "volume_z20", "volume_z50", "volume_change", "spread_relative_100",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "session_asia", "session_london", "session_newyork", "session_overlap_london_ny",
    "h1_rsi14", "h1_rsi_slope_3", "h1_adx14", "h1_atr_pct",
    "h1_ema8_21_atr", "h1_ema21_55_atr", "h1_trend_slope_12_atr", "h1_momentum_12_atr",
    "h1_volatility_ratio", "h1_bb_z20", "h1_di_spread14",
    "h1_range_position_20", "h1_range_position_50", "m5_h1_alignment",
    "setup_trend_buy", "setup_trend_sell",
    "setup_momentum_buy", "setup_momentum_sell",
    "setup_breakout_buy", "setup_breakout_sell",
    "setup_pullback_buy", "setup_pullback_sell",
    "setup_regime_quality", "setup_buy_score", "setup_sell_score", "setup_score_gap",
]

_SCHEMA_SEMANTICS = {
    "schema_version": FEATURE_SCHEMA_VERSION,
    "features": FEATURE_NAMES,
    "m5_bar_minutes": M5_BAR_MINUTES,
    "h1_bar_minutes": H1_BAR_MINUTES,
    "technical_rule_version": "quality_stationary_v3_structural_meta",
    "m5_decision_rule": "M5 feature row is available only after the M5 candle closes",
    "h1_availability_rule": "H1 feature row is available only after the H1 candle closes",
    "h1_join_rule": "latest H1 available_time <= M5 decision_time; backward asof merge",
    "same_candle_rule": "no future candle information is used in any feature",
    "structural_setup_rule": "causal trend/breakout/pullback scores computed from closed M5 and available H1 state",
}


def _calculate_hash() -> str:
    payload = json.dumps(_SCHEMA_SEMANTICS, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


FEATURE_HASH = _calculate_hash()


def feature_schema_metadata() -> dict:
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "feature_names": list(FEATURE_NAMES),
        "m5_bar_minutes": M5_BAR_MINUTES,
        "h1_bar_minutes": H1_BAR_MINUTES,
    }
