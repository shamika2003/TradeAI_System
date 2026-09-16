# filename: shared/tradeai_core/feature_schema.py

from __future__ import annotations

import hashlib
import json


FEATURE_SCHEMA_VERSION = "tradeai_features_v4_20260907"

M5_BAR_MINUTES = 5
H1_BAR_MINUTES = 60

# IMPORTANT:
# This list is the production contract. Training and live prediction must use
# exactly these names, in exactly this order.
FEATURE_NAMES = [
    "return",
    "range",
    "ma5",
    "ma20",
    "trend",
    "tick_volume",
    "spread",
    "rsi",
    "atr",
    "momentum_3",
    "momentum_10",
    "volatility",
    "range_position",
    "volume_change",
    "atr_ratio",
    "trend_strength",
    "h1_ma20",
    "h1_rsi",
    "h1_trend_strength",
    "h1_trend_bias",
    "m5_h1_alignment",
    "structure_bias",
    "volatility_regime",
    "price_zscore",
    "volatility_change",
    "dist_to_high_50",
    "dist_to_low_50",
    "hour_sin",
    "hour_cos",
]

# Hash semantics as well as names. This prevents two implementations from
# silently using the same column names with different meanings.
_SCHEMA_SEMANTICS = {
    "schema_version": FEATURE_SCHEMA_VERSION,
    "features": FEATURE_NAMES,
    "m5_bar_minutes": M5_BAR_MINUTES,
    "h1_bar_minutes": H1_BAR_MINUTES,
    "h1_availability_rule": (
        "H1 OHLC/derived features become available at h1_open_time + 60 minutes"
    ),
    "m5_decision_rule": (
        "M5 feature row is evaluated at m5_open_time + 5 minutes (closed candle)"
    ),
    "h1_join_rule": (
        "latest H1 available_time <= M5 decision_time; backward merge; exact match allowed"
    ),
    "m5_h1_alignment_rule": (
        "1 when sign(m5 trend_strength) == sign(h1 trend_strength), otherwise 0"
    ),
    "technical_rule_version": "technical_v3_causal_mtf",
}


def _calculate_hash() -> str:
    payload = json.dumps(
        _SCHEMA_SEMANTICS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
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
