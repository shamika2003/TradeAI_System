from __future__ import annotations

import numpy as np
import pandas as pd

from Trade_Bot_Training.data_collector import (
    MarketDataCollector,
)
from Trade_Bot_Training.precision_feature_engine import (
    build_precision_features,
)
from Trade_Bot_Training.precision_model import (
    PrecisionDirectionModel,
)


RECENT_M5_BARS = 1200


def predict_latest(
    symbol: str,
) -> None:
    symbol = symbol.upper()

    collector = MarketDataCollector()

    try:
        # Use a generous history window. The model's longest rolling feature is
        # 288 M5 bars; 1,200 bars gives ample warm-up and resilience to gaps.
        m5 = collector.fetch_recent_closed(
            symbol,
            "M5",
            RECENT_M5_BARS,
        )
    finally:
        collector.disconnect()

    if m5.empty:
        raise RuntimeError(
            f"No closed M5 candles returned for {symbol}."
        )

    frame = build_precision_features(
        m5
    )

    if frame.empty:
        raise RuntimeError(
            f"Unable to build precision features for {symbol}."
        )

    model = PrecisionDirectionModel(
        symbol
    )

    missing_columns = [
        c
        for c in model.feature_columns
        if c not in frame.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Precision feature-engine/model mismatch. "
            "Missing columns: "
            + ", ".join(
                missing_columns[:20]
            )
        )

    # IMPORTANT:
    # Never silently fall back to an older row. The live prediction must use
    # the newest fully closed M5 candle or fail loudly.
    latest = frame.tail(1).copy()

    values = latest[
        model.feature_columns
    ].astype(
        "float64"
    )

    bad_columns = [
        c
        for c in model.feature_columns
        if not np.isfinite(
            values[
                c
            ].iloc[
                0
            ]
        )
    ]

    latest_bar_time = pd.Timestamp(
        latest[
            "time"
        ].iloc[
            0
        ]
    )

    decision_time = pd.Timestamp(
        latest[
            "decision_time"
        ].iloc[
            0
        ]
    )

    newest_mt5_bar_time = pd.Timestamp(
        m5[
            "time"
        ].iloc[
            -1
        ]
    )

    expected_decision_time = (
        newest_mt5_bar_time
        + pd.Timedelta(
            minutes=5
        )
    )

    if latest_bar_time != newest_mt5_bar_time:
        raise RuntimeError(
            "LIVE FRESHNESS ERROR: feature engine did not "
            "finish on the newest closed M5 candle. "
            f"Newest MT5 candle={newest_mt5_bar_time}, "
            f"feature candle={latest_bar_time}."
        )

    if decision_time != expected_decision_time:
        raise RuntimeError(
            "LIVE ALIGNMENT ERROR: decision time is not the "
            "next M5 boundary. "
            f"Expected={expected_decision_time}, "
            f"actual={decision_time}."
        )

    if bad_columns:
        raise RuntimeError(
            "LATEST M5 FEATURES ARE INVALID. "
            "The predictor will NOT use an older candle. "
            "Invalid features: "
            + ", ".join(
                bad_columns[:30]
            )
        )

    pred = model.predict(
        latest
    ).iloc[
        0
    ]

    now_utc = pd.Timestamp.now(
        tz="UTC"
    )

    data_age = (
        now_utc
        - decision_time
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"TRADEAI SELECTIVE PRECISION MODEL | "
        f"{symbol}"
    )

    print(
        "=" * 78
    )

    print(
        f"Latest closed M5 : "
        f"{latest_bar_time}"
    )

    print(
        f"Decision time    : "
        f"{decision_time}"
    )

    print(
        f"Data age         : "
        f"{data_age}"
    )

    print(
        f"P(DOWN)          : "
        f"{pred['p_down']:.4f}"
    )

    print(
        f"P(HOLD)          : "
        f"{pred['p_hold']:.4f}"
    )

    print(
        f"P(UP)            : "
        f"{pred['p_up']:.4f}"
    )

    print(
        f"Raw class        : "
        f"{pred['raw_class']}"
    )

    print(
        f"Confidence       : "
        f"{pred['confidence']:.4f}"
    )

    print(
        f"Signal           : "
        f"{pred['signal']}"
    )

    print(
        f"Tradeable        : "
        f"{bool(pred['tradeable'])}"
    )

    print(
        "MODEL OUTPUT ONLY - execution is a separate layer."
    )
