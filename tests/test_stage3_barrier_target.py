import numpy as np
import pandas as pd

from shared.tradeai_core.target_definition import (
    BUY_CLASS,
    MAX_HOLD_BARS,
    add_training_targets,
)


def _base_frame(n=40):
    return pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=n, freq="5min"),
            "open": np.ones(n),
            "high": np.ones(n) * 1.001,
            "low": np.ones(n) * 0.999,
            "close": np.ones(n),
            "atr": np.ones(n) * 0.01,
            "spread": np.zeros(n),
            "symbol": ["EURUSD"] * n,
        }
    )


def test_same_bar_sl_wins_conservatively():
    df = _base_frame()
    # Row 0 entry=1.0, SL=0.985, TP=1.03. Both are touched in future bar 1.
    df.loc[1, "low"] = 0.980
    df.loc[1, "high"] = 1.040
    out = add_training_targets(df, symbol="EURUSD")
    assert out.loc[0, "target_buy_r"] < 0


def test_target_does_not_look_beyond_max_horizon():
    df1 = _base_frame(n=MAX_HOLD_BARS + 8)
    # Mild positive timeout move inside the horizon => BUY candidate.
    df1.loc[MAX_HOLD_BARS, "close"] = 1.010
    out1 = add_training_targets(df1, symbol="EURUSD")
    first_label = out1.loc[0, "target_class"]

    df2 = df1.copy()
    # Huge move strictly AFTER row-0 horizon must not change row-0 label.
    df2.loc[MAX_HOLD_BARS + 1, ["high", "close"]] = [2.0, 2.0]
    out2 = add_training_targets(df2, symbol="EURUSD")
    assert out2.loc[0, "target_class"] == first_label
