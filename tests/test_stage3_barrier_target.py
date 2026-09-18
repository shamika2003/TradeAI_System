import numpy as np
import pandas as pd

from shared.tradeai_core.target_definition import (
    BUY_CLASS,
    HOLD_CLASS,
    MAX_HOLD_BARS,
    add_training_targets,
)


def _base_frame(n=MAX_HOLD_BARS + 8):
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
    # Mild positive timeout move inside the horizon must remain HOLD; v5 only
    # awards a directional label to a full TP hit.
    df1.loc[MAX_HOLD_BARS, "close"] = 1.010
    out1 = add_training_targets(df1, symbol="EURUSD")
    first_label = out1.loc[0, "target_class"]
    assert first_label == HOLD_CLASS

    df2 = df1.copy()
    # Huge move strictly AFTER row-0 horizon must not change row-0 label.
    df2.loc[MAX_HOLD_BARS + 1, ["high", "close"]] = [2.0, 2.0]
    out2 = add_training_targets(df2, symbol="EURUSD")
    assert out2.loc[0, "target_class"] == first_label


def test_payoff_contract_targets_two_and_half_r_gross_reward():
    from shared.tradeai_core.target_definition import target_contract

    contract = target_contract()
    assert contract["gross_reward_to_risk"] == 2.5
    assert contract["historical_ohlc_side"] == "BID"


def test_buy_label_charges_full_entry_spread():
    # Build a clean TP hit where only row-0 spread differs.
    base = _base_frame(n=MAX_HOLD_BARS + 4)
    base["high"] = 1.0
    base["low"] = 1.0
    base["close"] = 1.0
    base["atr"] = 0.001
    base.loc[1, "high"] = 1.0040

    zero = base.copy()
    zero["spread"] = 0.0
    with_spread = base.copy()
    with_spread["spread"] = 0.0
    with_spread.loc[0, "spread"] = 20.0  # 2.0 pips at 5-digit EURUSD

    r0 = add_training_targets(zero, symbol="EURUSD").loc[0, "target_buy_r"]
    r1 = add_training_targets(with_spread, symbol="EURUSD").loc[0, "target_buy_r"]
    assert r1 < r0


def test_sell_label_uses_future_ask_for_stop_trigger():
    df = _base_frame(n=MAX_HOLD_BARS + 4)
    df["high"] = 1.0
    df["low"] = 1.0
    df["close"] = 1.0
    df["atr"] = 0.001
    df["spread"] = 10.0  # 1 pip ask premium

    # Sell stop from row 0 BID close is 1.00125. Future BID high remains below
    # it at 1.00120, but ASK high = 1.00130, so production would stop out.
    df.loc[1, "high"] = 1.00120
    out = add_training_targets(df, symbol="EURUSD")
    assert out.loc[0, "target_sell_r"] < 0


def test_small_positive_timeout_is_hold_but_full_tp_is_directional():
    quiet = _base_frame(n=MAX_HOLD_BARS + 4)
    quiet["high"] = 1.0
    quiet["low"] = 1.0
    quiet["close"] = 1.0
    quiet["atr"] = 0.001
    quiet.loc[MAX_HOLD_BARS, "close"] = 1.0004
    out = add_training_targets(quiet, symbol="EURUSD")
    assert out.loc[0, "target_class"] == HOLD_CLASS

    winner = quiet.copy()
    winner.loc[1, "high"] = 1.0035  # beyond 3.125 ATR TP from 1.0
    out2 = add_training_targets(winner, symbol="EURUSD")
    assert out2.loc[0, "target_class"] == BUY_CLASS
    assert out2.loc[0, "target_buy_tp_hit"] == 1
