from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
TRADEAI = ROOT / "TradeAI"
for p in (ROOT, TRAINING, TRADEAI):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared.tradeai_core.target_definition import (
    QUALITY_SUCCESS_R,
    TARGET_VERSION,
    add_training_targets,
)
from training_utils import _apply_probability_calibration, _apply_ev_calibration


def _frame(n=60):
    rows = []
    for i in range(n):
        rows.append({
            "time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=5*i),
            "open": 1.0, "high": 1.0002, "low": 0.9998, "close": 1.0,
            "atr": 0.001, "spread": 0.0,
        })
    return pd.DataFrame(rows)


def test_quality_target_is_one_r_entry_gate_not_full_take_profit():
    assert TARGET_VERSION == "tradeai_compact_quality_target_v8_20260918"
    assert QUALITY_SUCCESS_R == pytest.approx(1.0)
    df = _frame()
    # Enough for +1R executable quality barrier, deliberately below the 2.5R TP.
    df.loc[1, "high"] = 1.0018
    out = add_training_targets(df, symbol="EURUSD")
    assert int(out.loc[0, "target_buy_quality_hit"]) == 1
    assert int(out.loc[0, "target_buy_tp_hit"]) == 0


def test_same_bar_stop_beats_quality_barrier():
    df = _frame()
    df.loc[1, "high"] = 1.0020
    df.loc[1, "low"] = 0.9980
    out = add_training_targets(df, symbol="EURUSD")
    assert int(out.loc[0, "target_buy_quality_hit"]) == 0


def test_probability_and_ev_calibration_are_applied_monotonically():
    p = np.array([0.2, 0.5, 0.8])
    calibrated = _apply_probability_calibration(p, {"kind": "platt", "a": 1.2, "b": -0.1})
    assert np.all(np.diff(calibrated) > 0)
    ev = _apply_ev_calibration(np.array([-1.0, 0.0, 1.0]), {"kind": "linear", "slope": 0.5, "intercept": 0.1})
    assert ev.tolist() == pytest.approx([-0.4, 0.1, 0.6])
