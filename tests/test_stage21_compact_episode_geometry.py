from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
TRADEAI = ROOT / "TradeAI"
for path in (ROOT, TRAINING, TRADEAI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from TradeAI.config import settings
from shared.tradeai_core.decision_policy import SIGNAL_EPISODE_RESET_BARS
from shared.tradeai_core.target_definition import STOP_ATR_MULTIPLIER, TAKE_ATR_MULTIPLIER
from training_utils import decluster_actions
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, SELL_CLASS


def test_compact_geometry_keeps_2_5r_but_reduces_min_lot_stop_exposure():
    assert STOP_ATR_MULTIPLIER == pytest.approx(0.75)
    assert TAKE_ATR_MULTIPLIER == pytest.approx(1.875)
    assert TAKE_ATR_MULTIPLIER / STOP_ATR_MULTIPLIER == pytest.approx(2.5)
    assert settings.ATR_SL_MULTIPLIER == pytest.approx(STOP_ATR_MULTIPLIER)
    assert settings.ATR_TP_MULTIPLIER == pytest.approx(TAKE_ATR_MULTIPLIER)
    assert STOP_ATR_MULTIPLIER < 1.0


def test_decluster_actions_counts_continuous_signal_as_one_opportunity_episode():
    raw = np.array([
        HOLD_CLASS,
        BUY_CLASS, BUY_CLASS, BUY_CLASS,
        HOLD_CLASS, HOLD_CLASS,
        BUY_CLASS,  # not reset yet
        HOLD_CLASS, HOLD_CLASS, HOLD_CLASS,
        BUY_CLASS,  # new episode
        SELL_CLASS, SELL_CLASS,  # direct reversal = one new SELL episode
    ])
    out = decluster_actions(raw, reset_bars=3)
    selected = np.flatnonzero(out != HOLD_CLASS).tolist()
    assert selected == [1, 10, 11]
    assert out[1] == BUY_CLASS
    assert out[10] == BUY_CLASS
    assert out[11] == SELL_CLASS
    assert SIGNAL_EPISODE_RESET_BARS == 3


def test_compact_stop_improves_minimum_lot_feasibility_at_same_risk_cap():
    # EURUSD 0.01 lot is about $0.10/pip. With $0.07 commission and a
    # 0.5625% hard cap on $150, the stop can be ~7.74 pips. A 10-pip ATR
    # therefore fits under the new 0.85 ATR stop but not the old 1.25 ATR stop.
    balance = 150.0
    hard_cap = balance * (0.45 * 1.25) / 100.0
    commission = 0.07
    pip_value_001 = 0.10
    atr_pips = 10.0

    old_risk = (atr_pips * 1.25) * pip_value_001 + commission
    new_risk = (atr_pips * STOP_ATR_MULTIPLIER) * pip_value_001 + commission

    assert old_risk > hard_cap
    assert new_risk <= hard_cap
