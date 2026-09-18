from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
for path in (ROOT, TRAINING):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import config_model as cfg
from training_utils import actions_from_probabilities, compute_weights
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, SELL_CLASS


class DummyModel:
    classes_ = np.array([SELL_CLASS, HOLD_CLASS, BUY_CLASS], dtype=np.int32)


def test_hold_margin_blocks_epsilon_directional_wins():
    proba = np.array([
        [0.20, 0.39, 0.41],
        [0.20, 0.30, 0.50],
    ])
    actions, *_ = actions_from_probabilities(
        DummyModel(),
        proba,
        min_confidence=0.40,
        signal_threshold=0.10,
        hold_margin=0.05,
    )
    assert actions.tolist() == [HOLD_CLASS, BUY_CLASS]


def test_selective_class_weights_do_not_force_directional_frequency():
    y = np.array([HOLD_CLASS] * 90 + [BUY_CLASS] * 5 + [SELL_CLASS] * 5, dtype=np.int32)
    weights = compute_weights(y, np.ones(len(y)))
    assert np.all(weights[:90] == 1.0)
    assert weights.max() <= 2.5
    assert weights[-1] > 1.0


def test_risk_selection_is_not_allowed_to_force_high_risk_for_trade_count():
    assert max(cfg.STAGE5_RISK_GRID) <= 0.45
    assert cfg.STAGE5_CAL_MIN_TRADES == 0
    assert cfg.CALIBRATION_MAX_COVERAGE <= 0.08
