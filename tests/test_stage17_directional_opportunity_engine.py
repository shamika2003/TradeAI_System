from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
TRADEAI = ROOT / "TradeAI"
for p in (ROOT, TRAINING, TRADEAI):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from training_utils import OpportunityPredictions, choose_opportunities
from shared.tradeai_core.decision_policy import DECISION_POLICY_VERSION, validate_decision_policy
from shared.tradeai_core.management_policy import opportunity_thesis_exit
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, SELL_CLASS, PREDICTION_TYPE
from config import settings


def test_model_objective_is_directional_opportunity_not_multiclass():
    assert PREDICTION_TYPE == "directional_quality_barrier_plus_expected_r"


def test_opportunity_selector_can_hold_for_weak_or_conflicted_setups():
    pred = OpportunityPredictions(
        buy_tp_probability=np.array([0.70, 0.70, 0.52]),
        sell_tp_probability=np.array([0.20, 0.66, 0.20]),
        buy_expected_r=np.array([0.60, 0.60, 0.10]),
        sell_expected_r=np.array([-0.10, 0.55, 0.00]),
    )
    actions, *_ = choose_opportunities(
        pred,
        min_tp_probability=0.58,
        min_expected_r=0.20,
        min_ev_gap=0.10,
        min_probability_gap=0.03,
    )
    assert actions.tolist() == [BUY_CLASS, HOLD_CLASS, HOLD_CLASS]


def test_opportunity_selector_supports_clean_sell():
    pred = OpportunityPredictions(
        buy_tp_probability=np.array([0.20]),
        sell_tp_probability=np.array([0.72]),
        buy_expected_r=np.array([-0.10]),
        sell_expected_r=np.array([0.65]),
    )
    actions, *_ = choose_opportunities(
        pred,
        min_tp_probability=0.58,
        min_expected_r=0.20,
        min_ev_gap=0.10,
        min_probability_gap=0.03,
    )
    assert actions.tolist() == [SELL_CLASS]


def test_policy_contract_preserves_quality_thresholds():
    policy = {
        "version": DECISION_POLICY_VERSION,
        "symbols": {
            "EURUSD": {
                "enabled": True,
                "min_tp_probability": 0.60,
                "min_expected_r": 0.25,
                "min_ev_gap": 0.15,
                "min_probability_gap": 0.04,
            }
        },
    }
    out = validate_decision_policy(policy, ["EURUSD"])
    p = out["symbols"]["EURUSD"]
    assert p["min_tp_probability"] == pytest.approx(0.60)
    assert p["min_expected_r"] == pytest.approx(0.25)
    assert p["min_ev_gap"] == pytest.approx(0.15)


def test_thesis_exit_cuts_degraded_loser_but_not_healthy_trade():
    bad = {
        "policy_enabled": True,
        "p_buy": 0.24,
        "p_sell": 0.62,
        "buy_expected_r": -0.18,
        "sell_expected_r": 0.42,
        "min_tp_probability": 0.58,
        "min_expected_r": 0.20,
    }
    should, reason = opportunity_thesis_exit(
        direction="BUY", r_multiple=-0.20, bars_held=4, prediction=bad,
        min_bars=3, adverse_r=0.12, stale_bars=18, stale_max_r=0.20,
        own_probability_fraction=0.70, own_ev_floor=-0.05,
        opposite_ev_margin=0.15, opposite_probability_margin=0.05,
    )
    assert should is True
    assert reason == "THESIS_FAILURE"

    healthy = dict(bad, p_buy=0.64, p_sell=0.30, buy_expected_r=0.38, sell_expected_r=-0.05)
    should, _ = opportunity_thesis_exit(
        direction="BUY", r_multiple=-0.20, bars_held=4, prediction=healthy,
        min_bars=3, adverse_r=0.12, stale_bars=18, stale_max_r=0.20,
        own_probability_fraction=0.70, own_ev_floor=-0.05,
        opposite_ev_margin=0.15, opposite_probability_margin=0.05,
    )
    assert should is False


def test_profit_management_moves_stop_only_after_trade_proves_itself():
    assert settings.USE_BREAK_EVEN is True
    assert settings.BREAK_EVEN_TRIGGER_R >= 1.0
    assert settings.USE_PROFIT_LOCK is True
    assert settings.PROFIT_LOCK_TRIGGER_R > settings.BREAK_EVEN_TRIGGER_R
    assert settings.PROFIT_LOCK_R > 0
