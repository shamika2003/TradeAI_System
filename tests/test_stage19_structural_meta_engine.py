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

from config_model import (
    DEPLOYMENT_CALIBRATION_END_FRACTION,
    POLICY_SELECTION_END_FRACTION,
    QUALIFICATION_SCORE_CAL_END_FRACTION,
    QUALIFICATION_TRAIN_FRACTION,
)
from opportunity_validate import _block_stability
from shared.tradeai_core.decision_policy import DECISION_POLICY_VERSION, validate_decision_policy
from shared.tradeai_core.setup_engine import add_structural_setup_features
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, SELL_CLASS
from training_utils import OpportunityPredictions, choose_opportunities


def _strong_trend_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ema21_55_atr": [1.2], "ema8_21_atr": [0.8],
        "trend_slope_12_atr": [0.7], "ema21_slope_6_atr": [0.5],
        "h1_ema21_55_atr": [1.1], "h1_trend_slope_12_atr": [0.5],
        "momentum_3_atr": [0.6], "momentum_12_atr": [0.9],
        "momentum_24_atr": [1.1], "h1_momentum_12_atr": [0.8],
        "di_spread14": [0.5], "adx14": [0.45], "h1_adx14": [0.40],
        "trend_efficiency_12": [0.75], "trend_efficiency_48": [0.70],
        "range_expansion_20": [1.4], "volume_z20": [0.7],
        "breakout_up_20_atr": [0.8], "breakout_down_20_atr": [0.0],
        "breakout_up_50_atr": [0.5], "breakout_down_50_atr": [0.0],
        "range_position_20": [0.9], "range_position_50": [0.85],
        "close_location": [0.85], "signed_body_atr": [0.5],
        "wick_imbalance": [0.3], "rsi14": [62.0], "rsi_slope_3": [0.05],
        "ema8_gap_atr": [0.2], "volatility_ratio": [1.05],
        "volatility_term_5_60": [1.1], "atr_rank_200": [0.6],
        "spread_relative_100": [0.9],
    })


def test_structural_setup_scores_are_bounded_and_directional():
    out = add_structural_setup_features(_strong_trend_frame())
    for name in (
        "setup_trend_buy", "setup_trend_sell", "setup_regime_quality",
        "setup_buy_score", "setup_sell_score",
    ):
        assert 0.0 <= float(out.loc[0, name]) <= 1.0
    assert float(out.loc[0, "setup_buy_score"]) > float(out.loc[0, "setup_sell_score"])
    assert float(out.loc[0, "setup_score_gap"]) > 0.0


def test_selector_requires_structural_setup_before_model_edge():
    pred = OpportunityPredictions(
        buy_tp_probability=np.array([0.80, 0.80]),
        sell_tp_probability=np.array([0.20, 0.20]),
        buy_expected_r=np.array([0.80, 0.80]),
        sell_expected_r=np.array([-0.10, -0.10]),
    )
    setup = pd.DataFrame({
        "setup_buy_score": [0.45, 0.78],
        "setup_sell_score": [0.20, 0.20],
        "setup_regime_quality": [0.8, 0.8],
    })
    actions, *_ = choose_opportunities(
        pred,
        min_tp_probability=0.55,
        min_expected_r=0.10,
        min_ev_gap=0.05,
        min_probability_gap=0.02,
        setup_frame=setup,
        min_setup_score=0.60,
        min_setup_gap=0.05,
    )
    assert actions.tolist() == [HOLD_CLASS, BUY_CLASS]


def test_policy_can_enable_only_one_direction():
    pred = OpportunityPredictions(
        buy_tp_probability=np.array([0.20, 0.78]),
        sell_tp_probability=np.array([0.82, 0.79]),
        buy_expected_r=np.array([-0.10, 0.70]),
        sell_expected_r=np.array([0.75, 0.72]),
    )
    setup = pd.DataFrame({
        "setup_buy_score": [0.15, 0.80],
        "setup_sell_score": [0.82, 0.15],
        "setup_regime_quality": [0.9, 0.9],
    })
    actions, *_ = choose_opportunities(
        pred,
        min_tp_probability=0.55,
        min_expected_r=0.10,
        min_ev_gap=0.05,
        min_probability_gap=0.02,
        setup_frame=setup,
        min_setup_score=0.60,
        min_setup_gap=0.05,
        allow_buy=True,
        allow_sell=False,
    )
    assert actions.tolist() == [HOLD_CLASS, BUY_CLASS]


def test_sparse_stability_ignores_zero_trade_blocks(monkeypatch):
    import opportunity_validate as ov

    sequence = iter([
        {"trades": 0, "avg_r": 0.0, "profit_factor_r": 0.0},
        {"trades": 3, "avg_r": 0.25, "profit_factor_r": 1.4},
        {"trades": 0, "avg_r": 0.0, "profit_factor_r": 0.0},
        {"trades": 2, "avg_r": 0.12, "profit_factor_r": 1.2},
        {"trades": 0, "avg_r": 0.0, "profit_factor_r": 0.0},
        {"trades": 4, "avg_r": 0.18, "profit_factor_r": 1.3},
        {"trades": 0, "avg_r": 0.0, "profit_factor_r": 0.0},
        {"trades": 1, "avg_r": -0.4, "profit_factor_r": 0.5},
    ])
    monkeypatch.setattr(ov, "evaluate_opportunities", lambda *a, **k: next(sequence))

    data = pd.DataFrame({"x": np.arange(16)})
    pred = OpportunityPredictions(
        np.full(16, 0.6), np.full(16, 0.4), np.full(16, 0.3), np.full(16, 0.1)
    )
    policy = {
        "min_tp_probability": 0.55, "min_expected_r": 0.1,
        "min_ev_gap": 0.05, "min_probability_gap": 0.02,
        "min_setup_score": 0.5, "min_setup_gap": 0.03,
        "allow_buy": True, "allow_sell": True,
    }
    result = _block_stability(data, pred, policy, blocks=8)
    assert result["active_blocks"] == 4
    assert result["positive_blocks"] == 3
    assert result["stable"] is True


def test_structural_policy_contract_preserves_direction_and_setup_gates():
    raw = {
        "version": DECISION_POLICY_VERSION,
        "symbols": {
            "EURUSD": {
                "enabled": True,
                "allow_buy": True,
                "allow_sell": False,
                "min_setup_score": 0.62,
                "min_setup_gap": 0.08,
                "min_tp_probability": 0.58,
                "min_expected_r": 0.20,
                "min_ev_gap": 0.08,
                "min_probability_gap": 0.03,
            }
        },
    }
    p = validate_decision_policy(raw, ["EURUSD"])["symbols"]["EURUSD"]
    assert p["enabled"] is True
    assert p["allow_buy"] is True and p["allow_sell"] is False
    assert p["min_setup_score"] == pytest.approx(0.62)
    assert p["min_setup_gap"] == pytest.approx(0.08)


def test_stage11_chronological_zones_are_strictly_separated():
    assert 0.0 < QUALIFICATION_TRAIN_FRACTION < QUALIFICATION_SCORE_CAL_END_FRACTION
    assert QUALIFICATION_SCORE_CAL_END_FRACTION < POLICY_SELECTION_END_FRACTION
    assert POLICY_SELECTION_END_FRACTION < DEPLOYMENT_CALIBRATION_END_FRACTION < 1.0


def test_structural_collapse_can_cut_an_adverse_trade_only_with_model_weakening():
    from shared.tradeai_core.management_policy import opportunity_thesis_exit

    degraded = {
        "policy_enabled": True,
        "p_buy": 0.48, "p_sell": 0.50,
        "buy_expected_r": 0.08, "sell_expected_r": 0.10,
        "min_tp_probability": 0.58, "min_expected_r": 0.20,
        "setup_buy_score": 0.30, "setup_sell_score": 0.46,
        "min_setup_score": 0.60, "min_setup_gap": 0.05,
    }
    should, reason = opportunity_thesis_exit(
        direction="BUY", r_multiple=-0.18, bars_held=4, prediction=degraded,
        min_bars=3, adverse_r=0.12, stale_bars=18, stale_max_r=0.20,
        own_probability_fraction=0.70, own_ev_floor=-0.05,
        opposite_ev_margin=0.15, opposite_probability_margin=0.05,
        setup_fraction=0.70,
    )
    assert should is True
    assert reason == "STRUCTURAL_FAILURE"

    healthy_model = dict(degraded, p_buy=0.66, buy_expected_r=0.45)
    should, _ = opportunity_thesis_exit(
        direction="BUY", r_multiple=-0.18, bars_held=4, prediction=healthy_model,
        min_bars=3, adverse_r=0.12, stale_bars=18, stale_max_r=0.20,
        own_probability_fraction=0.70, own_ev_floor=-0.05,
        opposite_ev_margin=0.15, opposite_probability_margin=0.05,
        setup_fraction=0.70,
    )
    assert should is False
