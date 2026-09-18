from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
TRADEAI = ROOT / "TradeAI"
for path in (ROOT, TRAINING, TRADEAI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stage4_production_validate as s4
from config import settings
from shared.tradeai_core import target_definition as target


def test_runtime_geometry_matches_selective_two_and_half_r_target_contract():
    assert settings.ATR_SL_MULTIPLIER == pytest.approx(target.STOP_ATR_MULTIPLIER)
    assert settings.ATR_TP_MULTIPLIER == pytest.approx(target.TAKE_ATR_MULTIPLIER)
    assert settings.MAX_HOLD_BARS == target.MAX_HOLD_BARS
    assert settings.ATR_TP_MULTIPLIER / settings.ATR_SL_MULTIPLIER == pytest.approx(2.5)
    assert settings.MAX_OPEN_POSITIONS == 2
    assert settings.MAX_PORTFOLIO_RISK_PERCENT == pytest.approx(0.80)
    assert settings.USE_PROFIT_LOCK is True
    assert settings.PROFIT_LOCK_TRIGGER_R == pytest.approx(1.75)
    assert settings.PROFIT_LOCK_R == pytest.approx(0.55)
    assert settings.USE_TRAILING_STOP is False
    assert settings.BREAK_EVEN_TRIGGER_R > 1.0


def test_portfolio_backtest_allows_two_symbols_but_blocks_third(monkeypatch):
    base = pd.Timestamp("2026-05-01 10:00:00")
    candidates = [
        {"time": base, "symbol": "EURUSD", "index": 0, "direction": "BUY", "edge": .3, "confidence": .7},
        {"time": base, "symbol": "GBPUSD", "index": 0, "direction": "BUY", "edge": .3, "confidence": .7},
        {"time": base, "symbol": "USDJPY", "index": 0, "direction": "BUY", "edge": .3, "confidence": .7},
    ]
    dummy_data = {s: pd.DataFrame({"time": [base]}) for s in ("EURUSD", "GBPUSD", "USDJPY")}

    def fake_path(data, idx, candidate, balance, scenario, settings_obj, execution=None):
        return {
            "symbol": candidate["symbol"],
            "direction": candidate["direction"],
            "open_time": candidate["time"],
            "close_time": candidate["time"] + pd.Timedelta(minutes=30),
            "profit": 0.50,
            "commission": 0.07,
            "entry_cost_money": 0.01,
            "planned_net_risk": 0.30,
            "worst_equity": balance - 0.30,
        }, None

    monkeypatch.setattr(s4, "_simulate_trade_path", fake_path)
    cfg = SimpleNamespace(
        MAX_OPEN_POSITIONS=2,
        MAX_PORTFOLIO_RISK_PERCENT=0.90,
        COOLDOWN_SECONDS=0,
        MAX_DAILY_LOSS_PERCENT=100.0,
        MAX_DRAWDOWN_PERCENT=100.0,
    )
    result = s4._portfolio_backtest(
        candidates,
        dummy_data,
        150.0,
        s4.CostScenario("NORMAL", 0.2, 0.2, 7.0),
        cfg,
        execution=object(),
    )
    assert result["trades"] == 2
    assert result["rejected"]["position_busy"] == 1
    assert result["max_concurrent_positions"] == 2


def test_policy_calibration_constants_are_money_first():
    import config_model as cfg

    assert cfg.POLICY_CALIBRATION_BALANCE == pytest.approx(150.0)
    assert cfg.POLICY_CALIBRATION_RISK_PERCENT == pytest.approx(0.45)
    assert cfg.POLICY_CALIBRATION_NORMAL_MIN_PROFIT_FACTOR > 1.0
    assert cfg.POLICY_CALIBRATION_STRESS_MIN_PROFIT_FACTOR > 1.0
    assert cfg.POLICY_CALIBRATION_MIN_PAYOFF_RATIO >= 1.20
