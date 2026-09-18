from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TRAINING) not in sys.path:
    sys.path.insert(0, str(TRAINING))

import stage4_production_validate as s4
from shared.tradeai_core.validation_execution import ValidationExecutionContext


def _settings(tmp_path: Path, *, profile: dict | None = None):
    profile_path = tmp_path / "broker_profile.json"
    if profile is not None:
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
    return SimpleNamespace(
        BROKER_PROFILE_PATH=profile_path,
        USE_BROKER_PROFILE=profile is not None,
        DEFAULT_SPREAD_PIPS=1.2,
        SIMULATED_SLIPPAGE_PIPS=0.2,
        COMMISSION_PER_LOT=7.0,
        STRESS_SPREAD_PIPS=1.8,
        STRESS_SLIPPAGE_PIPS=0.5,
        STRESS_COMMISSION_PER_LOT=9.0,
        HISTORICAL_OHLC_SIDE="BID",
        MIN_LOT=0.001,
        MAX_LOT=1.0,
        LOT_STEP=0.001,
        RISK_PERCENT=0.30,
        MAX_ACTUAL_RISK_PERCENT=1.25,
        ATR_SL_MULTIPLIER=0.75,
        ATR_TP_MULTIPLIER=1.875,
        BREAK_EVEN_BUFFER_PIPS=0.15,
        BREAK_EVEN_MIN_PROFIT_MONEY=0.02,
        USE_BREAK_EVEN=True,
        BREAK_EVEN_TRIGGER_R=1.35,
        USE_PROFIT_LOCK=False,
        PROFIT_LOCK_TRIGGER_R=2.00,
        PROFIT_LOCK_R=1.00,
        USE_TRAILING_STOP=False,
        TRAILING_TRIGGER_R=2.40,
        TRAILING_ATR_MULTIPLIER=0.90,
        USE_AI_DEFENSIVE_EXIT=True,
        AI_DEFENSIVE_EXIT_MIN_BARS=2,
        AI_DEFENSIVE_EXIT_ADVERSE_R=0.25,
        AI_DEFENSIVE_EXIT_SIGNAL_MULTIPLIER=1.0,
        USE_MAX_HOLD=True,
        MAX_HOLD_BARS=48,
        MAX_OPEN_POSITIONS=2,
        MAX_PORTFOLIO_RISK_PERCENT=0.80,
        COOLDOWN_SECONDS=300,
        MAX_DAILY_LOSS_PERCENT=5.0,
        MAX_DRAWDOWN_PERCENT=12.0,
    )


def test_validation_net_risk_includes_bid_side_entry_and_commission(tmp_path):
    settings = _settings(tmp_path)
    ctx = ValidationExecutionContext(settings)
    costs = ctx.costs("EURUSD", "NORMAL")

    lot, info = ctx.net_risk_sized_lot(
        symbol="EURUSD",
        direction="BUY",
        requested_price=1.10000,
        stop_loss=1.09850,
        balance=150.0,
        risk_percent=0.30,
        costs=costs,
        max_actual_risk_percent=0.375,
    )

    assert lot == pytest.approx(0.002)
    assert info["entry_price"] == pytest.approx(1.10014)
    assert info["allowed_risk"] == pytest.approx(0.45)
    assert info["net_risk"] <= info["allowed_risk"] + 1e-12


def test_validation_uses_real_broker_volume_step(tmp_path):
    profile = {
        "version": 1,
        "source": "TEST",
        "symbols": {
            "EURUSD": {
                "digits": 5,
                "point": 0.00001,
                "trade_tick_size": 0.00001,
                "trade_tick_value": 1.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_stops_level": 0,
                "trade_freeze_level": 0,
                "spread_pips": 1.2,
                "slippage_pips": 0.2,
                "commission_per_lot": 7.0,
            }
        },
    }
    settings = _settings(tmp_path, profile=profile)
    ctx = ValidationExecutionContext(settings, require_profile=True)
    costs = ctx.costs("EURUSD", "NORMAL")

    lot, info = ctx.net_risk_sized_lot(
        symbol="EURUSD",
        direction="BUY",
        requested_price=1.10000,
        stop_loss=1.09850,
        balance=150.0,
        risk_percent=0.30,
        costs=costs,
        max_actual_risk_percent=0.375,
    )

    assert lot is None
    assert info["reason"] == "actual_risk_limit"


def test_sell_exit_prices_use_ask_side_for_bid_ohlc(tmp_path):
    settings = _settings(tmp_path)
    ctx = ValidationExecutionContext(settings)
    costs = ctx.costs("EURUSD", "NORMAL")
    high, low, close = ctx.sell_side_prices(
        "EURUSD",
        high=1.10000,
        low=1.09900,
        close=1.09950,
        costs=costs,
    )
    assert high == pytest.approx(1.10012)
    assert low == pytest.approx(1.09912)
    assert close == pytest.approx(1.09962)


def _path_frame():
    rows = []
    base = pd.Timestamp("2026-05-01 00:00:00")
    for i in range(30):
        close = 1.10000
        if i == 1:
            close = 1.09950
        rows.append({
            "time": base + pd.Timedelta(minutes=5 * i),
            "open": close,
            "high": close + 0.00020,
            "low": close - 0.00020,
            "close": close,
            "atr": 0.00100,
            "_mgmt_signal": -0.30 if i >= 1 else 0.0,
            "_mgmt_confidence": 0.80 if i >= 1 else 0.0,
            "_mgmt_min_confidence": 0.50,
            "_mgmt_signal_threshold": 0.10,
            "_mgmt_policy_enabled": True,
        })
    return pd.DataFrame(rows)


def test_stage4_path_uses_stage3_defensive_exit(tmp_path):
    settings = _settings(tmp_path)
    data = _path_frame()
    candidate = {
        "time": data.iloc[0]["time"],
        "symbol": "EURUSD",
        "index": 0,
        "direction": "BUY",
        "edge": 0.25,
        "confidence": 0.8,
    }
    trade, reason = s4._simulate_trade_path(
        data,
        0,
        candidate,
        150.0,
        s4.CostScenario("NORMAL", 1.2, 0.2, 7.0),
        settings,
    )
    assert reason is None
    assert trade is not None
    assert trade["exit_reason"] == "AI_DEFENSIVE_EXIT"
    assert abs(trade["profit"]) < trade["allowed_risk"]


def test_final_validation_targets_150_dollar_account():
    spec = importlib.util.spec_from_file_location(
        "tradeai_stage11_config",
        TRAINING / "config_model.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.FINAL_BACKTEST_BALANCE == pytest.approx(150.0)
    assert module.PRIMARY_ACCEPTANCE_BALANCES == [150.0]
    assert 150.0 in module.VALIDATION_BALANCES
