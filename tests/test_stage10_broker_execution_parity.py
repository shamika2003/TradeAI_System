from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TradeAI"))

import config.settings as settings
from execution.paper_executor import PaperExecutor
from shared.tradeai_core.broker_profile import save_broker_profile


def _write_profile(path: Path):
    payload = {
        "version": 1,
        "source": "TEST",
        "captured_at_utc": "2026-09-17T00:00:00+00:00",
        "account_currency": "USD",
        "symbols": {
            "EURUSD": {
                "digits": 5,
                "point": 0.00001,
                "trade_tick_size": 0.00001,
                "trade_tick_value": 1.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_stops_level": 20,
                "trade_freeze_level": 0,
                "spread_pips": 1.4,
                "slippage_pips": 0.2,
                "commission_per_lot": 7.0,
            },
            "USDJPY": {
                "digits": 3,
                "point": 0.001,
                "trade_tick_size": 0.001,
                "trade_tick_value": 0.623,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
                "trade_stops_level": 0,
                "trade_freeze_level": 0,
                "spread_pips": 1.2,
                "slippage_pips": 0.2,
                "commission_per_lot": 7.0,
            },
        },
    }
    save_broker_profile(path, payload)


def _executor_with_profile(tmp_path, monkeypatch):
    profile_path = tmp_path / "broker_profile.json"
    _write_profile(profile_path)
    monkeypatch.setattr(settings, "BROKER_PROFILE_PATH", profile_path)
    monkeypatch.setattr(settings, "USE_BROKER_PROFILE", True)
    return PaperExecutor(capital=150.0)


def test_paper_uses_captured_broker_volume_rules(tmp_path, monkeypatch):
    ex = _executor_with_profile(tmp_path, monkeypatch)

    limits = ex.get_lot_limits("EURUSD")
    assert limits == {"min": 0.01, "max": 100.0, "step": 0.01}

    assert ex.normalize_lot("EURUSD", 0.019) == 0.01
    assert ex.normalize_lot("EURUSD", 0.009) is None


def test_bid_side_historical_entry_matches_mt5_buy_sell_semantics(tmp_path, monkeypatch):
    ex = _executor_with_profile(tmp_path, monkeypatch)

    buy = ex.estimate_entry_price("EURUSD", "BUY", 1.10000)
    sell = ex.estimate_entry_price("EURUSD", "SELL", 1.10000)

    # BID bar + 1.4 pip spread + 0.2 pip adverse slippage.
    assert buy == pytest.approx(1.10016, abs=1e-12)
    # SELL executes on BID, then 0.2 pip adverse slippage.
    assert sell == pytest.approx(1.09998, abs=1e-12)


def test_sell_stop_uses_ask_side_not_raw_bid_candle(tmp_path, monkeypatch):
    ex = _executor_with_profile(tmp_path, monkeypatch)

    assert ex.open_trade(
        symbol="EURUSD",
        direction="SELL",
        price=1.10000,
        lot=0.01,
        stop_loss=1.10050,
        take_profit=1.09850,
    ) is True

    # Raw BID high stays below SL, but ASK high = 1.10038 + 0.00014
    # = 1.10052, so a real SELL position's stop is touched.
    closed = ex.update_candle(
        "EURUSD",
        {
            "open": 1.10010,
            "high": 1.10038,
            "low": 1.09980,
            "close": 1.10020,
        },
    )

    assert closed is True
    assert ex.trade_history[-1]["exit_reason"] == "STOP_LOSS"


def test_paper_enforces_broker_minimum_stop_distance(tmp_path, monkeypatch):
    ex = _executor_with_profile(tmp_path, monkeypatch)

    # Profile requires 20 points = 0.00020 minimum distance.
    # BUY fill is 1.10016, so SL 1.10005 is only 0.00011 away.
    assert ex.open_trade(
        symbol="EURUSD",
        direction="BUY",
        price=1.10000,
        lot=0.01,
        stop_loss=1.10005,
        take_profit=1.10100,
    ) is False


def test_profile_commission_and_tick_value_feed_net_risk(tmp_path, monkeypatch):
    ex = _executor_with_profile(tmp_path, monkeypatch)

    estimate = ex.estimate_stop_loss_risk(
        "EURUSD",
        "BUY",
        1.10000,
        1.09900,
        0.01,
    )

    assert estimate["commission"] == pytest.approx(0.07)
    assert estimate["pip_value_per_lot"] == pytest.approx(10.0)
    assert estimate["net_risk"] > estimate["price_risk"]
