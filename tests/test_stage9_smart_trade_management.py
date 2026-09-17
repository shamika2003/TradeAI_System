from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TradeAI"))

from core.trade_manager import TradeManager


class FakeExecutor:
    def __init__(self, position):
        self.positions = {position["symbol"]: dict(position)}
        self.closed = []
        self.modifications = []

    def get_all_positions(self):
        return list(self.positions.values())

    def get_position(self, symbol):
        return self.positions.get(symbol)

    def pip_size(self, symbol):
        return 0.0001

    def pip_value_per_lot(self, symbol, price=None):
        return 10.0

    def commission_for_lot(self, symbol, lot):
        return 7.0 * float(lot)

    def modify_position(self, symbol, stop_loss=None, take_profit=None):
        position = self.positions.get(symbol)
        if position is None:
            return False
        if stop_loss is not None:
            position["stop_loss"] = float(stop_loss)
        if take_profit is not None:
            position["take_profit"] = float(take_profit)
        self.modifications.append((symbol, stop_loss, take_profit))
        return True

    def close_position(self, symbol, reason="AI_CLOSE", candle_time=None):
        position = self.positions.pop(symbol, None)
        if position is None:
            return False
        self.closed.append({
            "symbol": symbol,
            "reason": reason,
            "candle_time": candle_time,
            "price": position["current_price"],
        })
        return True


def make_buy_position(current=1.1000, type_value="BUY"):
    return {
        "symbol": "EURUSD",
        "type": type_value,
        "entry_price": 1.1000,
        "current_price": current,
        "volume": 0.01,
        "stop_loss": 1.0990,
        "take_profit": 1.1020,
        "commission": 0.07,
        "open_time": None,
    }


def strong_sell_prediction():
    return {
        "signal": -0.30,
        "confidence": 0.68,
        "p_buy": 0.15,
        "p_hold": 0.17,
        "p_sell": 0.68,
        "signal_threshold": 0.10,
        "min_confidence": 0.45,
        "policy_enabled": True,
    }


def test_mt5_numeric_buy_direction_is_normalized(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ex = FakeExecutor(make_buy_position(type_value=0))
    manager = TradeManager(ex)

    assert manager.register_trade(ex.get_position("EURUSD")) is True
    assert manager.active_trades["EURUSD"]["direction"] == "BUY"


def test_break_even_is_r_based_and_covers_commission(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    position = make_buy_position(current=1.1012)  # +1.20R
    ex = FakeExecutor(position)
    manager = TradeManager(ex)
    assert manager.register_trade(position)

    assert manager.update("EURUSD", atr=0.0005) is True

    trade = manager.active_trades["EURUSD"]
    # 0.01 lot = $0.10/pip. $0.07 commission = 0.7 pip.
    # +$0.02 minimum positive-cash floor needs another 0.2 pip, so the
    # protected exit is 0.9 pip above entry after covering commission.
    assert trade["be_done"] is True
    assert trade["sl"] == pytest.approx(1.10009, abs=1e-10)
    assert trade["sl"] > trade["entry"]


def test_trailing_uses_r_trigger_and_never_goes_back_below_cost_be(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    position = make_buy_position(current=1.1024)  # +2.40R
    ex = FakeExecutor(position)
    manager = TradeManager(ex)
    assert manager.register_trade(position)

    assert manager.update("EURUSD", atr=0.0005) is True

    trade = manager.active_trades["EURUSD"]
    # Late trailing uses 1.25 ATR after the +0.75R profit lock:
    # 1.1024 - (1.25 * 0.0005) = 1.101775.
    assert trade["be_done"] is True
    assert trade["sl"] == pytest.approx(1.101775, abs=1e-10)


def test_profit_lock_preserves_meaningful_gain_before_runner_trail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    position = make_buy_position(current=1.1018)  # +1.80R
    ex = FakeExecutor(position)
    manager = TradeManager(ex)
    assert manager.register_trade(position)

    assert manager.update("EURUSD", atr=0.0005) is True

    trade = manager.active_trades["EURUSD"]
    assert trade["be_done"] is True
    # +0.75R lock on a 10-pip initial risk = +7.5 pips.
    assert trade["sl"] == pytest.approx(1.10075, abs=1e-10)


def test_strong_opposite_model_edge_cuts_meaningful_loser_early(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    position = make_buy_position(current=1.0998)  # -0.20R
    ex = FakeExecutor(position)
    manager = TradeManager(ex)
    assert manager.register_trade(position)

    # First completed bar: too early and not adverse enough.
    assert manager.update(
        "EURUSD",
        atr=0.0005,
        prediction=strong_sell_prediction(),
    ) is True
    assert ex.get_position("EURUSD") is not None

    # Second completed bar: -0.40R plus a calibrated strong SELL edge.
    ex.get_position("EURUSD")["current_price"] = 1.0996
    assert manager.update(
        "EURUSD",
        atr=0.0005,
        prediction=strong_sell_prediction(),
    ) is True

    assert ex.get_position("EURUSD") is None
    assert ex.closed[-1]["reason"] == "AI_DEFENSIVE_EXIT"


def test_weak_or_hold_model_does_not_force_loss_exit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    position = make_buy_position(current=1.0995)  # -0.50R
    ex = FakeExecutor(position)
    manager = TradeManager(ex)
    assert manager.register_trade(position)

    weak_prediction = {
        "signal": 0.0,
        "confidence": 0.35,
        "p_buy": 0.20,
        "p_hold": 0.45,
        "p_sell": 0.35,
        "signal_threshold": 0.10,
        "min_confidence": 0.45,
        "policy_enabled": True,
    }

    assert manager.update("EURUSD", atr=0.0005, prediction=weak_prediction) is True
    assert manager.update("EURUSD", atr=0.0005, prediction=weak_prediction) is True

    assert ex.get_position("EURUSD") is not None
    assert ex.closed == []
