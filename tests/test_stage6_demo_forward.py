from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "TradeAI"))

from shared.tradeai_core.demo_forward import DemoForwardLedger
from core.risk_manager import RiskManager


def test_demo_forward_ledger_persists_and_gates(tmp_path):
    path = tmp_path / "forward.json"
    ledger = DemoForwardLedger(
        path,
        initial_balance=20.0,
        min_trades=3,
        max_dd_percent=12.0,
        min_profit_factor=1.2,
    )
    assert ledger.record_trade({"ticket": 1, "symbol": "EURUSD", "profit": 1.0, "open_time": "a", "close_time": "b"})
    assert ledger.record_trade({"ticket": 2, "symbol": "GBPUSD", "profit": -0.4, "open_time": "c", "close_time": "d"})
    assert ledger.record_trade({"ticket": 3, "symbol": "USDJPY", "profit": 1.0, "open_time": "e", "close_time": "f"})
    s = ledger.summary()
    assert s["closed_trades"] == 3
    assert s["virtual_balance"] == 21.6
    assert s["profit_factor"] == 5.0
    assert s["accepted"] is True
    reloaded = DemoForwardLedger(path, 20.0, 3, 12.0, 1.2)
    assert reloaded.summary()["closed_trades"] == 3


def test_duplicate_trade_is_not_counted_twice(tmp_path):
    path = tmp_path / "forward.json"
    ledger = DemoForwardLedger(path, 20.0, 1, 12.0, 1.2)
    trade = {"ticket": 7, "symbol": "EURUSD", "profit": 0.5, "open_time": "x", "close_time": "y"}
    assert ledger.record_trade(trade) is True
    assert ledger.record_trade(trade) is False
    assert ledger.summary()["closed_trades"] == 1
    assert ledger.summary()["virtual_balance"] == 20.5


class _StubExecutor:
    initial_capital = 20.0
    def get_balance(self): return 3000.0
    def get_equity(self): return 3000.0
    def get_risk_balance(self): return 20.0
    def get_risk_equity(self): return 20.0
    def get_risk_high_watermark(self): return 20.0
    def pip_size(self, symbol): return 0.0001
    def pip_value_per_lot(self, symbol, price=None): return 10.0
    def normalize_lot(self, symbol, lot): return 0.001
    def record_risk_rejection(self, reason): self.last_rejection = reason


def test_calibrated_risk_cannot_be_silently_rounded_far_up():
    ex = _StubExecutor()
    rm = RiskManager(executor=ex, risk_percent=0.30)
    # 10 pip stop, 0.001 lot = $0.10 = 0.50% of $20.
    # Stage 6 allows only 25% tolerance above 0.30% => 0.375%.
    lot = rm.calculate_lot("EURUSD", 10 * 0.0001, price=1.10)
    assert lot is None
    assert ex.last_rejection == "MIN_LOT_TOO_RISKY"
