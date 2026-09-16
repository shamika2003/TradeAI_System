from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT)
)

sys.path.insert(
    0,
    str(
        ROOT /
        "TradeAI"
    )
)

from core.risk_manager import RiskManager


class StubExecutor:

    def __init__(
            self,
            balance,
            min_lot=0.01
    ):

        self.balance = float(
            balance
        )

        self.equity = float(
            balance
        )

        self.initial_capital = float(
            balance
        )

        self.min_lot = float(
            min_lot
        )

        self.rejection = None


    def get_balance(self):
        return self.balance


    def get_equity(self):
        return self.equity


    def get_risk_balance(self):
        return self.balance


    def get_risk_equity(self):
        return self.equity


    def get_risk_high_watermark(self):
        return self.balance


    def pip_size(
            self,
            symbol
    ):
        return 0.0001


    def pip_value_per_lot(
            self,
            symbol,
            price=None
    ):
        return 10.0


    def get_lot_limits(
            self,
            symbol
    ):

        return {
            "min": self.min_lot,
            "max": 100.0,
            "step": self.min_lot
        }


    def normalize_lot(
            self,
            symbol,
            lot
    ):

        lot = float(
            lot
        )

        if lot < self.min_lot:
            return None

        return self.min_lot


    def record_risk_rejection(
            self,
            reason
    ):
        self.rejection = reason


def test_safe_broker_minimum_is_not_thrown_away():

    ex = StubExecutor(
        50.0
    )

    rm = RiskManager(
        executor=ex,
        risk_percent=0.30
    )

    # 1.5 pip stop:
    # 0.01 lot * $10/pip/lot * 1.5 pips
    # = $0.15 = exactly 0.30% of $50.
    lot = rm.calculate_lot(
        "EURUSD",
        1.5 * 0.0001,
        price=1.10
    )

    assert lot == 0.01
    assert ex.rejection is None


def test_unsafe_broker_minimum_is_still_rejected():

    ex = StubExecutor(
        50.0
    )

    rm = RiskManager(
        executor=ex,
        risk_percent=0.30
    )

    # 10 pip stop:
    # 0.01 lot * $10/pip/lot * 10 pips
    # = $1.00 = 2.0% of $50.
    # This must remain rejected.
    lot = rm.calculate_lot(
        "EURUSD",
        10.0 * 0.0001,
        price=1.10
    )

    assert lot is None
    assert ex.rejection == "MIN_LOT_TOO_RISKY"
