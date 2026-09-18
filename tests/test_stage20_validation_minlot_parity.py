from types import SimpleNamespace

from shared.tradeai_core.validation_execution import (
    ValidationExecutionContext,
    ValidationExecutionCosts,
)


class _FakeValidationExecution(ValidationExecutionContext):
    def __init__(self):
        self.settings = SimpleNamespace(HISTORICAL_OHLC_SIDE="BID")
        self.profile = None

    def entry_price(self, symbol, direction, requested_bid, costs):
        return float(requested_bid)

    def normalize_price(self, symbol, price):
        return float(price)

    def pip_value_per_lot(self, symbol, price):
        return 10.0

    def lot_limits(self, symbol):
        return 0.01, 100.0, 0.01


def test_validation_uses_safe_broker_minimum_lot_fallback_like_runtime():
    execution = _FakeValidationExecution()
    costs = ValidationExecutionCosts(
        spread_pips=0.0,
        slippage_pips=0.0,
        commission_per_lot=0.0,
    )

    # 5 pips * $10/pip/lot = $50 risk per lot.
    # $150 * 0.30% = $0.45 -> raw lot 0.009, below broker min 0.01.
    # Broker min risks $0.50 = 0.333%, which is still below a 0.375% hard cap.
    lot, info = execution.net_risk_sized_lot(
        symbol="EURUSD",
        direction="BUY",
        requested_price=1.1000,
        stop_loss=1.0995,
        balance=150.0,
        risk_percent=0.30,
        costs=costs,
        max_actual_risk_percent=0.375,
    )

    assert lot == 0.01
    assert info["used_minimum_fallback"] is True
    assert info["raw_lot"] < 0.01
    assert info["actual_risk_percent"] <= 0.375


def test_validation_minimum_lot_fallback_still_obeys_hard_actual_risk_cap():
    execution = _FakeValidationExecution()
    costs = ValidationExecutionCosts(0.0, 0.0, 0.0)

    lot, info = execution.net_risk_sized_lot(
        symbol="EURUSD",
        direction="BUY",
        requested_price=1.1000,
        stop_loss=1.0995,
        balance=150.0,
        risk_percent=0.30,
        costs=costs,
        max_actual_risk_percent=0.30,
    )

    assert lot is None
    assert info["used_minimum_fallback"] is True
    assert info["reason"] == "actual_risk_limit"
