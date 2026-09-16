from __future__ import annotations

import math

from shared.tradeai_core.decision_policy import (
    DECISION_POLICY_VERSION,
    validate_decision_policy,
)
from shared.tradeai_core.production_economics import (
    fallback_pip_value_per_lot,
    normalize_lot,
    risk_sized_lot,
)


def test_usd_base_pip_values_are_price_converted():
    assert math.isclose(fallback_pip_value_per_lot("EURUSD", 1.10), 10.0, rel_tol=1e-9)
    assert math.isclose(fallback_pip_value_per_lot("USDJPY", 150.0), 1000.0 / 150.0, rel_tol=1e-9)
    assert math.isclose(fallback_pip_value_per_lot("USDCNH", 7.10), 10.0 / 7.10, rel_tol=1e-9)


def test_low_balance_lot_is_never_rounded_up():
    assert normalize_lot(0.0009, 0.001, 1.0, 0.001) is None
    assert normalize_lot(0.0029, 0.001, 1.0, 0.001) == 0.002

    lot, info = risk_sized_lot(
        balance=10.0,
        risk_percent=1.0,
        stop_distance=0.0020,  # 20 pips EURUSD
        symbol="EURUSD",
        price=1.10,
        minimum=0.001,
        maximum=1.0,
        step=0.001,
        max_actual_risk_percent=1.25,
    )
    assert lot is None
    assert info["reason"] == "below_minimum_or_invalid_lot"


def test_decision_policy_requires_every_symbol():
    symbols = ["EURUSD", "GBPUSD"]
    policy = {
        "version": DECISION_POLICY_VERSION,
        "symbols": {
            "EURUSD": {"enabled": True, "min_confidence": 0.5, "signal_threshold": 0.1},
            "GBPUSD": {"enabled": False, "min_confidence": 0.99, "signal_threshold": 0.99},
        },
    }
    checked = validate_decision_policy(policy, symbols)
    assert checked["symbols"]["EURUSD"]["enabled"] is True
    assert checked["symbols"]["GBPUSD"]["enabled"] is False
