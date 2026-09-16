import copy

import pytest

from shared.tradeai_core.model_contract import attach_risk_policy, require_risk_policy
from shared.tradeai_core.risk_policy import RISK_POLICY_VERSION, validate_risk_policy


def _policy(risk=0.40):
    return {
        "version": RISK_POLICY_VERSION,
        "risk_percent": risk,
        "source": "unit_test",
    }


def test_valid_risk_policy_round_trip():
    policy = validate_risk_policy(_policy(0.40))
    assert policy["risk_percent"] == pytest.approx(0.40)


def test_invalid_risk_policy_rejected():
    with pytest.raises(RuntimeError):
        validate_risk_policy(_policy(0.0))
    with pytest.raises(RuntimeError):
        validate_risk_policy({"version": "wrong", "risk_percent": 0.4, "source": "x"})


def test_attach_and_require_risk_policy():
    artifact = {
        "contract": {"symbols": ["EURUSD"]},
        "models": {"EURUSD": object()},
    }
    promoted = attach_risk_policy(copy.deepcopy(artifact), _policy(0.35))
    loaded = require_risk_policy(promoted)
    assert loaded["risk_percent"] == pytest.approx(0.35)
