from __future__ import annotations

from typing import Any


RISK_POLICY_VERSION = "tradeai_risk_policy_v1_20260907"


def validate_risk_policy(policy: Any) -> dict:
    if not isinstance(policy, dict):
        raise RuntimeError("Production risk policy is missing")

    if policy.get("version") != RISK_POLICY_VERSION:
        raise RuntimeError(
            f"Unsupported production risk policy version: {policy.get('version')!r}"
        )

    try:
        risk_percent = float(policy["risk_percent"])
    except Exception as exc:
        raise RuntimeError("Production risk policy missing risk_percent") from exc

    if not (0.0 < risk_percent <= 2.0):
        raise RuntimeError(f"Invalid calibrated risk_percent: {risk_percent}")

    source = str(policy.get("source", ""))
    if not source:
        raise RuntimeError("Production risk policy missing source")

    normalized = dict(policy)
    normalized["risk_percent"] = risk_percent
    normalized["version"] = RISK_POLICY_VERSION
    normalized["source"] = source
    return normalized
