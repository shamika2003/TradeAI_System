from __future__ import annotations

DECISION_POLICY_VERSION = "tradeai_decision_policy_v1_20260907"


def validate_symbol_policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise RuntimeError("Symbol decision policy must be a dictionary")

    enabled = bool(policy.get("enabled", True))
    min_confidence = float(policy.get("min_confidence", 0.45))
    signal_threshold = float(policy.get("signal_threshold", 0.10))

    if not (0.0 <= min_confidence <= 1.0):
        raise RuntimeError("Decision min_confidence must be between 0 and 1")
    if not (0.0 <= signal_threshold <= 1.0):
        raise RuntimeError("Decision signal_threshold must be between 0 and 1")

    return {
        **policy,
        "enabled": enabled,
        "min_confidence": min_confidence,
        "signal_threshold": signal_threshold,
    }


def validate_decision_policy(policy: dict, symbols) -> dict:
    if not isinstance(policy, dict):
        raise RuntimeError("Decision policy missing")
    if policy.get("version") != DECISION_POLICY_VERSION:
        raise RuntimeError("Unsupported decision policy version")

    symbol_map = policy.get("symbols")
    if not isinstance(symbol_map, dict):
        raise RuntimeError("Decision policy symbol map missing")

    expected = set(symbols)
    if set(symbol_map) != expected:
        raise RuntimeError(
            f"Decision policy symbols mismatch: expected {sorted(expected)}, got {sorted(symbol_map)}"
        )

    normalized = {
        symbol: validate_symbol_policy(symbol_map[symbol])
        for symbol in symbols
    }
    return {
        **policy,
        "symbols": normalized,
    }
