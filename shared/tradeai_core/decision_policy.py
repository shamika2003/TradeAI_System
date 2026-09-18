from __future__ import annotations

SIGNAL_EPISODE_RESET_BARS = 3

DECISION_POLICY_VERSION = "tradeai_structural_meta_policy_v3_20260918"


def validate_symbol_policy(policy: dict) -> dict:
    if not isinstance(policy, dict):
        raise RuntimeError("Symbol decision policy must be a dictionary")

    enabled = bool(policy.get("enabled", True))
    min_tp_probability = float(policy.get("min_tp_probability", policy.get("min_confidence", 0.55)))
    min_expected_r = float(policy.get("min_expected_r", 0.10))
    min_ev_gap = float(policy.get("min_ev_gap", policy.get("signal_threshold", 0.10)))
    min_probability_gap = float(policy.get("min_probability_gap", 0.02))
    min_setup_score = float(policy.get("min_setup_score", 0.0))
    min_setup_gap = float(policy.get("min_setup_gap", 0.0))
    allow_buy = bool(policy.get("allow_buy", True))
    allow_sell = bool(policy.get("allow_sell", True))

    if not (allow_buy or allow_sell):
        enabled = False

    if not (0.0 <= min_tp_probability <= 1.0):
        raise RuntimeError("Decision min_tp_probability must be between 0 and 1")
    if not (-1.0 <= min_expected_r <= 5.0):
        raise RuntimeError("Decision min_expected_r is outside the supported range")
    if not (0.0 <= min_ev_gap <= 5.0):
        raise RuntimeError("Decision min_ev_gap must be nonnegative")
    if not (0.0 <= min_probability_gap <= 1.0):
        raise RuntimeError("Decision min_probability_gap must be between 0 and 1")
    if not (0.0 <= min_setup_score <= 1.0):
        raise RuntimeError("Decision min_setup_score must be between 0 and 1")
    if not (0.0 <= min_setup_gap <= 1.0):
        raise RuntimeError("Decision min_setup_gap must be between 0 and 1")

    # Compatibility aliases are retained because existing runtime telemetry and
    # the defensive manager still consume confidence/signal threshold fields.
    return {
        **policy,
        "enabled": enabled,
        "min_tp_probability": min_tp_probability,
        "min_expected_r": min_expected_r,
        "min_ev_gap": min_ev_gap,
        "min_probability_gap": min_probability_gap,
        "min_setup_score": min_setup_score,
        "min_setup_gap": min_setup_gap,
        "allow_buy": allow_buy,
        "allow_sell": allow_sell,
        "min_confidence": min_tp_probability,
        "signal_threshold": min_ev_gap,
        "hold_margin": 0.0,
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

    normalized = {symbol: validate_symbol_policy(symbol_map[symbol]) for symbol in symbols}
    return {**policy, "symbols": normalized}
