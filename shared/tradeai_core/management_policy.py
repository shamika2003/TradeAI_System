from __future__ import annotations

import math


def _num(mapping, key, default=0.0):
    try:
        value = float(mapping.get(key, default))
        return value if math.isfinite(value) else float(default)
    except Exception:
        return float(default)


def opportunity_thesis_exit(
    *,
    direction: str,
    r_multiple: float,
    bars_held: int,
    prediction: dict | None,
    min_bars: int,
    adverse_r: float,
    stale_bars: int,
    stale_max_r: float,
    own_probability_fraction: float,
    own_ev_floor: float,
    opposite_ev_margin: float,
    opposite_probability_margin: float,
    setup_fraction: float = 0.70,
) -> tuple[bool, str]:
    """Pure causal exit rule shared by runtime and historical validation.

    The manager never exits just because a bar is red. It needs evidence that
    the original directional thesis has degraded: weak own TP probability / EV,
    or a materially stronger opposite opportunity. A stale low-progress trade
    may also be released to free capital after enough closed bars.
    """
    if not isinstance(prediction, dict) or not bool(prediction.get("policy_enabled", True)):
        return False, "no_prediction"
    direction = str(direction).upper()
    if direction not in {"BUY", "SELL"}:
        return False, "invalid_direction"
    if int(bars_held) < int(min_bars):
        return False, "too_early"

    min_p = max(0.0, _num(prediction, "min_tp_probability", prediction.get("min_confidence", 1.0)))
    min_ev = _num(prediction, "min_expected_r", 0.0)
    if direction == "BUY":
        own_p = _num(prediction, "p_buy", 0.0)
        opp_p = _num(prediction, "p_sell", 0.0)
        own_ev = _num(prediction, "buy_expected_r", 0.0)
        opp_ev = _num(prediction, "sell_expected_r", 0.0)
        own_setup = _num(prediction, "setup_buy_score", 0.0)
        opp_setup = _num(prediction, "setup_sell_score", 0.0)
    else:
        own_p = _num(prediction, "p_sell", 0.0)
        opp_p = _num(prediction, "p_buy", 0.0)
        own_ev = _num(prediction, "sell_expected_r", 0.0)
        opp_ev = _num(prediction, "buy_expected_r", 0.0)
        own_setup = _num(prediction, "setup_sell_score", 0.0)
        opp_setup = _num(prediction, "setup_buy_score", 0.0)

    probability_floor = min_p * max(0.0, float(own_probability_fraction))
    own_collapsed = own_p < probability_floor or own_ev <= max(float(own_ev_floor), min_ev * 0.20)
    opposite_dominates = (
        opp_ev >= own_ev + max(0.0, float(opposite_ev_margin))
        and opp_p >= own_p + max(0.0, float(opposite_probability_margin))
    )

    # Structural scores are an additional causal confirmation, not a standalone
    # candle-noise exit.  Only use them when the predictor supplied the new
    # structural-meta fields, and require model weakening at the same time.
    has_setup = "setup_buy_score" in prediction and "setup_sell_score" in prediction
    min_setup = max(0.0, _num(prediction, "min_setup_score", 0.0))
    min_setup_gap = max(0.0, _num(prediction, "min_setup_gap", 0.0))
    setup_floor = min_setup * max(0.0, min(1.0, float(setup_fraction)))
    structural_collapsed = bool(
        has_setup
        and min_setup > 0.0
        and own_setup < setup_floor
        and (own_p < min_p or own_ev < min_ev)
    )
    structural_reversal = bool(
        has_setup
        and opp_setup >= own_setup + min_setup_gap
        and opposite_dominates
    )

    if float(r_multiple) <= -abs(float(adverse_r)):
        if structural_collapsed or structural_reversal:
            return True, "STRUCTURAL_FAILURE"
        if own_collapsed or opposite_dominates:
            return True, "THESIS_FAILURE"

    if int(bars_held) >= int(stale_bars) and float(r_multiple) <= float(stale_max_r):
        if structural_collapsed:
            return True, "STALE_STRUCTURE"
        if own_collapsed:
            return True, "STALE_EDGE"

    return False, "hold"
