from __future__ import annotations

import math

CONTRACT_SIZE = 100_000.0


def pip_size(symbol: str) -> float:
    symbol = str(symbol).upper()
    return 0.01 if "JPY" in symbol else 0.0001


def fallback_pip_value_per_lot(symbol: str, price: float | None = None) -> float:
    """USD-account FX pip value fallback for one standard lot.

    - XXXUSD pairs: one pip is CONTRACT_SIZE * pip_size USD.
    - USDXXX pairs: pip value is in quote currency and is converted back to USD
      by dividing by the current pair price.
    - Other crosses fall back to the USD-quoted approximation because a second
      conversion rate would be required.
    """
    symbol = str(symbol).upper()
    pip = pip_size(symbol)
    quote_value = CONTRACT_SIZE * pip

    if symbol.endswith("USD"):
        return quote_value

    if symbol.startswith("USD") and price is not None:
        try:
            px = float(price)
            if math.isfinite(px) and px > 0:
                return quote_value / px
        except Exception:
            pass

    # Backward-compatible fallback for unsupported crosses.
    if "JPY" in symbol:
        return 6.7
    return 10.0


def normalize_lot(
    raw_lot: float,
    minimum: float,
    maximum: float,
    step: float,
) -> float | None:
    try:
        raw = float(raw_lot)
        minimum = float(minimum)
        maximum = float(maximum)
        step = float(step)
    except Exception:
        return None

    if not all(math.isfinite(x) for x in (raw, minimum, maximum, step)):
        return None
    if raw <= 0 or minimum <= 0 or maximum <= 0 or step <= 0 or maximum < minimum:
        return None
    if raw < minimum:
        return None

    raw = min(raw, maximum)
    steps = math.floor(((raw - minimum) / step) + 1e-9)
    normalized = minimum + steps * step
    normalized = max(minimum, min(normalized, maximum))

    decimals = 0
    text = f"{step:.10f}".rstrip("0")
    if "." in text:
        decimals = len(text.split(".")[1])
    normalized = round(normalized, decimals)

    if normalized < minimum or normalized > maximum:
        return None
    return normalized


def risk_sized_lot(
    *,
    balance: float,
    risk_percent: float,
    stop_distance: float,
    symbol: str,
    price: float,
    minimum: float,
    maximum: float,
    step: float,
    max_actual_risk_percent: float | None = None,
) -> tuple[float | None, dict]:
    balance = float(balance)
    risk_percent = float(risk_percent)
    stop_distance = float(stop_distance)
    px = float(price)

    if balance <= 0 or risk_percent <= 0 or stop_distance <= 0 or px <= 0:
        return None, {"reason": "invalid_input"}

    pip = pip_size(symbol)
    stop_pips = stop_distance / pip
    pip_value = fallback_pip_value_per_lot(symbol, px)
    if stop_pips <= 0 or pip_value <= 0:
        return None, {"reason": "invalid_pip_economics"}

    allowed_risk = balance * risk_percent / 100.0
    raw_lot = allowed_risk / (stop_pips * pip_value)
    lot = normalize_lot(raw_lot, minimum, maximum, step)

    info = {
        "allowed_risk": allowed_risk,
        "raw_lot": raw_lot,
        "stop_pips": stop_pips,
        "pip_value_per_lot": pip_value,
    }

    if lot is None:
        info["reason"] = "below_minimum_or_invalid_lot"
        return None, info

    actual_risk = stop_pips * pip_value * lot
    actual_risk_percent = actual_risk / balance * 100.0
    info.update({
        "lot": lot,
        "actual_risk": actual_risk,
        "actual_risk_percent": actual_risk_percent,
    })

    if max_actual_risk_percent is not None:
        if actual_risk_percent > float(max_actual_risk_percent) + 1e-12:
            info["reason"] = "actual_risk_limit"
            return None, info

    return lot, info


def entry_price_with_costs(
    *,
    requested_price: float,
    direction: str,
    symbol: str,
    spread_pips: float,
    slippage_pips: float,
) -> float:
    requested = float(requested_price)
    pip = pip_size(symbol)
    half_spread = float(spread_pips) * pip / 2.0
    slippage = float(slippage_pips) * pip

    direction = str(direction).upper()
    if direction == "BUY":
        return requested + half_spread + slippage
    if direction == "SELL":
        return requested - half_spread - slippage
    raise ValueError(f"Invalid direction: {direction}")
