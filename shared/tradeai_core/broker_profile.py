from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROFILE_VERSION = 1


def _finite_positive(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number <= 0:
        return default
    return number


def _finite_nonnegative(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number < 0:
        return default
    return number


def pip_size_from_point_digits(point: float, digits: int) -> float:
    point = float(point)
    digits = int(digits)
    if point <= 0:
        raise ValueError("point must be positive")
    return point * 10.0 if digits in (3, 5) else point


def load_broker_profile(path: str | Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", 0)) != PROFILE_VERSION:
        return None
    if not isinstance(payload.get("symbols"), dict):
        return None
    return payload


def save_broker_profile(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def get_symbol_spec(profile: dict | None, symbol: str) -> dict | None:
    if not profile:
        return None
    symbols = profile.get("symbols")
    if not isinstance(symbols, dict):
        return None
    spec = symbols.get(str(symbol).upper())
    return spec if isinstance(spec, dict) else None


def symbol_info_from_profile(profile: dict | None, symbol: str):
    spec = get_symbol_spec(profile, symbol)
    if spec is None:
        return None

    point = _finite_positive(spec.get("point"))
    digits = int(spec.get("digits", 0) or 0)
    tick_size = _finite_positive(spec.get("trade_tick_size"), point)
    tick_value = _finite_positive(spec.get("trade_tick_value"), 0.0)

    if point is None or digits <= 0:
        return None

    return SimpleNamespace(
        digits=digits,
        point=point,
        trade_tick_size=tick_size or point,
        trade_tick_value=tick_value or 0.0,
        volume_min=_finite_positive(spec.get("volume_min"), 0.0) or 0.0,
        volume_max=_finite_positive(spec.get("volume_max"), 0.0) or 0.0,
        volume_step=_finite_positive(spec.get("volume_step"), 0.0) or 0.0,
        trade_stops_level=int(spec.get("trade_stops_level", 0) or 0),
        trade_freeze_level=int(spec.get("trade_freeze_level", 0) or 0),
    )


def get_execution_costs(
    profile: dict | None,
    symbol: str,
    *,
    default_spread_pips: float,
    default_slippage_pips: float,
    default_commission_per_lot: float,
) -> dict:
    spec = get_symbol_spec(profile, symbol) or {}

    spread = _finite_nonnegative(
        spec.get("spread_pips"),
        float(default_spread_pips),
    )
    slippage = _finite_nonnegative(
        spec.get("slippage_pips"),
        float(default_slippage_pips),
    )
    commission = _finite_nonnegative(
        spec.get("commission_per_lot"),
        float(default_commission_per_lot),
    )

    return {
        "spread_pips": float(spread),
        "slippage_pips": float(slippage),
        "commission_per_lot": float(commission),
    }


def capture_mt5_broker_profile(
    mt5_module,
    symbols,
    *,
    path: str | Path,
    commission_per_lot: float,
    slippage_pips: float,
) -> dict:
    account = mt5_module.account_info()
    account_currency = getattr(account, "currency", None) if account is not None else None

    payload = {
        "version": PROFILE_VERSION,
        "source": "MT5_CURRENT_BROKER_SNAPSHOT",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "account_currency": account_currency,
        "symbols": {},
    }

    for raw_symbol in symbols:
        symbol = str(raw_symbol).upper()
        info = mt5_module.symbol_info(symbol)
        if info is None:
            continue

        point = _finite_positive(getattr(info, "point", None))
        digits = int(getattr(info, "digits", 0) or 0)
        if point is None or digits <= 0:
            continue

        pip = pip_size_from_point_digits(point, digits)
        tick = mt5_module.symbol_info_tick(symbol)
        spread_pips = None
        if tick is not None:
            try:
                bid = float(getattr(tick, "bid", 0.0))
                ask = float(getattr(tick, "ask", 0.0))
                if ask > 0 and bid > 0 and ask >= bid:
                    spread_pips = (ask - bid) / pip
            except Exception:
                spread_pips = None

        if spread_pips is None:
            spread_points = _finite_nonnegative(getattr(info, "spread", None), 0.0) or 0.0
            spread_pips = (spread_points * point) / pip

        payload["symbols"][symbol] = {
            "digits": digits,
            "point": point,
            "trade_tick_size": _finite_positive(
                getattr(info, "trade_tick_size", None),
                point,
            ),
            "trade_tick_value": _finite_positive(
                getattr(info, "trade_tick_value", None),
                0.0,
            ) or 0.0,
            "volume_min": _finite_positive(getattr(info, "volume_min", None), 0.0) or 0.0,
            "volume_max": _finite_positive(getattr(info, "volume_max", None), 0.0) or 0.0,
            "volume_step": _finite_positive(getattr(info, "volume_step", None), 0.0) or 0.0,
            "trade_stops_level": int(getattr(info, "trade_stops_level", 0) or 0),
            "trade_freeze_level": int(getattr(info, "trade_freeze_level", 0) or 0),
            "spread_pips": float(spread_pips),
            "slippage_pips": float(slippage_pips),
            "commission_per_lot": float(commission_per_lot),
        }

    save_broker_profile(path, payload)
    return payload
