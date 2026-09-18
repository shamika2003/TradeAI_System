from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from shared.tradeai_core.broker_profile import (
    get_execution_costs,
    load_broker_profile,
    symbol_info_from_profile,
)
from shared.tradeai_core.production_economics import (
    fallback_pip_value_per_lot,
    normalize_lot,
    pip_size,
)


@dataclass(frozen=True)
class ValidationExecutionCosts:
    spread_pips: float
    slippage_pips: float
    commission_per_lot: float


class ValidationExecutionContext:
    """Production-like economics for Stage 4/5 offline validation.

    The runtime backtester now consumes a captured MT5 broker profile.  Stage 4
    and Stage 5 must use the same symbol rules or their calibrated policy/risk
    can drift from the final runtime backtest.  This helper centralises those
    rules for the validation scripts.
    """

    def __init__(self, settings, *, require_profile: bool = False):
        self.settings = settings
        profile_path = Path(getattr(settings, "BROKER_PROFILE_PATH", ""))
        self.profile = load_broker_profile(profile_path) if str(profile_path) else None
        if require_profile and not self.profile:
            raise RuntimeError(
                f"Broker profile missing or invalid: {profile_path}. "
                "Capture the MT5 broker profile before final validation."
            )

    def has_symbol_profile(self, symbol: str) -> bool:
        return symbol_info_from_profile(self.profile, symbol) is not None

    def assert_symbols_profiled(self, symbols) -> None:
        missing = [s for s in symbols if not self.has_symbol_profile(s)]
        if missing:
            raise RuntimeError(
                "Broker profile is incomplete for final validation: " + ", ".join(missing)
            )

    def costs(self, symbol: str, scenario_name: str = "NORMAL") -> ValidationExecutionCosts:
        base = get_execution_costs(
            self.profile,
            symbol,
            default_spread_pips=float(self.settings.DEFAULT_SPREAD_PIPS),
            default_slippage_pips=float(self.settings.SIMULATED_SLIPPAGE_PIPS),
            default_commission_per_lot=float(self.settings.COMMISSION_PER_LOT),
        )

        if str(scenario_name).upper() == "STRESS":
            spread_ratio = float(self.settings.STRESS_SPREAD_PIPS) / max(
                float(self.settings.DEFAULT_SPREAD_PIPS), 1e-12
            )
            slippage_ratio = float(self.settings.STRESS_SLIPPAGE_PIPS) / max(
                float(self.settings.SIMULATED_SLIPPAGE_PIPS), 1e-12
            )
            commission_ratio = float(self.settings.STRESS_COMMISSION_PER_LOT) / max(
                float(self.settings.COMMISSION_PER_LOT), 1e-12
            )
            return ValidationExecutionCosts(
                spread_pips=float(base["spread_pips"]) * spread_ratio,
                slippage_pips=float(base["slippage_pips"]) * slippage_ratio,
                commission_per_lot=float(base["commission_per_lot"]) * commission_ratio,
            )

        return ValidationExecutionCosts(
            spread_pips=float(base["spread_pips"]),
            slippage_pips=float(base["slippage_pips"]),
            commission_per_lot=float(base["commission_per_lot"]),
        )

    def symbol_info(self, symbol: str):
        return symbol_info_from_profile(self.profile, symbol)

    def lot_limits(self, symbol: str) -> tuple[float, float, float]:
        info = self.symbol_info(symbol)
        if info is not None:
            minimum = float(getattr(info, "volume_min", 0.0) or 0.0)
            maximum = float(getattr(info, "volume_max", 0.0) or 0.0)
            step = float(getattr(info, "volume_step", 0.0) or 0.0)
            if minimum > 0 and maximum >= minimum and step > 0:
                return minimum, maximum, step
        return (
            float(self.settings.MIN_LOT),
            float(self.settings.MAX_LOT),
            float(self.settings.LOT_STEP),
        )

    def normalize_price(self, symbol: str, price: float) -> float:
        value = float(price)
        info = self.symbol_info(symbol)
        if info is None:
            return value
        digits = int(getattr(info, "digits", 0) or 0)
        tick_size = float(
            getattr(info, "trade_tick_size", 0.0)
            or getattr(info, "point", 0.0)
            or 0.0
        )
        if tick_size > 0:
            value = round(value / tick_size) * tick_size
        if digits > 0:
            value = round(value, digits)
        return value

    def pip_value_per_lot(self, symbol: str, price: float) -> float:
        info = self.symbol_info(symbol)
        if info is not None:
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
            tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
            if tick_size > 0 and tick_value > 0:
                value = tick_value * (pip_size(symbol) / tick_size)
                if math.isfinite(value) and value > 0:
                    return value
        return float(fallback_pip_value_per_lot(symbol, price))

    def minimum_stop_distance(self, symbol: str) -> float:
        info = self.symbol_info(symbol)
        if info is None:
            return 0.0
        point = float(getattr(info, "point", 0.0) or 0.0)
        stops = int(getattr(info, "trade_stops_level", 0) or 0)
        freeze = int(getattr(info, "trade_freeze_level", 0) or 0)
        if point <= 0:
            return 0.0
        return max(stops, freeze) * point

    def entry_price(self, symbol: str, direction: str, requested_bid: float, costs: ValidationExecutionCosts) -> float:
        """Mirror PaperExecutor Stage 4 BID-side entry semantics."""
        direction = str(direction).upper()
        requested = float(requested_bid)
        pip = pip_size(symbol)
        spread = costs.spread_pips * pip
        slippage = costs.slippage_pips * pip
        side = str(getattr(self.settings, "HISTORICAL_OHLC_SIDE", "BID")).strip().upper()

        if direction not in {"BUY", "SELL"}:
            raise ValueError(f"Invalid direction: {direction}")

        if side == "BID":
            market = requested + spread if direction == "BUY" else requested
        else:
            market = requested + spread / 2.0 if direction == "BUY" else requested - spread / 2.0

        fill = market + slippage if direction == "BUY" else market - slippage
        return self.normalize_price(symbol, fill)

    def sell_side_prices(self, symbol: str, *, high: float, low: float, close: float, costs: ValidationExecutionCosts):
        """Return prices used to close a SELL position.

        MT5 historical FX OHLC is BID-side in this project, therefore a SELL
        position is closed on ASK = BID + spread.
        """
        side = str(getattr(self.settings, "HISTORICAL_OHLC_SIDE", "BID")).strip().upper()
        if side != "BID":
            return float(high), float(low), float(close)
        spread_price = costs.spread_pips * pip_size(symbol)
        return (
            float(high) + spread_price,
            float(low) + spread_price,
            float(close) + spread_price,
        )

    def net_risk_sized_lot(
        self,
        *,
        symbol: str,
        direction: str,
        requested_price: float,
        stop_loss: float,
        balance: float,
        risk_percent: float,
        costs: ValidationExecutionCosts,
        max_actual_risk_percent: float,
    ) -> tuple[float | None, dict]:
        entry = self.entry_price(symbol, direction, requested_price, costs)
        stop = self.normalize_price(symbol, stop_loss)
        pip = pip_size(symbol)
        stop_pips = abs(entry - stop) / pip
        pip_value = self.pip_value_per_lot(symbol, entry)
        risk_per_lot = stop_pips * pip_value + abs(float(costs.commission_per_lot))

        info = {
            "entry_price": float(entry),
            "stop_loss": float(stop),
            "stop_pips": float(stop_pips),
            "pip_value_per_lot": float(pip_value),
            "risk_per_lot": float(risk_per_lot),
        }

        if balance <= 0 or risk_percent <= 0 or stop_pips <= 0 or pip_value <= 0 or risk_per_lot <= 0:
            info["reason"] = "invalid_input"
            return None, info

        allowed_risk = float(balance) * float(risk_percent) / 100.0
        raw_lot = allowed_risk / risk_per_lot
        minimum, maximum, step = self.lot_limits(symbol)

        # Runtime RiskManager deliberately supports a broker-minimum fallback:
        # when the mathematically sized lot is below the broker minimum, try
        # the minimum lot and then judge the REAL net risk against the hard
        # max-actual-risk cap. Validation must mirror that exact behaviour or
        # a $150 account can show zero executable trades offline while runtime
        # would legally accept the same 0.01-lot setup.
        used_minimum_fallback = raw_lot < minimum
        lot_request = minimum if used_minimum_fallback else raw_lot
        lot = normalize_lot(lot_request, minimum, maximum, step)
        info.update({
            "allowed_risk": float(allowed_risk),
            "raw_lot": float(raw_lot),
            "minimum": float(minimum),
            "maximum": float(maximum),
            "step": float(step),
            "used_minimum_fallback": bool(used_minimum_fallback),
        })

        if lot is None:
            info["reason"] = "below_minimum_or_invalid_lot"
            return None, info

        net_risk = risk_per_lot * lot
        actual_pct = net_risk / float(balance) * 100.0
        info.update({
            "lot": float(lot),
            "net_risk": float(net_risk),
            "actual_risk_percent": float(actual_pct),
        })

        if actual_pct > float(max_actual_risk_percent) + 1e-12:
            info["reason"] = "actual_risk_limit"
            return None, info

        return float(lot), info
