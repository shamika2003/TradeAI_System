"""Sandbox-only USD FX broker. NO MetaTrader5 imports or outbound orders."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from math import isfinite
from typing import Any

from .data import Candle


@dataclass(frozen=True, slots=True)
class Signal:
    action: str  # BUY / SELL / CLOSE / HOLD
    lots: float = 0.01
    sl_pips: float = 15.0
    tp_pips: float = 25.0
    note: str = ""


@dataclass(slots=True)
class Position:
    symbol: str
    side: str
    lots: float
    entry_time: datetime
    entry_price: float
    sl: float
    tp: float
    entry_commission: float
    margin_usd: float
    note: str = ""


class PaperBroker:
    def __init__(self, profile: dict[str, Any], *, initial_balance: float = 150.0,
                 leverage: float = 100.0, slippage_multiplier: float = 1.0,
                 max_positions: int = 4, max_lots: float = 0.10,
                 max_spread_pips: float | None = None,
                 stopout_level: float = 0.50, spread_multiplier: float = 1.0):
        if initial_balance <= 0 or leverage <= 0 or max_positions < 1 or max_lots <= 0:
            raise ValueError("Invalid account/risk configuration")
        if (slippage_multiplier < 0 or spread_multiplier < 1 or not (0 < stopout_level <= 1) or
                (max_spread_pips is not None and max_spread_pips < 0)):
            raise ValueError("Invalid execution/stop-out configuration")
        self.profile = profile["symbols"]
        self.balance = float(initial_balance)
        self.start_balance = float(initial_balance)
        self.leverage = leverage
        self.slippage_multiplier = slippage_multiplier
        self.spread_multiplier = spread_multiplier
        self.max_positions = max_positions
        self.max_lots = max_lots
        self.max_spread_pips = max_spread_pips
        self.stopout_level = stopout_level
        self.positions: dict[str, Position] = {}
        self.marks: dict[str, tuple[float, float]] = {}  # most recent bid, ask
        self.trades: list[dict[str, Any]] = []
        self.rejections: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.peak_equity = initial_balance
        self.max_drawdown_usd = 0.0
        self.max_drawdown_pct = 0.0
        self.min_equity = initial_balance
        self.last_equity = initial_balance
        self.negative_equity_seen = False
        self.entry_count = 0
        self.spread_points_seen = 0

    def point(self, symbol: str) -> float:
        return float(self.profile[symbol]["point"])

    def pip(self, symbol: str) -> float:
        p = self.point(symbol)
        digits = int(self.profile[symbol]["digits"])
        return p * (10 if digits in (3, 5) else 1)

    def spread(self, candle: Candle) -> float:
        # A historical zero point spread is allowed; no invented value is inserted.
        return candle.spread_points * self.point(candle.symbol) * self.spread_multiplier

    def slip(self, symbol: str) -> float:
        return (float(self.profile[symbol].get("slippage_pips", 0)) *
                self.slippage_multiplier * self.pip(symbol))

    def commission(self, symbol: str, lots: float) -> float:
        # profile.commission_per_lot denotes the ROUND-TRIP USD commission.
        return lots * float(self.profile[symbol].get("commission_per_lot", 0)) / 2.0

    def _quote_to_usd(self, symbol: str, price: float) -> float:
        # Supplied pairs: EURUSD/GBPUSD (quote USD), USDJPY/USDCNH (base USD).
        if symbol.endswith("USD"):
            return 1.0
        if symbol.startswith("USD"):
            return 1.0 / price
        raise ValueError(f"Unsupported FX conversion for {symbol}; add real-time currency conversion")

    def pnl(self, pos: Position, exit_price: float) -> float:
        direction = 1 if pos.side == "BUY" else -1
        contract = float(self.profile[pos.symbol].get("contract_size", 100_000))
        return direction * (exit_price - pos.entry_price) * contract * pos.lots * self._quote_to_usd(pos.symbol, exit_price)

    def margin(self, symbol: str, lots: float, price: float) -> float:
        contract = float(self.profile[symbol].get("contract_size", 100_000))
        notional = lots * contract * (price if symbol.endswith("USD") else 1.0)
        return notional / self.leverage

    def quote(self, candle: Candle, *, at_open: bool) -> tuple[float, float]:
        bid = candle.open if at_open else candle.close
        return bid, bid + self.spread(candle)

    @property
    def margin_used(self) -> float:
        return sum(p.margin_usd for p in self.positions.values())

    @property
    def equity(self) -> float:
        e = self.balance
        for pos in self.positions.values():
            if pos.symbol in self.marks:
                bid, ask = self.marks[pos.symbol]
                e += self.pnl(pos, bid if pos.side == "BUY" else ask)
        return e

    def update_mark(self, candle: Candle, *, at_open: bool) -> None:
        self.marks[candle.symbol] = self.quote(candle, at_open=at_open)

    def reject(self, at: datetime, symbol: str, reason: str) -> None:
        self.rejections.append({"time": at.isoformat(), "symbol": symbol, "reason": reason})

    def _close(self, symbol: str, at: datetime, bid: float, ask: float, reason: str) -> None:
        pos = self.positions.pop(symbol)
        price = (bid - self.slip(symbol)) if pos.side == "BUY" else (ask + self.slip(symbol))
        price = max(self.point(symbol), price)
        gross = self.pnl(pos, price)
        exit_fee = self.commission(symbol, pos.lots)
        self.balance += gross - exit_fee
        net = gross - exit_fee - pos.entry_commission
        self.trades.append({
            "symbol": symbol, "side": pos.side, "lots": pos.lots,
            "entry_time": pos.entry_time.isoformat(), "exit_time": at.isoformat(),
            "entry_price": pos.entry_price, "exit_price": price, "sl": pos.sl, "tp": pos.tp,
            "gross_pnl_usd": round(gross, 8), "net_pnl_usd": round(net, 8),
            "commission_usd": round(pos.entry_commission + exit_fee, 8),
            "close_reason": reason, "note": pos.note,
        })
        self.events.append({"time": at.isoformat(), "symbol": symbol, "event": "CLOSE", "reason": reason})

    def check_open_gap(self, candle: Candle) -> None:
        """If a stop was crossed at a gap, fill at worse opening quote (not hoped-for SL)."""
        pos = self.positions.get(candle.symbol)
        if pos is None:
            return
        bid, ask = self.quote(candle, at_open=True)
        if pos.side == "BUY":
            if bid <= pos.sl:
                self._close(candle.symbol, candle.time, bid, ask, "SL_GAP")
            elif bid >= pos.tp:
                self._close(candle.symbol, candle.time, pos.tp, ask, "TP_GAP")
        else:
            if ask >= pos.sl:
                self._close(candle.symbol, candle.time, bid, ask, "SL_GAP")
            elif ask <= pos.tp:
                self._close(candle.symbol, candle.time, bid, pos.tp, "TP_GAP")

    def execute(self, candle: Candle, signal: Signal) -> None:
        symbol = candle.symbol
        action = signal.action.strip().upper()
        if action in ("HOLD", ""):
            return
        if action not in {"BUY", "SELL", "CLOSE"}:
            self.reject(candle.time, symbol, "invalid signal action")
            return
        bid, ask = self.quote(candle, at_open=True)
        # All actions are executed using the current candle OPEN only.
        if action == "CLOSE":
            if symbol in self.positions:
                self._close(symbol, candle.time, bid, ask, "SIGNAL_CLOSE")
            return
        if symbol in self.positions:
            if self.positions[symbol].side == action:
                return  # no pyramiding in v1
            self._close(symbol, candle.time, bid, ask, "REVERSE")
        params = self.profile[symbol]
        step = float(params["volume_step"])
        lot = signal.lots
        min_lot = float(params["volume_min"])
        max_lot = min(float(params["volume_max"]), self.max_lots)
        if not isfinite(lot) or not (min_lot <= lot <= max_lot) or abs(round(lot / step) * step - lot) > 1e-8:
            self.reject(candle.time, symbol, "lot outside broker/risk limits or invalid step")
            return
        if not all(isfinite(v) and v > 0 for v in (signal.sl_pips, signal.tp_pips)):
            self.reject(candle.time, symbol, "SL/TP must be finite positive pip distances")
            return
        spread_pips = (ask - bid) / self.pip(symbol)
        if self.max_spread_pips is not None and spread_pips > self.max_spread_pips:
            self.reject(candle.time, symbol, "spread cap exceeded")
            return
        fill = (ask + self.slip(symbol)) if action == "BUY" else (bid - self.slip(symbol))
        sl = fill + (-signal.sl_pips if action == "BUY" else signal.sl_pips) * self.pip(symbol)
        tp = fill + (signal.tp_pips if action == "BUY" else -signal.tp_pips) * self.pip(symbol)
        # Ensure both exits are genuinely away from the current *closing-side* quote.
        valid = (sl < bid and tp > bid) if action == "BUY" else (sl > ask and tp < ask)
        if not valid:
            self.reject(candle.time, symbol, "SL/TP inside opening spread/slippage")
            return
        if len(self.positions) >= self.max_positions:
            self.reject(candle.time, symbol, "maximum open positions exceeded")
            return
        initial_margin = self.margin(symbol, lot, fill)
        entry_fee = self.commission(symbol, lot)
        immediate_pnl = self.pnl(Position(symbol, action, lot, candle.time, fill, sl, tp, entry_fee, initial_margin),
                                 bid if action == "BUY" else ask)
        if self.equity - entry_fee + immediate_pnl < self.margin_used + initial_margin:
            self.reject(candle.time, symbol, "insufficient free margin including spread/commission")
            return
        self.balance -= entry_fee
        self.positions[symbol] = Position(symbol, action, lot, candle.time, fill, sl, tp,
                                          entry_fee, initial_margin, signal.note)
        self.entry_count += 1
        self.events.append({"time": candle.time.isoformat(), "symbol": symbol, "event": "OPEN", "reason": action})

    def check_range(self, candle: Candle) -> None:
        pos = self.positions.get(candle.symbol)
        if pos is None:
            return
        # Spread is approximated as constant across this bar. HIGH/LOW have no reliable event ordering.
        spread = self.spread(candle)
        if pos.side == "BUY":
            stop = candle.low <= pos.sl
            target = candle.high >= pos.tp
        else:
            stop = candle.high + spread >= pos.sl
            target = candle.low + spread <= pos.tp
        # Conservative ambiguity: stop wins if both touched in the same candle.
        if stop:
            if pos.side == "BUY":
                self._close(candle.symbol, candle.ends_at, pos.sl, pos.sl + spread, "SL" if not target else "SL_BOTH_TOUCHED")
            else:
                self._close(candle.symbol, candle.ends_at, pos.sl - spread, pos.sl, "SL" if not target else "SL_BOTH_TOUCHED")
        elif target:
            if pos.side == "BUY":
                self._close(candle.symbol, candle.ends_at, pos.tp, pos.tp + spread, "TP")
            else:
                self._close(candle.symbol, candle.ends_at, pos.tp - spread, pos.tp, "TP")

    def record_equity(self) -> None:
        e = self.equity
        self.last_equity = e
        self.min_equity = min(self.min_equity, e)
        self.negative_equity_seen |= e < 0
        self.peak_equity = max(self.peak_equity, e)
        dd = max(0, self.peak_equity - e)
        self.max_drawdown_usd = max(self.max_drawdown_usd, dd)
        if self.peak_equity > 0:
            self.max_drawdown_pct = max(self.max_drawdown_pct, dd / self.peak_equity * 100)

    def check_stopout(self, candle: Candle) -> None:
        # Bar-close stopout approximation, not intra-bar stopout.
        while self.positions and self.equity <= self.margin_used * self.stopout_level:
            # Liquidate largest floating loss first; deterministic tie-break.
            symbol = sorted(self.positions, key=lambda s: (
                self.pnl(self.positions[s], self.marks[s][0] if self.positions[s].side == "BUY" else self.marks[s][1]), s))[0]
            bid, ask = self.marks[symbol]
            self._close(symbol, candle.ends_at, bid, ask, "MARGIN_STOPOUT_APPROX")

    def liquidate_end(self, at: datetime) -> None:
        for symbol in sorted(list(self.positions)):
            if symbol in self.marks:
                bid, ask = self.marks[symbol]
                self._close(symbol, at, bid, ask, "END_OF_REPLAY")
        self.record_equity()
