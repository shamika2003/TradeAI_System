from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from Trade_Bot_Training.precision_config import (
    ACTION_CONFIDENCE,
    HORIZON_BARS,
    HORIZON_MINUTES,
    MIN_MEANINGFUL_MOVE_ATR,
    SYMBOLS as MODEL_SYMBOLS,
)
from TradeAI.analytics.logger import log
from TradeAI.config.settings import DEVIATION
from TradeAI.precision_live_engine import (
    PrecisionLiveEngine,
    _atomic_json,
    _read_json,
    _utc_now,
)


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = SYSTEM_ROOT / "artifacts" / "reports"

EXECUTION_STATE_PATH = REPORT_DIR / "precision_demo_execution_state.json"
EXECUTION_EVENTS_CSV = REPORT_DIR / "precision_demo_execution_events.csv"
CLOSED_TRADES_CSV = REPORT_DIR / "precision_demo_closed_trades.csv"

# Distinct from the old TradeAI executor magic so precision-demo positions can
# never be confused with positions opened by legacy engines.
PRECISION_DEMO_MAGIC = 26061055

COMMENT_OPEN_PREFIX = "TAI-P"
COMMENT_CLOSE_PREFIX = "TAI-X"

# We deliberately use broker minimum size. This is an execution/forward-test
# harness, not a position-sizing experiment.
USE_BROKER_MINIMUM_VOLUME = True

# One position per symbol prevents stacking repeated M5 signals on top of each
# other. All signals are still preserved by precision_live_engine for model
# auditing even when execution is skipped.
MAX_BOT_POSITIONS = 3

EVENT_FIELDS = [
    "event_utc",
    "event",
    "decision_id",
    "client_tag",
    "symbol",
    "action",
    "signal",
    "confidence",
    "p_down",
    "p_hold",
    "p_up",
    "bar_time",
    "decision_time",
    "scheduled_exit_bar",
    "volume",
    "request_price",
    "result_price",
    "position_ticket",
    "order_ticket",
    "deal_ticket",
    "retcode",
    "retcode_text",
    "spread_pips",
    "broker_profit",
    "note",
]

CLOSED_FIELDS = [
    "closed_utc",
    "decision_id",
    "client_tag",
    "symbol",
    "action",
    "signal",
    "confidence",
    "p_down",
    "p_hold",
    "p_up",
    "decision_time",
    "opened_utc",
    "closed_utc_broker_event",
    "scheduled_exit_bar",
    "volume",
    "entry_price",
    "exit_price",
    "position_ticket",
    "order_ticket",
    "deal_ticket",
    "close_order_ticket",
    "close_deal_ticket",
    "broker_profit",
    "close_reason",
]


def _append_csv(path: Path, row: dict[str, Any], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        if not exists:
            writer.writeheader()

        safe: dict[str, Any] = {}
        for field in fields:
            value = row.get(field)
            if isinstance(value, (pd.Timestamp, datetime)):
                value = str(value)
            safe[field] = value
        writer.writerow(safe)


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _client_tag(decision_id: str) -> str:
    digest = hashlib.sha256(
        decision_id.encode("utf-8")
    ).hexdigest()[:12].upper()
    return digest


def _short_comment(prefix: str, tag: str) -> str:
    # MT5/brokers commonly limit order comments. Keep this comfortably short.
    return f"{prefix}-{tag}"[:30]


class PrecisionDemoEngine(PrecisionLiveEngine):
    """Precision model + hard-blocked MT5 DEMO execution.

    This intentionally does NOT use the legacy direction signal policy.

    Entry:
      * only a new closed-M5 PrecisionDirectionModel decision;
      * only when model row tradeable=True;
      * only when the symbol's historical gate passes;
      * DEMO account re-verified immediately before every order;
      * broker minimum volume;
      * no duplicate decision execution;
      * no stacking on a symbol that already has any broker position.

    Exit:
      * closes only this engine's own magic/comment-tagged position;
      * fixed model horizon: after the sixth M5 target bar closes;
      * no SL/TP/trailing/break-even layer is added here.

    This keeps the demo execution test aligned with the model's 30-minute
    prediction contract instead of hiding model quality behind trade management.
    """

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        poll_seconds: float = 1.0,
    ) -> None:
        super().__init__(
            symbols=symbols,
            poll_seconds=poll_seconds,
        )

        raw = _read_json(
            EXECUTION_STATE_PATH,
            {},
        )
        if not isinstance(raw, dict):
            raw = {}

        self.execution_state: dict[str, Any] = {
            "version": "tradeai_precision_demo_execution_v1",
            "started_utc": str(raw.get("started_utc") or _utc_now()),
            "updated_utc": _utc_now(),
            "magic": PRECISION_DEMO_MAGIC,
            "mode": "DEMO_ONLY",
            "execution_enabled": True,
            "broker_minimum_volume": True,
            "max_bot_positions": MAX_BOT_POSITIONS,
            "open_positions": (
                raw.get("open_positions", {})
                if isinstance(raw.get("open_positions"), dict)
                else {}
            ),
            "decisions": (
                raw.get("decisions", {})
                if isinstance(raw.get("decisions"), dict)
                else {}
            ),
            "completed": (
                raw.get("completed", {})
                if isinstance(raw.get("completed"), dict)
                else {}
            ),
            "account": {},
        }

        self._startup_login: int | None = None
        self._demo_verified = False

        # Make the existing live-state artifact accurately say execution is on.
        self.state["execution_enabled"] = True
        self.state["execution_mode"] = "DEMO_ONLY"
        self.state["execution_magic"] = PRECISION_DEMO_MAGIC
        self.state["note"] = (
            "Precision model demo executor. Real-money accounts are hard-blocked. "
            "Broker minimum lot is used. Positions close at the model's fixed "
            f"{HORIZON_MINUTES}-minute horizon."
        )

    # ------------------------------------------------------------------
    # Persistent execution state
    # ------------------------------------------------------------------

    def _save_execution_state(self) -> None:
        self.execution_state["updated_utc"] = _utc_now()
        _atomic_json(
            EXECUTION_STATE_PATH,
            self.execution_state,
        )

    def _event(
        self,
        event: str,
        *,
        row: dict[str, Any] | None = None,
        record: dict[str, Any] | None = None,
        note: str = "",
        request_price: float | None = None,
        result_price: float | None = None,
        retcode: Any = None,
        retcode_text: str = "",
        order_ticket: Any = None,
        deal_ticket: Any = None,
        broker_profit: float | None = None,
    ) -> None:
        source = {}
        if isinstance(record, dict):
            source.update(record)
        if isinstance(row, dict):
            source.update(row)

        payload = {
            "event_utc": _utc_now(),
            "event": event,
            "decision_id": source.get("decision_id", ""),
            "client_tag": source.get("client_tag", ""),
            "symbol": source.get("symbol", ""),
            "action": source.get("action", ""),
            "signal": source.get("signal", ""),
            "confidence": source.get("confidence"),
            "p_down": source.get("p_down"),
            "p_hold": source.get("p_hold"),
            "p_up": source.get("p_up"),
            "bar_time": source.get("bar_time", ""),
            "decision_time": source.get("decision_time", ""),
            "scheduled_exit_bar": source.get("scheduled_exit_bar", ""),
            "volume": source.get("volume"),
            "request_price": request_price,
            "result_price": result_price,
            "position_ticket": source.get("position_ticket"),
            "order_ticket": (
                order_ticket
                if order_ticket is not None
                else source.get("order_ticket")
            ),
            "deal_ticket": (
                deal_ticket
                if deal_ticket is not None
                else source.get("deal_ticket")
            ),
            "retcode": retcode,
            "retcode_text": retcode_text,
            "spread_pips": source.get("spread_pips"),
            "broker_profit": broker_profit,
            "note": note,
        }

        _append_csv(
            EXECUTION_EVENTS_CSV,
            payload,
            EVENT_FIELDS,
        )

    # ------------------------------------------------------------------
    # DEMO-only preflight
    # ------------------------------------------------------------------

    def _assert_demo_account(self) -> Any:
        if self.mt5 is None:
            raise RuntimeError("MT5 is not connected.")

        account = self.mt5.account_info()
        if account is None:
            raise RuntimeError(
                f"MT5 account unavailable: {self.mt5.last_error()}"
            )

        demo_mode = int(
            getattr(
                self.mt5,
                "ACCOUNT_TRADE_MODE_DEMO",
                0,
            )
        )
        current_mode = int(
            getattr(
                account,
                "trade_mode",
                -1,
            )
        )

        if current_mode != demo_mode:
            raise RuntimeError(
                "REAL-MONEY ACCOUNT DETECTED. "
                "Precision demo execution is HARD BLOCKED."
            )

        login = int(getattr(account, "login", 0) or 0)
        if self._startup_login is not None and login != self._startup_login:
            raise RuntimeError(
                "MT5 account changed while the engine was running. "
                "Execution is blocked until the engine is restarted and "
                "the new DEMO account is verified."
            )

        terminal = self.mt5.terminal_info()
        if terminal is None:
            raise RuntimeError("MT5 terminal information is unavailable.")

        if getattr(terminal, "trade_allowed", True) is False:
            raise RuntimeError(
                "MT5 terminal AutoTrading/API trading is disabled."
            )

        if getattr(account, "trade_allowed", True) is False:
            raise RuntimeError(
                "Trading is disabled for the current DEMO account."
            )

        if getattr(account, "trade_expert", True) is False:
            raise RuntimeError(
                "Expert/API trading is disabled for the current DEMO account."
            )

        return account

    def demo_preflight(self) -> dict[str, Any]:
        account = self._assert_demo_account()

        login = int(getattr(account, "login", 0) or 0)
        if self._startup_login is None:
            self._startup_login = login

        symbols: dict[str, Any] = {}

        for symbol in self.symbols:
            model = self.models[symbol].model
            info = self.mt5.symbol_info(symbol)

            if info is None:
                raise RuntimeError(
                    f"Broker does not expose required symbol: {symbol}"
                )

            if not getattr(info, "visible", True):
                if not self.mt5.symbol_select(symbol, True):
                    raise RuntimeError(
                        f"Unable to enable broker symbol: {symbol}"
                    )
                info = self.mt5.symbol_info(symbol)

            if info is None:
                raise RuntimeError(
                    f"Broker symbol disappeared after selection: {symbol}"
                )

            disabled_mode = getattr(
                self.mt5,
                "SYMBOL_TRADE_MODE_DISABLED",
                0,
            )
            trade_mode = int(
                getattr(info, "trade_mode", disabled_mode)
            )

            tradable = trade_mode != int(disabled_mode)
            volume_min = _finite_float(
                getattr(info, "volume_min", None)
            )
            volume_step = _finite_float(
                getattr(info, "volume_step", None)
            )
            volume_max = _finite_float(
                getattr(info, "volume_max", None)
            )

            if (
                model.historical_gate_pass
                and not tradable
            ):
                raise RuntimeError(
                    f"{symbol}: historical model gate passes but broker "
                    "reports the symbol as non-tradable."
                )

            if (
                model.historical_gate_pass
                and (
                    volume_min is None
                    or volume_min <= 0
                    or volume_step is None
                    or volume_step <= 0
                    or volume_max is None
                    or volume_max < volume_min
                )
            ):
                raise RuntimeError(
                    f"{symbol}: invalid broker volume rules."
                )

            symbols[symbol] = {
                "historical_gate_pass": bool(
                    model.historical_gate_pass
                ),
                "broker_tradable": bool(tradable),
                "volume_min": volume_min,
                "volume_step": volume_step,
                "volume_max": volume_max,
            }

        self._demo_verified = True

        self.execution_state["account"] = {
            # Deliberately do not persist the account login number.
            "verified_demo": True,
            "currency": str(
                getattr(account, "currency", "") or ""
            ),
            "balance": _finite_float(
                getattr(account, "balance", None)
            ),
            "equity": _finite_float(
                getattr(account, "equity", None)
            ),
            "server": str(
                getattr(account, "server", "") or ""
            ),
        }
        self.execution_state["symbols"] = symbols
        self._save_execution_state()

        return {
            "currency": str(
                getattr(account, "currency", "") or ""
            ),
            "balance": _finite_float(
                getattr(account, "balance", None)
            ),
            "equity": _finite_float(
                getattr(account, "equity", None)
            ),
            "server": str(
                getattr(account, "server", "") or ""
            ),
            "symbols": symbols,
        }

    def print_demo_preflight(
        self,
        result: dict[str, Any],
    ) -> None:
        print()
        print("=" * 78)
        print("TRADEAI PRECISION DEMO EXECUTION PREFLIGHT")
        print("=" * 78)
        print("Account mode     : DEMO VERIFIED")
        print(
            f"Balance / Equity : "
            f"{result.get('balance')} / {result.get('equity')} "
            f"{result.get('currency')}"
        )
        print(
            f"Execution size   : BROKER MINIMUM LOT"
        )
        print(
            f"Exit rule        : FIXED {HORIZON_MINUTES} MINUTES "
            f"({HORIZON_BARS} M5 bars)"
        )
        print(
            f"Magic            : {PRECISION_DEMO_MAGIC}"
        )
        print("Symbols:")

        for symbol, spec in result["symbols"].items():
            gate = (
                "PASS"
                if spec["historical_gate_pass"]
                else "DISABLED"
            )
            broker = (
                "YES"
                if spec["broker_tradable"]
                else "NO"
            )
            print(
                f"  {symbol:<6} model={gate:<8} "
                f"broker_trade={broker:<3} "
                f"min={spec['volume_min']} "
                f"step={spec['volume_step']}"
            )

        print("=" * 78)
        print()

    # ------------------------------------------------------------------
    # Broker helpers
    # ------------------------------------------------------------------

    def _managed_positions(self, symbol: str | None = None) -> list[Any]:
        if symbol:
            positions = self.mt5.positions_get(symbol=symbol)
        else:
            positions = self.mt5.positions_get()

        if positions is None:
            return []

        out = []
        for position in positions:
            if int(getattr(position, "magic", 0) or 0) != PRECISION_DEMO_MAGIC:
                continue
            out.append(position)
        return out

    def _all_symbol_positions(self, symbol: str) -> list[Any]:
        positions = self.mt5.positions_get(symbol=symbol)
        return list(positions or [])

    def _find_managed_position(
        self,
        symbol: str,
        *,
        position_ticket: int | None = None,
        client_tag: str = "",
    ) -> Any | None:
        positions = self._managed_positions(symbol)

        if position_ticket:
            for position in positions:
                if int(getattr(position, "ticket", 0) or 0) == int(position_ticket):
                    return position

        expected_open_comment = _short_comment(
            COMMENT_OPEN_PREFIX,
            client_tag,
        )

        if client_tag:
            for position in positions:
                comment = str(getattr(position, "comment", "") or "")
                if (
                    client_tag in comment
                    or comment == expected_open_comment
                ):
                    return position

        # One-per-symbol execution rule means a single managed position on this
        # symbol is unambiguous.
        if len(positions) == 1:
            return positions[0]

        return None

    def _minimum_volume(self, symbol: str) -> float:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(
                f"{symbol}: symbol_info unavailable."
            )

        volume_min = _finite_float(
            getattr(info, "volume_min", None)
        )
        volume_max = _finite_float(
            getattr(info, "volume_max", None)
        )

        if (
            volume_min is None
            or volume_min <= 0
            or volume_max is None
            or volume_min > volume_max
        ):
            raise RuntimeError(
                f"{symbol}: invalid broker minimum volume."
            )

        return float(volume_min)

    def _filling_candidates(self, info: Any) -> list[int]:
        mt5 = self.mt5
        candidates: list[int] = []

        broker_flags = int(
            getattr(info, "filling_mode", 0) or 0
        )

        symbol_fok = int(
            getattr(mt5, "SYMBOL_FILLING_FOK", 1)
        )
        symbol_ioc = int(
            getattr(mt5, "SYMBOL_FILLING_IOC", 2)
        )

        if broker_flags & symbol_fok:
            candidates.append(
                int(mt5.ORDER_FILLING_FOK)
            )

        if broker_flags & symbol_ioc:
            candidates.append(
                int(mt5.ORDER_FILLING_IOC)
            )

        # RETURN is valid for many broker execution modes but not all. It is
        # last because brokers that reject it return INVALID_FILL and we then
        # stop/retry only with another filling candidate.
        candidates.append(
            int(mt5.ORDER_FILLING_RETURN)
        )

        # Safe fallbacks for brokers exposing incomplete filling flags.
        candidates.extend(
            [
                int(mt5.ORDER_FILLING_IOC),
                int(mt5.ORDER_FILLING_FOK),
            ]
        )

        unique: list[int] = []
        for value in candidates:
            if value not in unique:
                unique.append(value)
        return unique

    def _send_market_request(
        self,
        *,
        symbol: str,
        order_type: int,
        volume: float,
        position_ticket: int | None,
        comment: str,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        self._assert_demo_account()

        info = self.mt5.symbol_info(symbol)
        tick = self.mt5.symbol_info_tick(symbol)

        if info is None or tick is None:
            raise RuntimeError(
                f"{symbol}: symbol/tick unavailable."
            )

        if order_type == int(self.mt5.ORDER_TYPE_BUY):
            price = float(tick.ask)
        else:
            price = float(tick.bid)

        if not math.isfinite(price) or price <= 0:
            raise RuntimeError(
                f"{symbol}: invalid market price."
            )

        base_request: dict[str, Any] = {
            "action": int(self.mt5.TRADE_ACTION_DEAL),
            "symbol": symbol,
            "volume": float(volume),
            "type": int(order_type),
            "price": float(price),
            "deviation": int(DEVIATION),
            "magic": int(PRECISION_DEMO_MAGIC),
            "comment": str(comment),
            "type_time": int(self.mt5.ORDER_TIME_GTC),
        }

        if position_ticket:
            base_request["position"] = int(position_ticket)

        accepted = {
            int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009)),
            int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
            int(getattr(self.mt5, "TRADE_RETCODE_PLACED", 10008)),
        }
        invalid_fill = int(
            getattr(
                self.mt5,
                "TRADE_RETCODE_INVALID_FILL",
                10030,
            )
        )

        last_result = None
        last_request = None

        for filling in self._filling_candidates(info):
            request = dict(base_request)
            request["type_filling"] = int(filling)

            result = self.mt5.order_send(request)
            last_result = result
            last_request = request

            if result is None:
                # None is not safe to retry blindly: terminal state can be
                # ambiguous. Fail closed and let the user inspect the log.
                break

            retcode = int(getattr(result, "retcode", -1))

            if retcode in accepted:
                return result, request

            if retcode != invalid_fill:
                break

        return last_result, last_request

    def _result_ok(self, result: Any) -> bool:
        if result is None:
            return False
        accepted = {
            int(getattr(self.mt5, "TRADE_RETCODE_DONE", 10009)),
            int(getattr(self.mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)),
            int(getattr(self.mt5, "TRADE_RETCODE_PLACED", 10008)),
        }
        return int(getattr(result, "retcode", -1)) in accepted

    def _broker_profit_for_position(
        self,
        position_ticket: int | None,
    ) -> float | None:
        if not position_ticket:
            return None

        try:
            deals = self.mt5.history_deals_get(
                position=int(position_ticket)
            )
        except Exception:
            deals = None

        if not deals:
            return None

        total = 0.0
        found = False

        for deal in deals:
            if int(getattr(deal, "magic", 0) or 0) != PRECISION_DEMO_MAGIC:
                continue

            found = True
            for field in (
                "profit",
                "commission",
                "swap",
                "fee",
            ):
                value = _finite_float(
                    getattr(deal, field, 0.0)
                )
                if value is not None:
                    total += value

        return total if found else None

    # ------------------------------------------------------------------
    # Open
    # ------------------------------------------------------------------

    def _execute_tradeable_signal(
        self,
        row: dict[str, Any],
    ) -> None:
        if not bool(row.get("tradeable", False)):
            return

        if not bool(row.get("historical_gate_pass", False)):
            return

        action = str(row.get("action", "")).upper()
        if action not in {"BUY", "SELL"}:
            return

        decision_id = self._decision_id(row)
        decisions = self.execution_state["decisions"]

        # Persisted idempotency: one decision can never be auto-sent twice.
        if decision_id in decisions:
            return

        symbol = str(row["symbol"]).upper()
        tag = _client_tag(decision_id)

        record: dict[str, Any] = {
            "decision_id": decision_id,
            "client_tag": tag,
            "symbol": symbol,
            "action": action,
            "signal": str(row.get("signal", "")),
            "confidence": row.get("confidence"),
            "p_down": row.get("p_down"),
            "p_hold": row.get("p_hold"),
            "p_up": row.get("p_up"),
            "bar_time": row.get("bar_time"),
            "decision_time": row.get("decision_time"),
            "spread_pips": row.get("spread_pips"),
            "status": "INTENT",
            "created_utc": _utc_now(),
        }

        decisions[decision_id] = record
        self._save_execution_state()

        try:
            self._assert_demo_account()

            # Never merge with or interfere with a manually opened position,
            # another EA, or a previous TradeAI engine.
            symbol_positions = self._all_symbol_positions(symbol)
            if symbol_positions:
                record["status"] = "SKIPPED_SYMBOL_ALREADY_OPEN"
                record["note"] = (
                    f"{len(symbol_positions)} broker position(s) already open "
                    f"on {symbol}; no stacking/merging."
                )
                self._save_execution_state()
                self._event(
                    "OPEN_SKIPPED",
                    row=row,
                    record=record,
                    note=record["note"],
                )
                print(
                    f"!!! DEMO SKIP | {symbol:<6} | {action:<4} | "
                    "position already open on symbol"
                )
                return

            managed = self._managed_positions()
            if len(managed) >= MAX_BOT_POSITIONS:
                record["status"] = "SKIPPED_BOT_POSITION_LIMIT"
                record["note"] = (
                    f"Bot already has {len(managed)} managed positions."
                )
                self._save_execution_state()
                self._event(
                    "OPEN_SKIPPED",
                    row=row,
                    record=record,
                    note=record["note"],
                )
                print(
                    f"!!! DEMO SKIP | {symbol:<6} | {action:<4} | "
                    f"bot position limit {MAX_BOT_POSITIONS}"
                )
                return

            volume = self._minimum_volume(symbol)
            record["volume"] = volume

            decision_time = pd.Timestamp(row["decision_time"])
            scheduled_exit_bar = (
                decision_time
                + pd.Timedelta(
                    minutes=5 * (HORIZON_BARS - 1)
                )
            )
            record["scheduled_exit_bar"] = str(
                scheduled_exit_bar
            )

            # Persist the exact intent before order_send. If Python/MT5 crashes
            # during order_send, restart will never blindly duplicate the order.
            record["status"] = "SENDING"
            self._save_execution_state()

            if action == "BUY":
                order_type = int(self.mt5.ORDER_TYPE_BUY)
            else:
                order_type = int(self.mt5.ORDER_TYPE_SELL)

            result, request = self._send_market_request(
                symbol=symbol,
                order_type=order_type,
                volume=volume,
                position_ticket=None,
                comment=_short_comment(
                    COMMENT_OPEN_PREFIX,
                    tag,
                ),
            )

            request_price = (
                _finite_float(request.get("price"))
                if isinstance(request, dict)
                else None
            )

            if not self._result_ok(result):
                retcode = (
                    getattr(result, "retcode", None)
                    if result is not None
                    else None
                )
                text = (
                    str(getattr(result, "comment", "") or "")
                    if result is not None
                    else str(self.mt5.last_error())
                )

                record["status"] = "OPEN_FAILED"
                record["retcode"] = retcode
                record["retcode_text"] = text
                self._save_execution_state()

                self._event(
                    "OPEN_FAILED",
                    row=row,
                    record=record,
                    request_price=request_price,
                    retcode=retcode,
                    retcode_text=text,
                    note="Order was not accepted.",
                )

                print(
                    f"XXX DEMO OPEN FAILED | {symbol:<6} | {action:<4} | "
                    f"retcode={retcode} {text}"
                )
                return

            # Give MT5 a brief moment to expose the resulting position.
            position = None
            for _ in range(10):
                position = self._find_managed_position(
                    symbol,
                    client_tag=tag,
                )
                if position is not None:
                    break
                time.sleep(0.1)

            result_price = _finite_float(
                getattr(result, "price", None)
            )
            order_ticket = int(
                getattr(result, "order", 0) or 0
            )
            deal_ticket = int(
                getattr(result, "deal", 0) or 0
            )

            position_ticket = (
                int(getattr(position, "ticket", 0) or 0)
                if position is not None
                else order_ticket
            )

            actual_volume = (
                _finite_float(
                    getattr(position, "volume", None)
                )
                if position is not None
                else _finite_float(
                    getattr(result, "volume", None)
                )
            )
            if actual_volume is None or actual_volume <= 0:
                actual_volume = volume

            entry_price = (
                _finite_float(
                    getattr(position, "price_open", None)
                )
                if position is not None
                else result_price
            )

            record.update(
                {
                    "status": "OPEN",
                    "opened_utc": _utc_now(),
                    "volume": actual_volume,
                    "entry_price": entry_price,
                    "request_price": request_price,
                    "result_price": result_price,
                    "position_ticket": position_ticket,
                    "order_ticket": order_ticket,
                    "deal_ticket": deal_ticket,
                    "retcode": int(getattr(result, "retcode", 0) or 0),
                    "retcode_text": str(
                        getattr(result, "comment", "") or ""
                    ),
                }
            )

            self.execution_state["open_positions"][
                decision_id
            ] = dict(record)

            self._save_execution_state()

            self._event(
                "OPENED",
                row=row,
                record=record,
                request_price=request_price,
                result_price=result_price,
                retcode=record["retcode"],
                retcode_text=record["retcode_text"],
                order_ticket=order_ticket,
                deal_ticket=deal_ticket,
                note=(
                    "DEMO market order accepted; "
                    f"scheduled fixed-horizon exit bar={scheduled_exit_bar}"
                ),
            )

            log(
                "PRECISION_DEMO_OPEN | "
                f"symbol={symbol} "
                f"action={action} "
                f"volume={actual_volume} "
                f"position={position_ticket} "
                f"confidence={float(row['confidence']):.6f} "
                f"decision={row['decision_time']}"
            )

            print(
                f">>> DEMO OPEN | {symbol:<6} | {action:<4} | "
                f"vol={actual_volume:g} "
                f"price={entry_price} "
                f"ticket={position_ticket} "
                f"conf={float(row['confidence']):.4f}"
            )

        except Exception as exc:
            record["status"] = "OPEN_BLOCKED"
            record["note"] = str(exc)
            self._save_execution_state()
            self._event(
                "OPEN_BLOCKED",
                row=row,
                record=record,
                note=str(exc),
            )
            print(
                f"XXX DEMO BLOCK | {symbol:<6} | {action:<4} | {exc}"
            )
            log(
                f"ERROR | PRECISION_DEMO_OPEN | "
                f"symbol={symbol} | {exc}"
            )

    # ------------------------------------------------------------------
    # Fixed-horizon close
    # ------------------------------------------------------------------

    def _close_record(
        self,
        decision_id: str,
        record: dict[str, Any],
        *,
        reason: str,
    ) -> bool:
        self._assert_demo_account()

        symbol = str(record["symbol"]).upper()
        tag = str(record.get("client_tag", "") or "")
        position_ticket = int(
            record.get("position_ticket", 0) or 0
        )

        position = self._find_managed_position(
            symbol,
            position_ticket=position_ticket or None,
            client_tag=tag,
        )

        if position is None:
            profit = self._broker_profit_for_position(
                position_ticket or None
            )

            record["status"] = "CLOSED_EXTERNALLY_OR_MISSING"
            record["closed_utc"] = _utc_now()
            record["broker_profit"] = profit
            record["close_reason"] = reason

            self.execution_state["completed"][
                decision_id
            ] = dict(record)
            self.execution_state["open_positions"].pop(
                decision_id,
                None,
            )
            self._save_execution_state()

            self._event(
                "POSITION_MISSING",
                record=record,
                broker_profit=profit,
                note=(
                    "Tracked DEMO position is no longer open. "
                    "It may have been manually closed."
                ),
            )

            print(
                f"<<< DEMO CLOSED ELSEWHERE | {symbol:<6} | "
                f"decision={record.get('decision_time')}"
            )
            return True

        volume = float(getattr(position, "volume", 0.0) or 0.0)
        if volume <= 0:
            raise RuntimeError(
                f"{symbol}: managed position has invalid volume."
            )

        position_type = int(
            getattr(position, "type", -1)
        )
        if position_type == int(self.mt5.POSITION_TYPE_BUY):
            close_type = int(self.mt5.ORDER_TYPE_SELL)
        elif position_type == int(self.mt5.POSITION_TYPE_SELL):
            close_type = int(self.mt5.ORDER_TYPE_BUY)
        else:
            raise RuntimeError(
                f"{symbol}: unsupported MT5 position type {position_type}."
            )

        result, request = self._send_market_request(
            symbol=symbol,
            order_type=close_type,
            volume=volume,
            position_ticket=int(
                getattr(position, "ticket")
            ),
            comment=_short_comment(
                COMMENT_CLOSE_PREFIX,
                tag,
            ),
        )

        request_price = (
            _finite_float(request.get("price"))
            if isinstance(request, dict)
            else None
        )

        if not self._result_ok(result):
            retcode = (
                getattr(result, "retcode", None)
                if result is not None
                else None
            )
            text = (
                str(getattr(result, "comment", "") or "")
                if result is not None
                else str(self.mt5.last_error())
            )

            record["last_close_attempt_utc"] = _utc_now()
            record["last_close_retcode"] = retcode
            record["last_close_error"] = text
            self._save_execution_state()

            self._event(
                "CLOSE_FAILED",
                record=record,
                request_price=request_price,
                retcode=retcode,
                retcode_text=text,
                note=reason,
            )

            print(
                f"XXX DEMO CLOSE FAILED | {symbol:<6} | "
                f"ticket={getattr(position, 'ticket', 0)} | "
                f"retcode={retcode} {text}"
            )
            return False

        result_price = _finite_float(
            getattr(result, "price", None)
        )
        close_order = int(
            getattr(result, "order", 0) or 0
        )
        close_deal = int(
            getattr(result, "deal", 0) or 0
        )

        # Give history a moment to settle.
        profit = None
        for _ in range(8):
            profit = self._broker_profit_for_position(
                int(getattr(position, "ticket"))
            )
            if profit is not None:
                break
            time.sleep(0.1)

        record.update(
            {
                "status": "CLOSED",
                "closed_utc": _utc_now(),
                "closed_utc_broker_event": _utc_now(),
                "exit_price": result_price,
                "close_order_ticket": close_order,
                "close_deal_ticket": close_deal,
                "broker_profit": profit,
                "close_reason": reason,
                "close_retcode": int(
                    getattr(result, "retcode", 0) or 0
                ),
                "close_retcode_text": str(
                    getattr(result, "comment", "") or ""
                ),
            }
        )

        self.execution_state["completed"][
            decision_id
        ] = dict(record)
        self.execution_state["open_positions"].pop(
            decision_id,
            None,
        )
        self._save_execution_state()

        closed_row = {
            "closed_utc": record["closed_utc"],
            **record,
        }
        _append_csv(
            CLOSED_TRADES_CSV,
            closed_row,
            CLOSED_FIELDS,
        )

        self._event(
            "CLOSED",
            record=record,
            request_price=request_price,
            result_price=result_price,
            retcode=record["close_retcode"],
            retcode_text=record["close_retcode_text"],
            order_ticket=close_order,
            deal_ticket=close_deal,
            broker_profit=profit,
            note=reason,
        )

        log(
            "PRECISION_DEMO_CLOSE | "
            f"symbol={symbol} "
            f"position={getattr(position, 'ticket', 0)} "
            f"profit={profit} "
            f"reason={reason}"
        )

        profit_text = (
            f"{profit:+.2f}"
            if profit is not None
            else "n/a"
        )

        print(
            f"<<< DEMO CLOSE | {symbol:<6} | "
            f"ticket={getattr(position, 'ticket', 0)} "
            f"exit={result_price} "
            f"P/L={profit_text} | {reason}"
        )
        return True

    def _close_due_positions(
        self,
        symbol: str,
        latest_closed_bar: pd.Timestamp,
    ) -> None:
        open_map = dict(
            self.execution_state.get(
                "open_positions",
                {},
            )
        )

        for decision_id, record in open_map.items():
            if str(record.get("symbol", "")).upper() != symbol:
                continue

            raw_exit = record.get("scheduled_exit_bar")
            if not raw_exit:
                continue

            try:
                scheduled_exit_bar = pd.Timestamp(raw_exit)
            except Exception:
                continue

            if latest_closed_bar < scheduled_exit_bar:
                continue

            try:
                self._close_record(
                    decision_id,
                    record,
                    reason=(
                        f"MODEL_HORIZON_{HORIZON_MINUTES}M"
                    ),
                )
            except Exception as exc:
                record["last_close_attempt_utc"] = _utc_now()
                record["last_close_error"] = str(exc)
                self.execution_state["open_positions"][
                    decision_id
                ] = record
                self._save_execution_state()

                print(
                    f"XXX DEMO CLOSE BLOCKED | {symbol:<6} | {exc}"
                )
                log(
                    f"ERROR | PRECISION_DEMO_CLOSE | "
                    f"symbol={symbol} | {exc}"
                )

    # ------------------------------------------------------------------
    # Crash/restart reconciliation
    # ------------------------------------------------------------------

    def reconcile_execution_state(self) -> None:
        """Adopt or finalize persisted records without ever duplicating an order."""
        broker_positions = self._managed_positions()

        # Map our currently visible broker positions by their comment tag.
        by_tag: dict[str, Any] = {}
        for position in broker_positions:
            comment = str(
                getattr(position, "comment", "") or ""
            )
            for decision_id, record in self.execution_state[
                "decisions"
            ].items():
                tag = str(record.get("client_tag", "") or "")
                if tag and tag in comment:
                    by_tag[tag] = position

        for decision_id, record in list(
            self.execution_state["decisions"].items()
        ):
            status = str(record.get("status", ""))
            tag = str(record.get("client_tag", "") or "")

            if status in {
                "CLOSED",
                "CLOSED_EXTERNALLY_OR_MISSING",
                "OPEN_FAILED",
                "OPEN_BLOCKED",
                "SKIPPED_SYMBOL_ALREADY_OPEN",
                "SKIPPED_BOT_POSITION_LIMIT",
                "UNCERTAIN_NO_BROKER_POSITION",
            }:
                continue

            position = by_tag.get(tag)

            if position is None:
                position = self._find_managed_position(
                    str(record.get("symbol", "")),
                    position_ticket=int(
                        record.get("position_ticket", 0) or 0
                    ) or None,
                    client_tag=tag,
                )

            if position is not None:
                record.update(
                    {
                        "status": "OPEN",
                        "position_ticket": int(
                            getattr(position, "ticket", 0) or 0
                        ),
                        "volume": _finite_float(
                            getattr(position, "volume", None)
                        ),
                        "entry_price": _finite_float(
                            getattr(position, "price_open", None)
                        ),
                    }
                )
                self.execution_state["open_positions"][
                    decision_id
                ] = dict(record)
            elif status in {"INTENT", "SENDING"}:
                # This is deliberately NOT retried. We cannot prove whether an
                # earlier order_send reached the broker before a crash.
                record["status"] = "UNCERTAIN_NO_BROKER_POSITION"
                record["note"] = (
                    "Previous send state was ambiguous after restart. "
                    "No automatic retry was attempted."
                )

        # Any open record no longer represented at the broker is finalized.
        for decision_id, record in list(
            self.execution_state["open_positions"].items()
        ):
            position = self._find_managed_position(
                str(record.get("symbol", "")),
                position_ticket=int(
                    record.get("position_ticket", 0) or 0
                ) or None,
                client_tag=str(record.get("client_tag", "") or ""),
            )

            if position is None:
                profit = self._broker_profit_for_position(
                    int(record.get("position_ticket", 0) or 0) or None
                )
                record["status"] = "CLOSED_EXTERNALLY_OR_MISSING"
                record["closed_utc"] = _utc_now()
                record["broker_profit"] = profit
                record["close_reason"] = "RECONCILE_POSITION_NOT_OPEN"
                self.execution_state["completed"][
                    decision_id
                ] = dict(record)
                self.execution_state["open_positions"].pop(
                    decision_id,
                    None,
                )

        self._save_execution_state()

    # ------------------------------------------------------------------
    # Integrate with PrecisionLiveEngine
    # ------------------------------------------------------------------

    def _record_decision(
        self,
        row: dict[str, Any],
    ) -> None:
        # Keep the complete signal-only audit exactly as before.
        super()._record_decision(row)

        # Then execute only the model's actual Tradeable=True rows.
        self._execute_tradeable_signal(row)

    def process_symbol(
        self,
        symbol: str,
        *,
        force: bool = False,
    ) -> bool:
        model_state = self.models[symbol]

        latest_time = self._latest_closed_bar_time(
            symbol
        )
        if latest_time is None:
            return False

        # Position closure is checked even when this M5 candle was already
        # processed, which makes restart recovery safe.
        self._close_due_positions(
            symbol,
            latest_time,
        )

        if (
            not force
            and model_state.last_processed_bar is not None
            and latest_time <= model_state.last_processed_bar
        ):
            return False

        history = self._closed_history(symbol)

        self._resolve_pending(
            symbol,
            history,
        )

        row = self._predict_symbol(
            symbol,
            history,
        )

        self._record_decision(row)
        self._write_dashboard_log(row)
        self._print_decision(row)

        model_state.last_processed_bar = pd.Timestamp(
            row["bar_time"]
        )

        self._publish_state(row)
        return True

    def _publish_state(
        self,
        row: dict[str, Any] | None = None,
    ) -> None:
        super()._publish_state(row)

        self.state["execution_enabled"] = True
        self.state["execution_mode"] = "DEMO_ONLY"
        self.state["managed_demo_positions"] = len(
            self.execution_state.get("open_positions", {})
        )
        self.state["demo_execution_state"] = str(
            EXECUTION_STATE_PATH
        )
        _atomic_json(
            Path(self.state_path) if hasattr(self, "state_path") else (
                SYSTEM_ROOT / "artifacts" / "reports" / "precision_live_state.json"
            ),
            self.state,
        )

    def print_startup(self) -> None:
        print()
        print("=" * 78)
        print("TRADEAI PRECISION DEMO AUTO-EXECUTION ENGINE")
        print("=" * 78)
        print("Account mode     : DEMO ONLY - REAL ACCOUNT HARD BLOCKED")
        print("Execution        : ENABLED")
        print("Position size    : BROKER MINIMUM LOT")
        print(
            f"Action gate      : {ACTION_CONFIDENCE:.2f}"
        )
        print(
            f"Move definition  : ±{MIN_MEANINGFUL_MOVE_ATR:.2f} ATR "
            f"over {HORIZON_MINUTES} minutes"
        )
        print(
            f"Exit             : FIXED MODEL HORIZON "
            f"({HORIZON_BARS} M5 bars)"
        )
        print(
            f"Max bot positions: {MAX_BOT_POSITIONS} "
            "(maximum one per symbol)"
        )
        print(
            f"Magic            : {PRECISION_DEMO_MAGIC}"
        )
        print(
            f"Poll interval    : {self.poll_seconds:.2f}s"
        )
        print("Symbols:")

        for symbol in self.symbols:
            model = self.models[symbol].model
            gate = (
                "PASS"
                if model.historical_gate_pass
                else "DISABLED"
            )
            print(
                f"  {symbol:<6} "
                f"historical_gate={gate:<8} "
                f"confidence={model.action_confidence:.2f}"
            )

        print()
        print(
            "HOLD does nothing. A Tradeable=True UP/DOWN signal sends one "
            "minimum-lot DEMO market order."
        )
        print(
            "The engine closes only its own tagged position at the exact "
            f"{HORIZON_MINUTES}-minute model horizon."
        )
        print(
            "USDCNH remains blocked because its historical model gate failed."
        )
        print(
            "Press Ctrl+C to stop the engine. Existing managed DEMO positions "
            "are NOT force-closed on Ctrl+C; restart the engine to continue "
            "their horizon management, or use --close-all."
        )
        print("=" * 78)
        print()

    def run_demo_forever(self) -> None:
        self.print_startup()
        self.connect()

        try:
            preflight = self.demo_preflight()
            self.print_demo_preflight(preflight)
            self.reconcile_execution_state()

            # Do not force/replay the already-seen bar from the old signal-only
            # engine. Automatic execution starts from the next new closed M5.
            for symbol in self.symbols:
                try:
                    self.process_symbol(
                        symbol,
                        force=False,
                    )
                except Exception as exc:
                    print(
                        f"ERROR | {symbol} | {exc}"
                    )
                    log(
                        f"ERROR | PRECISION_DEMO | "
                        f"symbol={symbol} | {exc}"
                    )

            self._publish_state()

            while True:
                any_processed = False

                for symbol in self.symbols:
                    try:
                        processed = self.process_symbol(
                            symbol,
                            force=False,
                        )
                        any_processed = (
                            any_processed
                            or processed
                        )
                    except Exception as exc:
                        print(
                            f"ERROR | {symbol} | {exc}"
                        )
                        log(
                            f"ERROR | PRECISION_DEMO | "
                            f"symbol={symbol} | {exc}"
                        )

                if any_processed:
                    self._publish_state()

                time.sleep(
                    self.poll_seconds
                )

        except KeyboardInterrupt:
            print()
            print("Stop requested.")
            open_count = len(
                self.execution_state.get(
                    "open_positions",
                    {},
                )
            )
            if open_count:
                print(
                    f"WARNING: {open_count} managed DEMO position(s) are "
                    "still open. Restart this engine to continue fixed-horizon "
                    "management, or run --close-all."
                )
        finally:
            self._save_execution_state()
            self._publish_state()
            self.disconnect()

    def close_all_managed_positions(self) -> None:
        self.connect()

        try:
            preflight = self.demo_preflight()
            self.print_demo_preflight(preflight)
            self.reconcile_execution_state()

            open_map = dict(
                self.execution_state.get(
                    "open_positions",
                    {},
                )
            )

            if not open_map:
                print("No managed precision-demo positions are open.")
                return

            for decision_id, record in open_map.items():
                try:
                    self._close_record(
                        decision_id,
                        record,
                        reason="MANUAL_ENGINE_CLOSE_ALL",
                    )
                except Exception as exc:
                    print(
                        f"XXX CLOSE-ALL FAILED | "
                        f"{record.get('symbol')} | {exc}"
                    )
        finally:
            self._save_execution_state()
            self.disconnect()


def _parse_symbols(raw: str) -> list[str]:
    if not raw.strip():
        return list(MODEL_SYMBOLS)

    return [
        part.strip().upper()
        for part in raw.split(",")
        if part.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TradeAI precision DEMO auto-execution engine"
        )
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Verify DEMO account, AutoTrading, symbols and broker minimum "
            "volume without sending any order."
        ),
    )

    parser.add_argument(
        "--close-all",
        action="store_true",
        help=(
            "Close only positions owned by the precision-demo engine magic/tag."
        ),
    )

    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="Seconds between market checks.",
    )

    parser.add_argument(
        "--symbols",
        default="",
        help=(
            "Comma-separated symbol subset. Default: all trained precision symbols."
        ),
    )

    args = parser.parse_args()

    engine = PrecisionDemoEngine(
        symbols=_parse_symbols(args.symbols),
        poll_seconds=args.poll,
    )

    if args.check:
        engine.connect()
        try:
            result = engine.demo_preflight()
            engine.print_demo_preflight(result)
            engine.reconcile_execution_state()
            print("PREFLIGHT PASS - no order was sent.")
        finally:
            engine.disconnect()
        return

    if args.close_all:
        engine.close_all_managed_positions()
        return

    engine.run_demo_forever()


if __name__ == "__main__":
    main()
