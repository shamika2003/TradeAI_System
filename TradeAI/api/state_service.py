from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SYSTEM_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = SYSTEM_ROOT / "artifacts" / "reports"

LIVE_STATE_PATH = REPORT_DIR / "precision_live_state.json"
EXECUTION_STATE_PATH = REPORT_DIR / "precision_demo_execution_state.json"

DEFAULT_MAGIC = 26061055
TELEMETRY_FRESH_SECONDS = 420.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _parse_utc(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_seconds(value: Any) -> float | None:
    dt = _parse_utc(value)
    if dt is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - dt).total_seconds())


def _latest_timestamp(*values: Any) -> str:
    parsed: list[tuple[datetime, str]] = []
    for value in values:
        dt = _parse_utc(value)
        if dt is not None:
            parsed.append((dt, str(value)))
    if not parsed:
        return ""
    parsed.sort(key=lambda x: x[0], reverse=True)
    return parsed[0][1]


def _finite_float(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _ensure_mt5() -> tuple[Any | None, str]:
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as exc:
        return None, f"MetaTrader5 import failed: {exc}"

    try:
        if mt5.terminal_info() is None:
            if not mt5.initialize():
                return None, f"MT5 initialize failed: {mt5.last_error()}"
        return mt5, ""
    except Exception as exc:
        return None, str(exc)


def _magic(execution: dict[str, Any]) -> int:
    try:
        return int(execution.get("magic", DEFAULT_MAGIC) or DEFAULT_MAGIC)
    except Exception:
        return DEFAULT_MAGIC


def _live_and_execution() -> tuple[dict[str, Any], dict[str, Any]]:
    return _read_json(LIVE_STATE_PATH), _read_json(EXECUTION_STATE_PATH)


def _low_balance(
    live: dict[str, Any],
    execution: dict[str, Any],
) -> dict[str, Any]:
    low = live.get("low_balance")
    if not isinstance(low, dict):
        low = execution.get("low_balance")
    return low if isinstance(low, dict) else {}


def _execution_account(execution: dict[str, Any]) -> dict[str, Any]:
    raw = execution.get("account")
    return raw if isinstance(raw, dict) else {}


def _symbols(live: dict[str, Any]) -> dict[str, Any]:
    raw = live.get("symbols")
    return raw if isinstance(raw, dict) else {}


def _dict_field(data: dict[str, Any], key: str) -> dict[str, Any]:
    raw = data.get(key)
    return raw if isinstance(raw, dict) else {}


def _position_side(mt5: Any, raw_type: int) -> str:
    if raw_type == int(getattr(mt5, "POSITION_TYPE_BUY", 0)):
        return "BUY"
    if raw_type == int(getattr(mt5, "POSITION_TYPE_SELL", 1)):
        return "SELL"
    return "UNKNOWN"


def get_positions() -> dict[str, Any]:
    _, execution = _live_and_execution()
    magic = _magic(execution)

    mt5, error = _ensure_mt5()
    if mt5 is None:
        return {
            "connected": False,
            "error": error,
            "magic": magic,
            "count": 0,
            "positions": [],
        }

    try:
        raw_positions = list(mt5.positions_get() or [])
    except Exception as exc:
        return {
            "connected": True,
            "error": str(exc),
            "magic": magic,
            "count": 0,
            "positions": [],
        }

    open_records = _dict_field(execution, "open_positions")
    positions: list[dict[str, Any]] = []

    for pos in raw_positions:
        try:
            if int(getattr(pos, "magic", 0) or 0) != magic:
                continue
        except Exception:
            continue

        ticket = int(getattr(pos, "ticket", 0) or 0)
        decision_meta: dict[str, Any] = {}

        for decision_id, record in open_records.items():
            if not isinstance(record, dict):
                continue
            try:
                record_ticket = int(record.get("position_ticket", 0) or 0)
            except Exception:
                record_ticket = 0
            if ticket and record_ticket == ticket:
                decision_meta = {
                    "decision_id": str(decision_id),
                    "confidence": _finite_float(record.get("confidence")),
                    "decision_time": str(record.get("decision_time", "") or ""),
                    "scheduled_exit_bar": str(
                        record.get("scheduled_exit_bar", "") or ""
                    ),
                }
                break

        opened_utc = ""
        try:
            opened = int(getattr(pos, "time", 0) or 0)
            if opened > 0:
                opened_utc = (
                    datetime.fromtimestamp(opened, tz=timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                )
        except Exception:
            pass

        positions.append(
            {
                "ticket": ticket,
                "symbol": str(getattr(pos, "symbol", "") or "").upper(),
                "side": _position_side(
                    mt5, int(getattr(pos, "type", -1))
                ),
                "volume": _finite_float(getattr(pos, "volume", None)),
                "entry_price": _finite_float(getattr(pos, "price_open", None)),
                "current_price": _finite_float(
                    getattr(pos, "price_current", None)
                ),
                "profit": _finite_float(getattr(pos, "profit", None)),
                "swap": _finite_float(getattr(pos, "swap", None)),
                "stop_loss": _finite_float(getattr(pos, "sl", None)),
                "take_profit": _finite_float(getattr(pos, "tp", None)),
                "opened_utc": opened_utc,
                "comment": str(getattr(pos, "comment", "") or ""),
                **decision_meta,
            }
        )

    return {
        "connected": True,
        "error": "",
        "magic": magic,
        "count": len(positions),
        "positions": positions,
    }


def get_health() -> dict[str, Any]:
    live, execution = _live_and_execution()

    latest_utc = _latest_timestamp(
        live.get("updated_utc"),
        execution.get("updated_utc"),
    )
    age = _age_seconds(latest_utc)
    fresh = age is not None and age <= TELEMETRY_FRESH_SECONDS

    mt5, mt5_error = _ensure_mt5()

    if not live and not execution:
        state = "NO_TELEMETRY"
    elif fresh:
        state = "ACTIVE_TELEMETRY"
    else:
        state = "STALE_TELEMETRY"

    return {
        "service": "TradeAI Local API",
        "version": "4.0.0",
        "api": "ONLINE",
        "read_only": True,
        "timestamp_utc": _utc_now(),
        "tradeai": {
            "engine": "PRECISION_DEMO",
            "state": state,
            "mode": str(
                live.get("execution_mode")
                or execution.get("mode")
                or "DEMO_ONLY"
            ).upper(),
            "execution_enabled": bool(
                live.get(
                    "execution_enabled",
                    execution.get("execution_enabled", False),
                )
            ),
            "last_telemetry_utc": latest_utc,
            "telemetry_age_seconds": age,
            "telemetry_fresh": fresh,
        },
        "mt5": {
            "connected": mt5 is not None,
            "error": mt5_error,
        },
    }


def get_account() -> dict[str, Any]:
    live, execution = _live_and_execution()
    low = _low_balance(live, execution)
    persisted = _execution_account(execution)

    mt5, mt5_error = _ensure_mt5()
    broker: dict[str, Any] = {
        "connected": mt5 is not None,
        "error": mt5_error,
        "verified_demo": bool(persisted.get("verified_demo", False)),
        "currency": str(persisted.get("currency", "") or ""),
        "balance": _finite_float(persisted.get("balance")),
        "equity": _finite_float(persisted.get("equity")),
        "free_margin": None,
        "server": str(persisted.get("server", "") or ""),
        "login": persisted.get("login"),
    }

    if mt5 is not None:
        try:
            info = mt5.account_info()
            if info is not None:
                broker.update(
                    {
                        "currency": str(getattr(info, "currency", "") or ""),
                        "balance": _finite_float(getattr(info, "balance", None)),
                        "equity": _finite_float(getattr(info, "equity", None)),
                        "free_margin": _finite_float(
                            getattr(info, "margin_free", None)
                        ),
                        "server": str(getattr(info, "server", "") or ""),
                        "login": getattr(info, "login", None),
                    }
                )
        except Exception:
            pass

    return {
        "timestamp_utc": _utc_now(),
        "broker": broker,
        "virtual": {
            "start_capital": _finite_float(low.get("start_capital")),
            "realized_profit": _finite_float(low.get("realized_profit")),
            "floating_profit": _finite_float(low.get("floating_profit")),
            "balance": _finite_float(low.get("balance")),
            "equity": _finite_float(low.get("equity")),
            "reserved_margin": _finite_float(low.get("reserved_margin")),
            "free_margin": _finite_float(low.get("free_margin")),
            "return_percent": _finite_float(low.get("return_percent")),
        },
    }


def get_signals(symbol: str | None = None) -> dict[str, Any]:
    live, _ = _live_and_execution()
    symbols = _symbols(live)

    requested = str(symbol or "").strip().upper()
    if requested:
        item = symbols.get(requested)
        selected = {requested: item} if isinstance(item, dict) else {}
    else:
        selected = {
            str(k).upper(): v
            for k, v in symbols.items()
            if isinstance(v, dict)
        }

    return {
        "timestamp_utc": _utc_now(),
        "model": str(live.get("model", "") or ""),
        "timeframe": str(live.get("timeframe", "") or ""),
        "horizon_minutes": live.get("horizon_minutes"),
        "action_confidence": live.get("action_confidence"),
        "count": len(selected),
        "signals": selected,
    }


def get_symbols() -> dict[str, Any]:
    live, _ = _live_and_execution()
    symbols = _symbols(live)

    result = []
    for symbol, data in symbols.items():
        if not isinstance(data, dict):
            continue
        result.append(
            {
                "symbol": str(symbol).upper(),
                "historical_gate_pass": bool(
                    data.get("historical_gate_pass", False)
                ),
                "tradeable": bool(data.get("tradeable", False)),
                "signal": str(data.get("signal", "") or ""),
                "action": str(data.get("action", "") or ""),
                "confidence": _finite_float(data.get("confidence")),
                "action_confidence": _finite_float(
                    data.get("action_confidence")
                ),
                "bid": _finite_float(data.get("bid")),
                "ask": _finite_float(data.get("ask")),
                "spread_pips": _finite_float(data.get("spread_pips")),
                "updated_utc": str(data.get("updated_utc", "") or ""),
            }
        )

    return {
        "timestamp_utc": _utc_now(),
        "count": len(result),
        "symbols": result,
    }


def _trade_sort_key(record: dict[str, Any]) -> datetime:
    for key in ("closed_utc", "opened_utc", "created_utc", "decision_time"):
        dt = _parse_utc(record.get(key))
        if dt is not None:
            return dt
    return datetime.min.replace(tzinfo=timezone.utc)


def _clean_trade(record: dict[str, Any]) -> dict[str, Any]:
    allowed = (
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
        "status",
        "created_utc",
        "opened_utc",
        "closed_utc",
        "closed_utc_broker_event",
        "scheduled_exit_bar",
        "volume",
        "entry_price",
        "exit_price",
        "request_price",
        "result_price",
        "position_ticket",
        "order_ticket",
        "deal_ticket",
        "close_order_ticket",
        "close_deal_ticket",
        "broker_profit",
        "spread_pips",
        "close_reason",
        "retcode",
        "retcode_text",
        "note",
    )
    out: dict[str, Any] = {}
    for key in allowed:
        if key in record:
            value = record.get(key)
            if key in {
                "confidence",
                "p_down",
                "p_hold",
                "p_up",
                "volume",
                "entry_price",
                "exit_price",
                "request_price",
                "result_price",
                "broker_profit",
                "spread_pips",
            }:
                value = _finite_float(value)
            out[key] = value
    return out


def get_trades(
    limit: int = 50,
    symbol: str | None = None,
    include_open: bool = True,
) -> dict[str, Any]:
    _, execution = _live_and_execution()

    completed = _dict_field(execution, "completed")
    open_records = _dict_field(execution, "open_positions")

    requested = str(symbol or "").strip().upper()
    rows: list[dict[str, Any]] = []

    for decision_id, raw in completed.items():
        if not isinstance(raw, dict):
            continue
        record = dict(raw)
        record.setdefault("decision_id", decision_id)
        record["_trade_state"] = "COMPLETED"
        if requested and str(record.get("symbol", "")).upper() != requested:
            continue
        rows.append(record)

    if include_open:
        for decision_id, raw in open_records.items():
            if not isinstance(raw, dict):
                continue
            record = dict(raw)
            record.setdefault("decision_id", decision_id)
            record["_trade_state"] = "OPEN"
            if requested and str(record.get("symbol", "")).upper() != requested:
                continue
            rows.append(record)

    rows.sort(key=_trade_sort_key, reverse=True)
    capped = max(1, min(int(limit), 500))
    rows = rows[:capped]

    clean = []
    for record in rows:
        item = _clean_trade(record)
        item["trade_state"] = record.get("_trade_state")
        clean.append(item)

    return {
        "timestamp_utc": _utc_now(),
        "symbol_filter": requested or None,
        "include_open": bool(include_open),
        "limit": capped,
        "count": len(clean),
        "trades": clean,
    }


def get_performance() -> dict[str, Any]:
    live, execution = _live_and_execution()
    low = _low_balance(live, execution)
    completed = _dict_field(execution, "completed")

    profits: list[float] = []
    missing_profit = 0
    by_symbol: dict[str, dict[str, Any]] = {}

    for raw in completed.values():
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("symbol", "UNKNOWN") or "UNKNOWN").upper()
        value = _finite_float(raw.get("broker_profit"))

        stats = by_symbol.setdefault(
            symbol,
            {
                "completed_trades": 0,
                "profit_known_trades": 0,
                "wins": 0,
                "losses": 0,
                "breakeven": 0,
                "net_profit": 0.0,
            },
        )
        stats["completed_trades"] += 1

        if value is None:
            missing_profit += 1
            continue

        profits.append(value)
        stats["profit_known_trades"] += 1
        stats["net_profit"] += value
        if value > 0:
            stats["wins"] += 1
        elif value < 0:
            stats["losses"] += 1
        else:
            stats["breakeven"] += 1

    wins = sum(1 for p in profits if p > 0)
    losses = sum(1 for p in profits if p < 0)
    breakeven = sum(1 for p in profits if p == 0)
    gross_profit = sum(p for p in profits if p > 0)
    gross_loss_abs = abs(sum(p for p in profits if p < 0))
    net = sum(profits)

    if gross_loss_abs > 0:
        profit_factor: float | None = gross_profit / gross_loss_abs
    elif gross_profit > 0:
        profit_factor = None  # mathematically infinite; JSON keeps this explicit
    else:
        profit_factor = 0.0

    known_count = len(profits)
    win_rate = (wins / known_count * 100.0) if known_count else 0.0

    for stats in by_symbol.values():
        known = int(stats["profit_known_trades"])
        stats["win_rate_percent"] = (
            float(stats["wins"]) / known * 100.0 if known else 0.0
        )
        stats["net_profit"] = float(stats["net_profit"])

    return {
        "timestamp_utc": _utc_now(),
        "completed_trades": len(completed),
        "profit_known_trades": known_count,
        "profit_missing_trades": missing_profit,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "win_rate_percent": win_rate,
        "gross_profit": gross_profit,
        "gross_loss": -gross_loss_abs,
        "net_broker_profit_from_completed_records": net,
        "profit_factor": profit_factor,
        "profit_factor_note": (
            "infinite (no losing trades)"
            if profit_factor is None and gross_profit > 0
            else ""
        ),
        "virtual_account": {
            "start_capital": _finite_float(low.get("start_capital")),
            "realized_profit": _finite_float(low.get("realized_profit")),
            "floating_profit": _finite_float(low.get("floating_profit")),
            "balance": _finite_float(low.get("balance")),
            "equity": _finite_float(low.get("equity")),
            "return_percent": _finite_float(low.get("return_percent")),
        },
        "forward_audit": (
            live.get("forward_audit", {})
            if isinstance(live.get("forward_audit"), dict)
            else {}
        ),
        "by_symbol": by_symbol,
    }


def get_status() -> dict[str, Any]:
    live, execution = _live_and_execution()
    low = _low_balance(live, execution)
    account = _execution_account(execution)
    symbols = _symbols(live)
    decisions = _dict_field(execution, "decisions")
    completed = _dict_field(execution, "completed")
    open_records = _dict_field(execution, "open_positions")
    positions = get_positions()

    latest_utc = _latest_timestamp(
        live.get("updated_utc"),
        execution.get("updated_utc"),
    )
    age = _age_seconds(latest_utc)

    return {
        "service": "TradeAI",
        "api_version": "4.0.0",
        "read_only": True,
        "timestamp_utc": _utc_now(),
        "engine": {
            "name": "PRECISION_DEMO",
            "mode": str(
                live.get("execution_mode")
                or execution.get("mode")
                or "DEMO_ONLY"
            ).upper(),
            "execution_enabled": bool(
                live.get(
                    "execution_enabled",
                    execution.get("execution_enabled", False),
                )
            ),
            "model": str(live.get("model", "") or ""),
            "timeframe": str(live.get("timeframe", "") or ""),
            "horizon_minutes": live.get("horizon_minutes"),
            "action_confidence": live.get("action_confidence"),
            "last_telemetry_utc": latest_utc,
            "telemetry_age_seconds": age,
            "telemetry_fresh": (
                age is not None and age <= TELEMETRY_FRESH_SECONDS
            ),
        },
        "broker_account": {
            "verified_demo": bool(account.get("verified_demo", False)),
            "currency": str(account.get("currency", "") or ""),
            "balance": _finite_float(account.get("balance")),
            "equity": _finite_float(account.get("equity")),
            "server": str(account.get("server", "") or ""),
        },
        "virtual_account": {
            "start_capital": _finite_float(low.get("start_capital")),
            "realized_profit": _finite_float(low.get("realized_profit")),
            "floating_profit": _finite_float(low.get("floating_profit")),
            "balance": _finite_float(low.get("balance")),
            "equity": _finite_float(low.get("equity")),
            "reserved_margin": _finite_float(low.get("reserved_margin")),
            "free_margin": _finite_float(low.get("free_margin")),
            "return_percent": _finite_float(low.get("return_percent")),
        },
        "trading": {
            "managed_open_positions": int(positions["count"]),
            "persisted_open_records": len(open_records),
            "completed_trades": len(completed),
            "decisions": len(decisions),
            "pending_forward_signals": int(
                live.get("pending_forward_signals", 0) or 0
            ),
            "max_bot_positions": int(
                execution.get("max_bot_positions", 0) or 0
            ),
            "magic": int(positions["magic"]),
        },
        "forward_audit": (
            live.get("forward_audit", {})
            if isinstance(live.get("forward_audit"), dict)
            else {}
        ),
        "symbols": symbols,
        "mt5": {
            "connected": bool(positions["connected"]),
            "error": str(positions["error"] or ""),
        },
    }
