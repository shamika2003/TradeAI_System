# filename: shared/tradeai_core/demo_forward.py

from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


DEMO_FORWARD_LEDGER_VERSION = "tradeai_demo_forward_v1_20260907"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(value: datetime | None = None) -> str:
    value = value or _utc_now()
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except Exception:
        return default


class DemoForwardLedger:
    """Persistent shadow-account ledger for the MT5 demo forward gate.

    The broker account remains the execution source of truth.  This ledger is
    the risk-capital source of truth, allowing a large demo account to be used
    while proving whether a small virtual account (for example $20) is viable.
    """

    def __init__(
        self,
        path,
        initial_balance: float,
        min_trades: int = 100,
        max_dd_percent: float = 12.0,
        min_profit_factor: float = 1.20,
        min_return_percent: float = 0.0,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.initial_balance = float(initial_balance)
        self.min_trades = int(min_trades)
        self.max_dd_percent = float(max_dd_percent)
        self.min_profit_factor = float(min_profit_factor)
        self.min_return_percent = float(min_return_percent)

        if self.initial_balance <= 0:
            raise ValueError("initial_balance must be positive")
        if self.min_trades <= 0:
            raise ValueError("min_trades must be positive")

        self.state = self._load_or_create()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "version": DEMO_FORWARD_LEDGER_VERSION,
            "created_utc": _iso_utc(),
            "initial_balance": self.initial_balance,
            "virtual_balance": self.initial_balance,
            "high_watermark": self.initial_balance,
            "max_drawdown_percent": 0.0,
            "closed_trades": 0,
            "wins": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "seen_trade_keys": [],
            "by_symbol": {},
            "daily": {},
            "rejections": {},
            "last_trade_utc": None,
        }

    def _load_or_create(self) -> Dict[str, Any]:
        if not self.path.exists():
            state = self._default_state()
            self._save_state(state)
            return state

        data = json.loads(self.path.read_text(encoding="utf-8"))
        if data.get("version") != DEMO_FORWARD_LEDGER_VERSION:
            raise RuntimeError(
                f"Unsupported demo-forward ledger version: {data.get('version')}"
            )

        stored_initial = _as_float(data.get("initial_balance"), 0.0)
        if abs(stored_initial - self.initial_balance) > 1e-9:
            raise RuntimeError(
                "Demo-forward capital does not match the existing ledger. "
                f"Existing=${stored_initial:.2f}, configured=${self.initial_balance:.2f}. "
                "Do not silently reset a forward-test account."
            )
        return data

    def _save_state(self, state: Dict[str, Any] | None = None) -> None:
        state = self.state if state is None else state
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(state, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)

    @staticmethod
    def _trade_key(trade: Dict[str, Any]) -> str:
        return "|".join(
            [
                str(trade.get("ticket", "")),
                str(trade.get("symbol", "")),
                str(trade.get("open_time", "")),
                str(trade.get("close_time", "")),
                f"{_as_float(trade.get('profit')):.8f}",
            ]
        )

    def record_trade(self, trade: Dict[str, Any]) -> bool:
        if not isinstance(trade, dict):
            return False

        key = self._trade_key(trade)
        seen = set(self.state.get("seen_trade_keys", []))
        if key in seen:
            return False

        profit = _as_float(trade.get("profit"), 0.0)
        symbol = str(trade.get("symbol") or "UNKNOWN")
        before = _as_float(self.state.get("virtual_balance"), self.initial_balance)
        after = before + profit

        self.state["virtual_balance"] = after
        self.state["net_profit"] = _as_float(self.state.get("net_profit")) + profit
        self.state["closed_trades"] = int(self.state.get("closed_trades", 0)) + 1
        if profit > 0:
            self.state["wins"] = int(self.state.get("wins", 0)) + 1
            self.state["gross_profit"] = _as_float(self.state.get("gross_profit")) + profit
        elif profit < 0:
            self.state["gross_loss"] = _as_float(self.state.get("gross_loss")) + abs(profit)

        peak = max(_as_float(self.state.get("high_watermark"), self.initial_balance), after)
        self.state["high_watermark"] = peak
        if peak > 0:
            dd = max(0.0, (peak - after) / peak * 100.0)
            self.state["max_drawdown_percent"] = max(
                _as_float(self.state.get("max_drawdown_percent")), dd
            )

        day = _utc_now().date().isoformat()
        daily = self.state.setdefault("daily", {})
        if day not in daily:
            daily[day] = {
                "start_balance": before,
                "net_profit": 0.0,
                "trades": 0,
            }
        daily[day]["net_profit"] = _as_float(daily[day].get("net_profit")) + profit
        daily[day]["trades"] = int(daily[day].get("trades", 0)) + 1

        by_symbol = self.state.setdefault("by_symbol", {})
        bucket = by_symbol.setdefault(
            symbol,
            {"trades": 0, "wins": 0, "net_profit": 0.0},
        )
        bucket["trades"] = int(bucket.get("trades", 0)) + 1
        if profit > 0:
            bucket["wins"] = int(bucket.get("wins", 0)) + 1
        bucket["net_profit"] = _as_float(bucket.get("net_profit")) + profit

        seen.add(key)
        self.state["seen_trade_keys"] = sorted(seen)
        self.state["last_trade_utc"] = _iso_utc()
        self._save_state()
        return True

    def record_rejection(self, reason: str) -> None:
        reason = str(reason or "UNKNOWN")
        rej = self.state.setdefault("rejections", {})
        rej[reason] = int(rej.get(reason, 0)) + 1
        self._save_state()

    @property
    def virtual_balance(self) -> float:
        return _as_float(self.state.get("virtual_balance"), self.initial_balance)

    @property
    def high_watermark(self) -> float:
        return _as_float(self.state.get("high_watermark"), self.initial_balance)

    def daily_loss_percent(self) -> float:
        day = _utc_now().date().isoformat()
        bucket = self.state.get("daily", {}).get(day)
        if not bucket:
            return 0.0
        start = _as_float(bucket.get("start_balance"), self.virtual_balance)
        net = _as_float(bucket.get("net_profit"), 0.0)
        if start <= 0 or net >= 0:
            return 0.0
        return abs(net) / start * 100.0

    def summary(self) -> Dict[str, Any]:
        trades = int(self.state.get("closed_trades", 0))
        wins = int(self.state.get("wins", 0))
        gp = _as_float(self.state.get("gross_profit"), 0.0)
        gl = _as_float(self.state.get("gross_loss"), 0.0)
        pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)
        balance = self.virtual_balance
        ret = (balance / self.initial_balance - 1.0) * 100.0
        dd = _as_float(self.state.get("max_drawdown_percent"), 0.0)

        passed = (
            trades >= self.min_trades
            and ret > self.min_return_percent
            and pf >= self.min_profit_factor
            and dd <= self.max_dd_percent
        )

        return {
            "version": DEMO_FORWARD_LEDGER_VERSION,
            "initial_balance": self.initial_balance,
            "virtual_balance": balance,
            "net_profit": balance - self.initial_balance,
            "return_percent": ret,
            "closed_trades": trades,
            "wins": wins,
            "win_rate_percent": (wins / trades * 100.0) if trades else 0.0,
            "profit_factor": pf,
            "max_drawdown_percent": dd,
            "daily_loss_percent": self.daily_loss_percent(),
            "min_trades_required": self.min_trades,
            "max_dd_percent_allowed": self.max_dd_percent,
            "min_profit_factor_required": self.min_profit_factor,
            "min_return_percent_required": self.min_return_percent,
            "by_symbol": self.state.get("by_symbol", {}),
            "rejections": self.state.get("rejections", {}),
            "accepted": passed,
            "created_utc": self.state.get("created_utc"),
            "last_trade_utc": self.state.get("last_trade_utc"),
        }
