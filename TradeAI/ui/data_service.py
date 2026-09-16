from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRADEAI_DIR = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = TRADEAI_DIR.parent
ARTIFACTS_DIR = SYSTEM_ROOT / "artifacts"
REPORT_DIR = ARTIFACTS_DIR / "reports"
BOT_LOG = TRADEAI_DIR / "logs" / "bot.log"
TRADE_HISTORY = TRADEAI_DIR / "logs" / "trade_history.csv"

RUNTIME_STATUS = REPORT_DIR / "runtime_account_status.json"
DEMO_STATE = REPORT_DIR / "demo_forward_state.json"
DECISION_POLICY = REPORT_DIR / "stage5_decision_policy.json"
RISK_POLICY = REPORT_DIR / "stage5_risk_policy.json"
RISK_VALIDATION = REPORT_DIR / "stage5_risk_validation.json"
PRIMARY_TRADES = REPORT_DIR / "stage5_primary_trades.csv"
RUNTIME_BACKTEST_SUMMARY = REPORT_DIR / "runtime_backtest_summary.json"
RUNTIME_BACKTEST_TRADES = REPORT_DIR / "runtime_backtest_trades.csv"
BACKTEST_HISTORY_DIR = REPORT_DIR / "backtest_runs"
MODEL_PATH = ARTIFACTS_DIR / "models" / "trading_model.pkl"
DATASET_PATH = ARTIFACTS_DIR / "datasets" / "market_dataset.csv"
ENGINE_STATUS_PATH = REPORT_DIR / "tradeai_engine_status.json"

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCNH"]


@dataclass
class SymbolQuote:
    symbol: str
    bid: float | None = None
    ask: float | None = None
    spread_pips: float | None = None
    change_pct: float | None = None


@dataclass
class OpenPosition:
    symbol: str
    side: str
    volume: float
    entry: float
    current: float
    profit: float
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_to_sl: float | None = None
    reward_to_tp: float | None = None
    reward_risk: float | None = None
    age_minutes: float | None = None


@dataclass
class RuntimeSnapshot:
    mt5_connected: bool = False
    terminal_name: str = "MT5"
    broker_balance: float = 0.0
    broker_equity: float = 0.0
    free_margin: float = 0.0
    currency: str = "USD"
    virtual_balance: float = 0.0
    initial_virtual_balance: float = 0.0
    net_profit: float = 0.0
    drawdown_percent: float = 0.0
    closed_trades: int = 0
    wins: int = 0
    open_positions: int = 0
    positions: list[OpenPosition] = field(default_factory=list)
    mode: str = "UNKNOWN"
    quotes: list[SymbolQuote] = field(default_factory=list)
    candles: list[tuple[float, float, float, float, float]] = field(default_factory=list)
    chart_symbol: str = "EURUSD"
    chart_bid: float | None = None
    chart_ask: float | None = None
    chart_spread_pips: float | None = None
    forward_equity_curve: list[float] = field(default_factory=list)
    floating_profit: float = 0.0
    current_risk_money: float = 0.0
    current_reward_money: float = 0.0
    risk_usage_percent: float = 0.0
    broker_drawdown_percent: float = 0.0
    daily_loss_percent: float = 0.0
    updated_at: str = ""


class TradeAIDataService:
    """Read-only bridge between the dashboard, TradeAI artifacts, logs, and MT5."""

    def __init__(self) -> None:
        self.mt5 = None
        self._mt5_import_error: str | None = None
        self.symbols = self._read_symbols_from_settings() or list(DEFAULT_SYMBOLS)
        self._load_mt5_module()

    @staticmethod
    def _safe_json(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _read_symbols_from_settings(self) -> list[str]:
        settings_path = TRADEAI_DIR / "config" / "settings.py"
        try:
            text = settings_path.read_text(encoding="utf-8")
            match = re.search(r"SYMBOLS\s*=\s*\[(.*?)\]", text, re.S)
            if not match:
                return []
            return re.findall(r"['\"]([A-Z0-9._-]+)['\"]", match.group(1))
        except Exception:
            return []

    def _read_mode_from_settings(self) -> str:
        settings_path = TRADEAI_DIR / "config" / "settings.py"
        try:
            text = settings_path.read_text(encoding="utf-8")
            match = re.search(r"^MODE\s*=\s*['\"]([^'\"]+)['\"]", text, re.M)
            return match.group(1) if match else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    def _load_mt5_module(self) -> None:
        try:
            import MetaTrader5 as mt5  # type: ignore
            self.mt5 = mt5
        except Exception as exc:
            self._mt5_import_error = str(exc)
            self.mt5 = None

    def _initialize_mt5(self) -> bool:
        if self.mt5 is None:
            return False
        try:
            if self.mt5.terminal_info() is not None:
                return True
            return bool(self.mt5.initialize())
        except Exception:
            return False

    def get_snapshot(self, chart_symbol: str = "EURUSD") -> RuntimeSnapshot:
        runtime = self._safe_json(RUNTIME_STATUS)
        demo = self._safe_json(DEMO_STATE)
        broker = runtime.get("broker", {}) if isinstance(runtime.get("broker"), dict) else {}
        shadow = runtime.get("shadow", {}) if isinstance(runtime.get("shadow"), dict) else {}
        snapshot = RuntimeSnapshot(
            broker_balance=float(broker.get("balance", 0.0) or 0.0),
            broker_equity=float(broker.get("equity", 0.0) or 0.0),
            free_margin=float(broker.get("free_margin", 0.0) or 0.0),
            currency=str(broker.get("currency", "USD") or "USD"),
            virtual_balance=float(shadow.get("virtual_balance", demo.get("virtual_balance", 0.0)) or 0.0),
            initial_virtual_balance=float(shadow.get("initial_balance", demo.get("initial_balance", 0.0)) or 0.0),
            net_profit=float(shadow.get("net_profit", demo.get("net_profit", 0.0)) or 0.0),
            drawdown_percent=float(shadow.get("max_drawdown_percent", demo.get("max_drawdown_percent", 0.0)) or 0.0),
            closed_trades=int(shadow.get("closed_trades", demo.get("closed_trades", 0)) or 0),
            wins=int(demo.get("wins", 0) or 0),
            mode=str(runtime.get("mode") or self._read_mode_from_settings()),
            chart_symbol=chart_symbol,
            updated_at=str(runtime.get("updated_at", "")),
        )
        snapshot.daily_loss_percent = self._daily_loss_percent(demo)

        # BACKTEST is intentionally offline. The dashboard snapshot worker must
        # not keep opening/querying MT5 while an offline replay is running.
        engine = self._safe_json(ENGINE_STATUS_PATH)
        engine_mode = str(engine.get("mode", "") or "").upper()
        engine_state = str(engine.get("state", "") or "").upper()
        if engine_mode:
            snapshot.mode = engine_mode
        if engine_mode == "BACKTEST" and engine_state in {"STARTING", "RUNNING", "PAUSED", "STOPPING"}:
            snapshot.mt5_connected = False
            snapshot.terminal_name = "Offline Replay"
            snapshot.forward_equity_curve = self.load_forward_equity_curve(
                snapshot.initial_virtual_balance, snapshot.virtual_balance, demo
            )
            return snapshot

        connected = self._initialize_mt5()
        snapshot.mt5_connected = connected
        if not connected or self.mt5 is None:
            return snapshot
        try:
            info = self.mt5.account_info()
            if info is not None:
                snapshot.broker_balance = float(info.balance)
                snapshot.broker_equity = float(info.equity)
                snapshot.free_margin = float(info.margin_free)
                snapshot.currency = str(info.currency)
                if snapshot.broker_balance > 0:
                    snapshot.broker_drawdown_percent = max(
                        0.0,
                        (snapshot.broker_balance - snapshot.broker_equity) / snapshot.broker_balance * 100.0,
                    )
        except Exception:
            pass
        try:
            snapshot.terminal_name = "MetaTrader 5" if self.mt5.terminal_info() is not None else "MT5"
        except Exception:
            pass
        try:
            positions = self.mt5.positions_get()
            snapshot.open_positions = len(positions) if positions is not None else 0
            if positions is not None:
                for pos in positions:
                    side_value = int(getattr(pos, "type", 0))
                    side = "BUY" if side_value == 0 else "SELL"
                    sl = float(getattr(pos, "sl", 0.0) or 0.0)
                    tp = float(getattr(pos, "tp", 0.0) or 0.0)
                    symbol = str(getattr(pos, "symbol", "?"))
                    volume = float(getattr(pos, "volume", 0.0) or 0.0)
                    entry = float(getattr(pos, "price_open", 0.0) or 0.0)
                    current = float(getattr(pos, "price_current", 0.0) or 0.0)
                    profit = float(getattr(pos, "profit", 0.0) or 0.0)
                    risk_to_sl, reward_to_tp = self._position_risk_reward(
                        symbol=symbol,
                        side=side,
                        volume=volume,
                        entry=entry,
                        stop_loss=sl if sl > 0 else None,
                        take_profit=tp if tp > 0 else None,
                    )
                    rr = None
                    if risk_to_sl is not None and risk_to_sl > 0 and reward_to_tp is not None:
                        rr = reward_to_tp / risk_to_sl
                    age_minutes = None
                    try:
                        opened = float(getattr(pos, "time", 0.0) or 0.0)
                        if opened > 0:
                            age_minutes = max(0.0, (datetime.now().timestamp() - opened) / 60.0)
                    except Exception:
                        pass
                    snapshot.positions.append(
                        OpenPosition(
                            symbol=symbol, side=side, volume=volume, entry=entry, current=current, profit=profit,
                            stop_loss=sl if sl > 0 else None, take_profit=tp if tp > 0 else None,
                            risk_to_sl=risk_to_sl, reward_to_tp=reward_to_tp, reward_risk=rr, age_minutes=age_minutes,
                        )
                    )
        except Exception:
            snapshot.open_positions = 0
            snapshot.positions = []
        snapshot.floating_profit = sum(float(p.profit or 0.0) for p in snapshot.positions)
        snapshot.current_risk_money = sum(float(p.risk_to_sl or 0.0) for p in snapshot.positions)
        snapshot.current_reward_money = sum(float(p.reward_to_tp or 0.0) for p in snapshot.positions)
        risk_base = snapshot.virtual_balance if snapshot.virtual_balance > 0 else snapshot.broker_equity
        if risk_base > 0:
            snapshot.risk_usage_percent = snapshot.current_risk_money / risk_base * 100.0
        snapshot.quotes = self._get_quotes()
        for quote in snapshot.quotes:
            if quote.symbol == chart_symbol:
                snapshot.chart_bid = quote.bid
                snapshot.chart_ask = quote.ask
                snapshot.chart_spread_pips = quote.spread_pips
                break
        # UI only: include the currently-forming M5 candle so the chart moves
        # with MT5. The trading engine remains unchanged and still evaluates
        # closed candles only.
        snapshot.candles = self._get_candles(chart_symbol)
        snapshot.forward_equity_curve = self.load_forward_equity_curve(
            snapshot.initial_virtual_balance, snapshot.virtual_balance, demo
        )
        return snapshot

    def _daily_loss_percent(self, demo_state: dict[str, Any] | None) -> float:
        """Return today's realized DEMO_FORWARD loss percentage.

        This mirrors the forward ledger calculation used by the trading engine:
        only a negative realized net result for the current UTC day counts as
        daily loss. Positive/flat days report 0.0%.
        """
        state = demo_state if isinstance(demo_state, dict) else {}
        daily = state.get("daily", {}) if isinstance(state.get("daily"), dict) else {}
        day = datetime.now(timezone.utc).date().isoformat()
        bucket = daily.get(day) if isinstance(daily, dict) else None
        if not isinstance(bucket, dict):
            return 0.0
        try:
            start = float(bucket.get("start_balance", state.get("virtual_balance", 0.0)) or 0.0)
            net = float(bucket.get("net_profit", 0.0) or 0.0)
        except Exception:
            return 0.0
        if start <= 0.0 or net >= 0.0:
            return 0.0
        return abs(net) / start * 100.0

    def _position_risk_reward(
        self,
        *,
        symbol: str,
        side: str,
        volume: float,
        entry: float,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> tuple[float | None, float | None]:
        """Estimate money risk/reward with MT5's own contract calculation.

        Using order_calc_profit keeps FX/JPY/CFD contract sizes and account
        currency conversion broker-correct instead of approximating pip values.
        """
        if self.mt5 is None or volume <= 0 or entry <= 0:
            return None, None
        try:
            order_type = (
                self.mt5.ORDER_TYPE_BUY
                if str(side).upper() == "BUY"
                else self.mt5.ORDER_TYPE_SELL
            )
        except Exception:
            return None, None

        risk = None
        reward = None
        if stop_loss is not None and stop_loss > 0:
            try:
                value = self.mt5.order_calc_profit(order_type, symbol, volume, entry, float(stop_loss))
                if value is not None:
                    risk = abs(float(value))
            except Exception:
                risk = None
        if take_profit is not None and take_profit > 0:
            try:
                value = self.mt5.order_calc_profit(order_type, symbol, volume, entry, float(take_profit))
                if value is not None:
                    reward = abs(float(value))
            except Exception:
                reward = None
        return risk, reward

    def _get_quotes(self) -> list[SymbolQuote]:
        if self.mt5 is None:
            return []
        quotes: list[SymbolQuote] = []
        for symbol in self.symbols:
            try:
                info = self.mt5.symbol_info(symbol)
                if info is not None and not info.visible:
                    self.mt5.symbol_select(symbol, True)
                tick = self.mt5.symbol_info_tick(symbol)
                if tick is None:
                    quotes.append(SymbolQuote(symbol=symbol)); continue
                bid, ask = float(tick.bid), float(tick.ask)
                pip = 0.01 if "JPY" in symbol.upper() else 0.0001
                quotes.append(SymbolQuote(symbol=symbol, bid=bid, ask=ask, spread_pips=(ask-bid)/pip if pip else None))
            except Exception:
                quotes.append(SymbolQuote(symbol=symbol))
        return quotes

    def _get_candles(self, symbol: str, count: int = 120) -> list[tuple[float, float, float, float, float]]:
        if self.mt5 is None:
            return []
        try:
            rates = self.mt5.copy_rates_from_pos(symbol, self.mt5.TIMEFRAME_M5, 0, count)
            if rates is None:
                return []
            return [(float(r["time"]), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])) for r in rates]
        except Exception:
            return []


    def load_forward_equity_curve(
        self,
        initial_balance: float,
        current_balance: float,
        demo_state: dict[str, Any] | None = None,
    ) -> list[float]:
        """Rebuild the persistent DEMO_FORWARD equity path from closed trades.

        TradeAI's forward ledger is the capital source of truth. The CSV is used
        only to reconstruct the visual curve; the final point is always forced
        to the ledger's current virtual balance.
        """
        initial = float(initial_balance or 0.0)
        current = float(current_balance or initial)
        if initial <= 0:
            return [current] if current else []

        created = None
        state = demo_state or self._safe_json(DEMO_STATE)
        raw_created = state.get("created_utc") if isinstance(state, dict) else None
        if raw_created:
            try:
                created = datetime.fromisoformat(str(raw_created).replace("Z", "+00:00"))
            except Exception:
                created = None

        curve = [initial]
        balance = initial
        try:
            with TRADE_HISTORY.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                for row in csv.DictReader(handle):
                    if created is not None:
                        raw_close = row.get("close_time")
                        if raw_close:
                            try:
                                close_dt = datetime.fromisoformat(str(raw_close).replace("Z", "+00:00"))
                                # Make a naive timestamp comparable when older logger rows
                                # were written without an offset.
                                if close_dt.tzinfo is None and created.tzinfo is not None:
                                    close_dt = close_dt.replace(tzinfo=created.tzinfo)
                                if close_dt < created:
                                    continue
                            except Exception:
                                pass
                    try:
                        profit = float(row.get("profit", 0.0) or 0.0)
                    except Exception:
                        continue
                    if not math.isfinite(profit):
                        continue
                    balance += profit
                    curve.append(balance)
        except Exception:
            pass

        # The ledger is authoritative and may have a trade not yet reflected in
        # the CSV read, so synchronize the visual endpoint to it.
        if not curve or abs(curve[-1] - current) > 1e-9:
            curve.append(current)
        return curve[-500:]

    def load_decision_policy(self) -> dict[str, Any]:
        data = self._safe_json(DECISION_POLICY)
        return data or self._safe_json(REPORT_DIR / "stage4_decision_policy.json")

    def load_risk_policy(self) -> dict[str, Any]:
        return self._safe_json(RISK_POLICY)

    def load_validation(self) -> dict[str, Any]:
        return self._safe_json(RISK_VALIDATION)

    def model_ready(self) -> bool:
        validation = self.load_validation()
        if "acceptance_pass" in validation:
            return bool(validation.get("acceptance_pass"))
        return bool(self.load_decision_policy())

    def latest_signal_states(self) -> dict[str, dict[str, float | str | bool | None]]:
        """Return the latest production-model state keyed by the real symbol.

        New engine builds emit ``MODEL_STATE`` records that carry the symbol and
        exact policy gates.  A compatibility fallback still understands the old
        generic ``Prediction signal=...`` lines so an existing log remains useful
        before the first new candle arrives.
        """
        policy = self.load_decision_policy().get("symbols", {})
        states: dict[str, dict[str, float | str | bool | None]] = {}
        for symbol in self.symbols:
            cfg = policy.get(symbol, {}) if isinstance(policy, dict) else {}
            states[symbol] = {
                "signal": None,
                "confidence": None,
                "min_confidence": float(cfg.get("min_confidence", 0.0) or 0.0),
                "signal_threshold": float(cfg.get("signal_threshold", 0.0) or 0.0),
                "enabled": bool(cfg.get("enabled", True)),
                "p_buy": None,
                "p_hold": None,
                "p_sell": None,
                "candle": "",
            }

        lines = self.tail_log(1200)
        structured_re = re.compile(
            r"MODEL_STATE\s*\|\s*symbol=(?P<symbol>[A-Z0-9._-]+)\s+"
            r"signal=(?P<signal>[+-]?[0-9.]+)\s+"
            r"confidence=(?P<confidence>[0-9.]+)\s+"
            r"min_confidence=(?P<min_confidence>[0-9.]+)\s+"
            r"signal_threshold=(?P<signal_threshold>[0-9.]+)\s+"
            r"enabled=(?P<enabled>[01])\s+"
            r"p_buy=(?P<p_buy>[0-9.]+)\s+p_hold=(?P<p_hold>[0-9.]+)\s+p_sell=(?P<p_sell>[0-9.]+)"
            r"(?:\s+candle=(?P<candle>.*))?$"
        )

        seen_structured: set[str] = set()
        for line in reversed(lines):
            m = structured_re.search(line)
            if not m:
                continue
            symbol = m.group("symbol")
            if symbol not in states or symbol in seen_structured:
                continue
            try:
                states[symbol].update({
                    "signal": float(m.group("signal")),
                    "confidence": float(m.group("confidence")),
                    "min_confidence": float(m.group("min_confidence")),
                    "signal_threshold": float(m.group("signal_threshold")),
                    "enabled": m.group("enabled") == "1",
                    "p_buy": float(m.group("p_buy")),
                    "p_hold": float(m.group("p_hold")),
                    "p_sell": float(m.group("p_sell")),
                    "candle": (m.group("candle") or "").strip(),
                })
                seen_structured.add(symbol)
            except (TypeError, ValueError):
                continue
            if len(seen_structured) >= len(states):
                break

        # Compatibility with the pre-structured log format.  Only fill symbols
        # for which no explicit MODEL_STATE has been seen.
        missing = [symbol for symbol in self.symbols if symbol not in seen_structured]
        if missing:
            pred_re = re.compile(r"Prediction\s+signal=([+-]?[0-9.]+)\s+confidence=([0-9.]+)")
            predictions: list[tuple[float, float]] = []
            for line in lines:
                m = pred_re.search(line)
                if m:
                    try:
                        predictions.append((float(m.group(1)), float(m.group(2))))
                    except ValueError:
                        pass
            if len(predictions) >= len(self.symbols):
                recent = predictions[-len(self.symbols):]
                for symbol, (signal, conf) in zip(self.symbols, recent):
                    if symbol in seen_structured:
                        continue
                    states[symbol]["signal"] = signal
                    states[symbol]["confidence"] = conf

        return states

    def load_backtest_summary(self) -> dict[str, Any]:
        path = PRIMARY_TRADES if PRIMARY_TRADES.exists() else REPORT_DIR / "stage4_primary_trades.csv"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            return {}
        if not rows:
            return {}
        profits: list[float] = []
        balances: list[float] = []
        wins = 0; gross_profit = 0.0; gross_loss = 0.0; by_symbol: dict[str, float] = {}
        for row in rows:
            try: profit = float(row.get("profit", 0.0) or 0.0)
            except Exception: profit = 0.0
            profits.append(profit)
            if profit > 0: wins += 1; gross_profit += profit
            elif profit < 0: gross_loss += abs(profit)
            symbol = str(row.get("symbol", "?")); by_symbol[symbol] = by_symbol.get(symbol, 0.0) + profit
            try: balances.append(float(row.get("balance_after", "nan")))
            except Exception: pass
        balances = [x for x in balances if math.isfinite(x)]
        end_balance = balances[-1] if balances else sum(profits)
        try:
            first_profit = float(rows[0].get("profit", 0.0) or 0.0)
            first_after = float(rows[0].get("balance_after", 0.0) or 0.0)
            start_balance = first_after - first_profit
        except Exception:
            start_balance = 0.0
        peak = start_balance; max_dd_pct = 0.0; curve = [start_balance] if start_balance else []
        for balance in balances:
            curve.append(balance); peak = max(peak, balance)
            if peak > 0: max_dd_pct = max(max_dd_pct, ((peak-balance)/peak)*100.0)
        total = len(rows)
        return_pct = ((end_balance/start_balance)-1.0)*100.0 if start_balance > 0 else 0.0
        pf = gross_profit/gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        return {
            "start_balance": start_balance,
            "end_balance": end_balance,
            "net_profit": sum(profits),
            "return_percent": return_pct,
            "total_trades": total,
            "wins": wins,
            "losses": max(0, total-wins),
            "win_rate": (wins/total)*100.0 if total else 0.0,
            "profit_factor": pf,
            "max_drawdown_percent": max_dd_pct,
            "curve": curve,
            "by_symbol": by_symbol,
            "best_symbol": max(by_symbol, key=by_symbol.get) if by_symbol else "—",
            "worst_symbol": min(by_symbol, key=by_symbol.get) if by_symbol else "—",
            "recent_trades": rows[-10:][::-1],
        }

    def load_runtime_backtest_summary(self) -> dict[str, Any]:
        """Load the latest desktop-triggered BACKTEST result.

        ``demo_bot.py`` writes this artifact only after an actual BACKTEST run
        completes.  Stage-5 validation remains a separate fallback source.
        """
        data = self._safe_json(RUNTIME_BACKTEST_SUMMARY)
        if not data:
            return {}

        # JSON cannot represent infinity, so the runtime writer stores an
        # explicit flag when there were gains but no gross losses.  Restore the
        # value expected by the dashboard metric formatter.
        if bool(data.get("profit_factor_infinite", False)):
            data["profit_factor"] = float("inf")

        # Compatibility/recovery: if an older runtime summary does not contain
        # embedded recent trades, rebuild a small tail from the companion CSV.
        if not isinstance(data.get("recent_trades"), list):
            recent: list[dict[str, Any]] = []
            try:
                with RUNTIME_BACKTEST_TRADES.open(
                    "r", encoding="utf-8", errors="replace", newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
                recent = rows[-20:][::-1]
            except Exception:
                recent = []
            data["recent_trades"] = recent

        return data

    def tail_log(self, lines: int = 80) -> list[str]:
        try:
            with BOT_LOG.open("r", encoding="utf-8", errors="replace") as handle:
                data = handle.readlines()
            return [line.rstrip() for line in data[-lines:]]
        except Exception:
            return []

    def report_files(self) -> list[Path]:
        if not REPORT_DIR.exists():
            return []
        return sorted(REPORT_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)

    def system_health(self) -> list[dict[str, Any]]:
        """Cheap diagnostics used by System Control; no trading state is mutated."""
        connected = self._initialize_mt5()
        checks = [
            {"name": "MT5", "ok": connected, "detail": "Terminal connected" if connected else (self._mt5_import_error or "Terminal unavailable")},
            {"name": "MODEL", "ok": MODEL_PATH.exists(), "detail": str(MODEL_PATH.name)},
            {"name": "DATASET", "ok": DATASET_PATH.exists(), "detail": str(DATASET_PATH.name)},
            {"name": "DECISION POLICY", "ok": bool(self.load_decision_policy()), "detail": DECISION_POLICY.name},
            {"name": "RISK POLICY", "ok": bool(self.load_risk_policy()), "detail": RISK_POLICY.name},
            {"name": "BOT LOG", "ok": BOT_LOG.exists(), "detail": BOT_LOG.name},
        ]
        return checks

    def latest_errors(self, limit: int = 8) -> list[str]:
        lines = self.tail_log(700)
        found = [line for line in lines if "ERROR |" in line or "CRITICAL |" in line]
        return found[-max(1, int(limit)):]

    def event_timeline(self, limit: int = 80) -> list[dict[str, str]]:
        events: list[dict[str, str]] = []
        for line in self.tail_log(max(300, limit * 4)):
            category = "ENGINE"
            upper = line.upper()
            if "MODEL_STATE" in upper or "PREDICTION" in upper:
                category = "MODEL"
            elif "RISK" in upper or "REJECT" in upper or "DRAWDOWN" in upper:
                category = "RISK"
            elif "OPEN " in upper or "CLOSE " in upper or "ORDER" in upper or "EXECUTOR" in upper:
                category = "EXECUTION"
            elif "MT5" in upper or "MARKET" in upper or "CANDLE" in upper:
                category = "MARKET"
            elif "ERROR" in upper or "WARNING" in upper:
                category = "SYSTEM"
            # Keep the raw line as the auditable source; category is only UI classification.
            events.append({"category": category, "text": line})
        return events[-max(1, int(limit)):]

    def load_backtest_runs(self, limit: int = 12) -> list[dict[str, Any]]:
        if not BACKTEST_HISTORY_DIR.exists():
            return []
        runs: list[dict[str, Any]] = []
        for path in sorted(BACKTEST_HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            data = self._safe_json(path)
            if data:
                data["_path"] = str(path)
                runs.append(data)
            if len(runs) >= limit:
                break
        return runs

    @staticmethod
    def formatted_time() -> str:
        return datetime.now().strftime("%d %b %Y  ·  %H:%M:%S")
