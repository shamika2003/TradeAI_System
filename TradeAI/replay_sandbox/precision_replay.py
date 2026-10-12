"""Read-only, accelerated replay of TradeAI's ACTUAL Precision M5 demo policy.

Standalone CLI: python -m TradeAI.replay_sandbox.precision_replay --help

No MetaTrader5 import, no access to live engine mutable state, no model fitting.
Uses the original PrecisionDirectionModel and build_precision_features functions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd

from Trade_Bot_Training.precision_config import HORIZON_BARS, HORIZON_MINUTES
from Trade_Bot_Training.precision_feature_engine import build_precision_features
from Trade_Bot_Training.precision_model import PrecisionDirectionModel
from TradeAI.precision_demo_engine_base import MAX_BOT_POSITIONS

from .broker import PaperBroker, Position
from .data import Candle, MT5Archive, parse_time

UTC = timezone.utc
SYSTEM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ROOT = SYSTEM_ROOT / "artifacts" / "models" / "precision_m5"
DEFAULT_OUTPUT = SYSTEM_ROOT / "artifacts" / "replay_sandbox" / "precision_demo_replay"
SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "USDCNH")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(t: datetime) -> str:
    return t.astimezone(UTC).isoformat()


def _to_frame(candles: list[Candle]) -> pd.DataFrame:
    return pd.DataFrame({
        "time": pd.to_datetime([c.time for c in candles], utc=True),
        "open": [c.open for c in candles],
        "high": [c.high for c in candles],
        "low": [c.low for c in candles],
        "close": [c.close for c in candles],
        "tick_volume": [c.tick_volume for c in candles],
        "spread": [c.spread_points for c in candles],
        "real_volume": 0,
    })


def _prepare_decisions(candles: list[Candle], model: PrecisionDirectionModel, start: datetime, end: datetime) -> tuple[dict[datetime, dict], dict[str, int]]:
    """Score only complete M5 bars; decisions may act at the FOLLOWING M5 open.

    Batch scoring is permissible because build_precision_features is causal:
    no negative shift, future-fill, or centered windows. No future OHLC is supplied
    to the model row; times are checked against its decision_time field.
    """
    if not candles:
        return {}, {"scored": 0, "tradeable": 0, "invalid_features": 0}
    feat = build_precision_features(_to_frame(candles))
    cols = model.feature_columns
    missing = [c for c in cols if c not in feat.columns]
    if missing:
        raise ValueError(f"{model.symbol}: missing model features: {missing[:10]}")
    valid = np.isfinite(feat[cols].to_numpy(dtype="float64", copy=False)).all(axis=1)
    decision_times = pd.to_datetime(feat["decision_time"], utc=True)
    active = (decision_times >= pd.Timestamp(start)) & (decision_times < pd.Timestamp(end))
    selected = feat.loc[valid & active].copy()
    if selected.empty:
        return {}, {"scored": 0, "tradeable": 0, "invalid_features": int((~valid & active).sum())}
    pred = model.predict(selected)  # SAME loaded model, gate, class map and confidence as live demo.
    decisions = {}
    for i, r in pred.iterrows():
        t = pd.Timestamp(selected.loc[i, "decision_time"]).to_pydatetime()
        bar_t = pd.Timestamp(selected.loc[i, "time"]).to_pydatetime()
        if t != bar_t + timedelta(minutes=5):
            raise RuntimeError("Model's 5-minute causal entry contract violated")
        if t in decisions:
            raise RuntimeError(f"Duplicate model decision time for {model.symbol} {t}")
        decisions[t] = {
            "bar_time": bar_t,
            "action": "BUY" if r["signal"] == "UP" else "SELL" if r["signal"] == "DOWN" else "HOLD",
            "signal": str(r["signal"]),
            "tradeable": bool(r["tradeable"]),
            "historical_gate_pass": bool(r["historical_gate_pass"]),
            "confidence": float(r["confidence"]),
            "p_down": float(r["p_down"]),
            "p_hold": float(r["p_hold"]),
            "p_up": float(r["p_up"]),
        }
    return decisions, {
        "scored": len(decisions),
        "tradeable": sum(d["tradeable"] for d in decisions.values()),
        "invalid_features": int((~valid & active).sum()),
    }


class PrecisionPaperBroker(PaperBroker):
    """Fixed-horizon paper orders without invented SL/TP or automatic reversal."""
    def __init__(self, profile, *, spread_floor_pips: float = 0., **kwargs):
        super().__init__(profile, max_positions=MAX_BOT_POSITIONS, max_lots=0.01, **kwargs)
        if spread_floor_pips < 0 or not isfinite(spread_floor_pips):
            raise ValueError("spread_floor_pips must be a finite non-negative number")
        self.spread_floor_pips = float(spread_floor_pips)
        self.due_at: dict[str, datetime] = {}

    def spread(self, candle: Candle) -> float:
        return max(super().spread(candle), self.spread_floor_pips * self.pip(candle.symbol))

    def open_fixed(self, candle: Candle, action: str, note: str) -> bool:
        symbol = candle.symbol
        if action not in ("BUY", "SELL"):
            return False
        if symbol in self.positions:
            self.reject(candle.time, symbol, "SKIP_SYMBOL_ALREADY_OPEN")
            return False
        if len(self.positions) >= self.max_positions:
            self.reject(candle.time, symbol, "SKIP_BOT_POSITION_LIMIT")
            return False
        params = self.profile[symbol]
        lots = float(params["volume_min"])  # Preserve broker minimum-lot rule.
        step = float(params["volume_step"])
        if not (0 < lots <= self.max_lots) or abs(round(lots / step) * step - lots) > 1e-8:
            self.reject(candle.time, symbol, "BROKER_MINIMUM_LOT_INCOMPATIBLE")
            return False
        bid, ask = self.quote(candle, at_open=True)
        spread_pips = (ask - bid) / self.pip(symbol)
        if self.max_spread_pips is not None and spread_pips > self.max_spread_pips:
            self.reject(candle.time, symbol, "SPREAD_CAP")
            return False
        entry = ask + self.slip(symbol) if action == "BUY" else bid - self.slip(symbol)
        entry_fee = self.commission(symbol, lots)
        margin = self.margin(symbol, lots, entry)
        position = Position(symbol, action, lots, candle.time, entry, None, None, entry_fee, margin, note)
        immediate = self.pnl(position, bid if action == "BUY" else ask)
        if self.equity - entry_fee + immediate < self.margin_used + margin:
            self.reject(candle.time, symbol, "LOW_BALANCE_MARGIN_BLOCK")
            return False
        self.balance -= entry_fee
        self.positions[symbol] = position
        self.due_at[symbol] = candle.time + timedelta(minutes=HORIZON_MINUTES)
        self.entry_count += 1
        self.events.append({"time": _iso(candle.time), "symbol": symbol, "event": "OPEN", "reason": action})
        return True

    def close_due(self, candle: Candle, *, at_open: bool) -> None:
        symbol = candle.symbol
        if symbol not in self.positions:
            self.due_at.pop(symbol, None)
            return
        due = self.due_at[symbol]
        moment = candle.time if at_open else candle.ends_at
        if moment >= due:
            bid, ask = self.quote(candle, at_open=at_open)
            reason = ("MODEL_HORIZON_GAP_AT_OPEN" if at_open
                      else f"MODEL_HORIZON_{HORIZON_MINUTES}M" if moment == due
                      else "MODEL_HORIZON_DELAYED_DATA_GAP")
            self._close(symbol, moment, bid, ask, reason)
            del self.due_at[symbol]

    def check_margin_stopout(self, candle: Candle) -> None:
        old = set(self.positions)
        self.check_stopout(candle)
        for sym in old - set(self.positions):
            self.due_at.pop(sym, None)


def run_precision_replay(*, data_zip: str | Path, start: datetime, end: datetime,
                         symbols: tuple[str, ...] = SYMBOLS,
                         model_root: str | Path = DEFAULT_MODEL_ROOT,
                         output: str | Path = DEFAULT_OUTPUT, balance: float = 150.,
                         leverage: float = 100., spread_floor_pips: float = 0.,
                         spread_multiplier: float = 1., slippage_multiplier: float = 1.,
                         stopout_level_pct: float = 50., max_spread_pips: float | None = None,
                         exploratory: bool = False, lock_time: datetime | None = None,
                         equity_every: int = 12,
                         entry_policy: Callable[[Candle, Candle | None, PrecisionPaperBroker], tuple[bool, str]] | None = None) -> dict[str, Any]:
    total_started = perf_counter()
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("Supply valid UTC start/end with start < end")
    start, end = start.astimezone(UTC), end.astimezone(UTC)
    if equity_every < 1:
        raise ValueError("equity_every must be >= 1")
    if not symbols or len(set(symbols)) != len(symbols) or any(s not in SYMBOLS for s in symbols):
        raise ValueError("Invalid symbol list")
    archive = MT5Archive(data_zip)
    if any(s not in archive.symbols for s in symbols):
        raise ValueError("Symbol missing from archive")
    models = {s: PrecisionDirectionModel(s, Path(model_root) / s) for s in symbols}
    created = {s: parse_time(str(models[s].metadata["created_utc"])) for s in symbols}
    if lock_time is not None:
        if lock_time.tzinfo is None:
            raise ValueError("--lock-time must include UTC offset")
        lock_time = lock_time.astimezone(UTC)
        if lock_time < max(created.values()):
            raise ValueError("Freeze timestamp predates a model's creation; cannot certify out-of-sample")
        if start <= lock_time:
            raise ValueError("Replay start must be strictly after the model lock")
    elif not exploratory:
        raise ValueError("The archive predates model creation. Use --exploratory, or a verified later dataset and --lock-time")
    model_root = Path(model_root)
    hashes = {s: {"metadata": sha256_file(model_root / s / "metadata.json"),
                  "model": sha256_file(model_root / s / "model.json")} for s in symbols}
    warmup = start - timedelta(days=30)
    raw = {s: list(archive.candles(s, "M5", start=warmup, end=end)) for s in symbols}
    predictions, counts = {}, {}
    for sym in symbols:
        predictions[sym], counts[sym] = _prepare_decisions(raw[sym], models[sym], start, end)
    broker = PrecisionPaperBroker(archive.broker_profile, initial_balance=balance,
                                  leverage=leverage, max_spread_pips=max_spread_pips,
                                  spread_floor_pips=spread_floor_pips,
                                  slippage_multiplier=slippage_multiplier,
                                  spread_multiplier=spread_multiplier,
                                  stopout_level=stopout_level_pct / 100.)
    active = []
    for sym in symbols:
        active.extend(c for c in raw[sym] if start <= c.time < end)
    active.sort(key=lambda c: (c.time, c.symbol))
    if not active:
        raise ValueError("No market data in selected interval")
    latest_market_time = active[-1].time
    pre_model_replay = latest_market_time < min(created.values())
    stats = defaultdict(int)
    equity_rows = []
    last = {}
    # Observed at the current entry OPEN: only a PREVIOUSLY COMPLETED M5 candle is available.
    # Seed this from pre-start warmup, avoiding a cold-start shortcut.
    last_closed = {sym: next((c for c in reversed(raw[sym]) if c.ends_at <= start), None)
                   for sym in symbols}
    elapsed_start = perf_counter()
    for n, candle in enumerate(active, 1):
        sym = candle.symbol
        last[sym] = candle
        broker.update_mark(candle, at_open=True)
        broker.close_due(candle, at_open=True)  # missing-bar / weekend delay is explicit
        decision = predictions[sym].get(candle.time)
        if decision is not None:
            stats["decisions"] += 1
            if decision["tradeable"] and decision["historical_gate_pass"] and decision["action"] in ("BUY", "SELL"):
                stats["qualified_signals"] += 1
                allowed, reason = (entry_policy(candle, last_closed[sym], broker)
                                   if entry_policy is not None else (True, ""))
                if not allowed:
                    if not reason or not reason.startswith("COST_POLICY_"):
                        raise ValueError("Policy must return a COST_POLICY_* rejection reason")
                    broker.reject(candle.time, sym, reason)
                    stats["cost_policy_rejected"] += 1
                elif broker.open_fixed(candle, decision["action"],
                                       f"precision:{decision['bar_time'].isoformat()} conf={decision['confidence']:.5f}"):
                    stats["entries"] += 1
            else:
                stats["hold_or_gated"] += 1
        broker.update_mark(candle, at_open=False)
        broker.check_margin_stopout(candle)
        broker.close_due(candle, at_open=False)  # 6th target M5 candle has finished
        broker.record_equity()
        last_closed[sym] = candle
        if n % equity_every == 0:
            equity_rows.append({"time": _iso(candle.ends_at), "balance_usd": round(broker.balance, 8),
                                "equity_usd": round(broker.equity, 8), "open_positions": len(broker.positions),
                                "margin_used_usd": round(broker.margin_used, 8)})
    time_spent = perf_counter() - elapsed_start
    # Last available per-symbol close is marked in broker; close residual exposure as explicitly truncated.
    final_time = max(c.ends_at for c in last.values())
    broker.liquidate_end(final_time)
    equity_rows.append({"time": _iso(final_time), "balance_usd": round(broker.balance, 8),
                        "equity_usd": round(broker.equity, 8), "open_positions": 0, "margin_used_usd": 0.})
    trades = broker.trades
    wins = [t for t in trades if t["net_pnl_usd"] > 0]
    plus = sum(t["net_pnl_usd"] for t in wins)
    minus = -sum(t["net_pnl_usd"] for t in trades if t["net_pnl_usd"] < 0)
    report = {
        "status": "SIMULATION_ONLY_NO_MT5_ORDER_ACCESS",
        "strategy": "TRADEAI_FROZEN_PRECISION_M5_DEMO_HORIZON_POLICY",
        "evaluation_label": "DECLARED_POST_LOCK_VERIFY_INDEPENDENTLY" if lock_time else "EXPLORATORY_IN_SAMPLE_OR_HOLDOUT_REUSE",
        "warning": ("Candle-only simulated execution is NOT a live/forward guarantee; actual MT5 fills, spreads, swaps, stop-outs and outages can differ. "
                    + ("All replayed candles PRECEDE these model creation timestamps: exploratory only, NOT independent forward evidence." if pre_model_replay
                       else "Verify the model freeze and historical training data exclusion before treating any segment as independent.")),
        "data_sha256": sha256_file(Path(data_zip)),
        "model_sha256": hashes,
        "model_created_utc": {s: _iso(created[s]) for s in symbols},
        "model_gate": {s: bool(models[s].historical_gate_pass) for s in symbols},
        "model_action_confidence": {s: models[s].action_confidence for s in symbols},
        "model_lock_time": _iso(lock_time) if lock_time else None,
        "symbol_decisions": counts,
        "symbols": list(symbols), "start_utc": _iso(start), "end_exclusive_utc": _iso(end),
        "last_historical_candle_utc": _iso(latest_market_time),
        "all_replayed_bars_precede_model_creation": pre_model_replay,
        "candles_replayed": len(active), "decision_count": stats["decisions"],
        "qualified_signals": stats["qualified_signals"], "cost_policy_rejected": stats["cost_policy_rejected"],
        "cost_policy_enabled": entry_policy is not None, "filled_entries": broker.entry_count,
        "rejected_entries": len(broker.rejections), "trades_closed": len(trades),
        "starting_balance_usd": round(balance, 8), "ending_balance_usd": round(broker.balance, 8),
        "net_profit_usd": round(broker.balance-balance, 8),
        "return_pct": round(100 * (broker.balance/balance-1), 6),
        "max_drawdown_pct": round(broker.max_drawdown_pct, 6),
        "max_drawdown_usd": round(broker.max_drawdown_usd, 8),
        "win_rate_pct": round(100*len(wins)/len(trades), 4) if trades else None,
        "profit_factor": round(plus/minus, 6) if minus else ("INF" if plus else None),
        "execution_loop_wall_seconds": round(time_spent, 3),
        "total_wall_seconds": round(perf_counter() - total_started, 3),
        "commission_usd": round(sum(t["commission_usd"] for t in trades), 8),
        "zero_spread_possible": bool(spread_floor_pips == 0),
        "equity_rows_sampled_every": equity_every,
        "spread_floor_pips": spread_floor_pips, "spread_multiplier": spread_multiplier,
        "slippage_multiplier": slippage_multiplier, "leverage_ASSUMED": leverage,
        "stopout_level_ASSUMED_pct": stopout_level_pct,
        "max_positions": MAX_BOT_POSITIONS,
        "execution": "BROKER_MIN_LOT_OPEN_NEXT_M5; NO_SL_TP; CLOSE_AFTER_6TH_M5; NO_REVERSAL",
    }
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    def write_csv(name, rows):
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    write_csv("trades.csv", broker.trades)
    write_csv("equity.csv", equity_rows)
    write_csv("rejections.csv", broker.rejections)
    write_csv("events.csv", broker.events)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def parse_bound(raw: str, end_date: bool = False) -> datetime:
    if len(raw) == 10:
        d = parse_time(raw + "T00:00:00+00:00")
        return d + timedelta(days=1) if end_date else d
    return parse_time(raw)


def main() -> None:
    p = argparse.ArgumentParser(description="TradeAI precision demo model replay - read-only, NO MT5")
    p.add_argument("--data", required=True, help="Historical MT5 M5/H1 ZIP")
    p.add_argument("--from", dest="start", required=True, help="UTC date inclusive")
    p.add_argument("--to", dest="end", required=True, help="UTC date inclusive")
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS), choices=list(SYMBOLS))
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    p.add_argument("--balance", type=float, default=150.0)
    p.add_argument("--leverage", type=float, default=100.0)
    p.add_argument("--spread-floor-pips", type=float, default=0.0, help="Optional minimum simulated spread for ALL pairs")
    p.add_argument("--spread-multiplier", type=float, default=1.0)
    p.add_argument("--slippage-multiplier", type=float, default=1.0)
    p.add_argument("--stopout-level", type=float, default=50., help="ASSUMED broker margin stopout percentage")
    p.add_argument("--max-spread-pips", type=float)
    p.add_argument("--equity-every", type=int, default=12)
    p.add_argument("--lock-time", help="Verified UTC model lock timestamp, requires STRICTLY later replay interval")
    p.add_argument("--exploratory", action="store_true", help="Explicitly accept model-training/holdout-period replay")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    a = p.parse_args()
    result = run_precision_replay(data_zip=a.data, start=parse_bound(a.start), end=parse_bound(a.end, True),
                                  symbols=tuple(a.symbols), model_root=a.model_root, output=a.output,
                                  balance=a.balance, leverage=a.leverage, spread_floor_pips=a.spread_floor_pips,
                                  spread_multiplier=a.spread_multiplier,
                                  slippage_multiplier=a.slippage_multiplier,
                                  stopout_level_pct=a.stopout_level, max_spread_pips=a.max_spread_pips,
                                  exploratory=a.exploratory,
                                  lock_time=parse_bound(a.lock_time) if a.lock_time else None,
                                  equity_every=a.equity_every)
    print(json.dumps(result, indent=2))
    print("Reports:", Path(a.output).resolve())


if __name__ == "__main__":
    main()
