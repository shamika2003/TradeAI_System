"""Leakage-resistant, streaming replay: decisions at prior M5 close, fills at next M5 open."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import csv
import hashlib
import importlib
import inspect
import json
from pathlib import Path
from time import perf_counter
from typing import Callable

from .broker import PaperBroker, Position, Signal
from .data import Candle, CompletedH1, MT5Archive


@dataclass(frozen=True, slots=True)
class Observation:
    symbol: str
    decision_time: datetime      # current new candle OPEN, so previous candle is completed
    completed_m5: tuple[Candle, ...]
    completed_h1: tuple[Candle, ...]
    balance_usd: float
    equity_usd: float
    existing_position: Position | None
    # Do not include current execution candle OHLC in observations.


Strategy = Callable[[Observation], Signal | None]


def import_strategy(spec: str) -> Strategy:
    module_name, sep, fn_name = spec.partition(":")
    if not sep or not module_name or not fn_name:
        raise ValueError("--strategy must be module.path:function_name")
    fn = getattr(importlib.import_module(module_name), fn_name)
    if not callable(fn):
        raise ValueError("Strategy must be callable")
    return fn


def _write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        out = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        out.writeheader()
        out.writerows(records)


def run_replay(*, data_zip: str | Path, symbols: list[str], start: datetime, end: datetime,
               strategy: Strategy, output_dir: str | Path, initial_balance: float = 150.0,
               leverage: float = 100.0, max_positions: int = 4,
               max_lots: float = 0.10, max_spread_pips: float | None = None,
               slippage_multiplier: float = 1.0, spread_multiplier: float = 1.0,
               stopout_level: float = 0.50,
               lock_time: datetime | None = None, exploratory: bool = False,
               strategy_id: str = "external", history_bars: int = 100,
               equity_every: int = 12, progress_every: int = 0) -> dict:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("--from / --to must be offset-aware UTC timestamps")
    start, end = start.astimezone(timezone.utc), end.astimezone(timezone.utc)
    if start >= end:
        raise ValueError("--from must precede --to")
    if lock_time is None and not exploratory:
        raise ValueError("Supply the REAL model freeze timestamp with --lock-time, or explicitly use --exploratory")
    if lock_time is not None:
        if lock_time.tzinfo is None:
            raise ValueError("lock_time must include timezone")
        lock_time = lock_time.astimezone(timezone.utc)
        if start <= lock_time:
            raise ValueError("Replay --from precedes model freeze or equals it. Select strictly post-lock data")
    if history_bars < 2 or equity_every < 1 or progress_every < 0:
        raise ValueError("Invalid history/sampling configuration")
    archive = MT5Archive(data_zip)
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Specify one or more distinct symbols")
    if any(s not in archive.symbols for s in symbols):
        raise ValueError(f"Symbols must be within {archive.symbols}")
    # Warm-up market history is permissible; entries only begin >= start.
    warmup_start = start - timedelta(days=30)
    broker = PaperBroker(archive.broker_profile, initial_balance=initial_balance,
                         leverage=leverage, slippage_multiplier=slippage_multiplier,
                         max_positions=max_positions, max_lots=max_lots,
                         max_spread_pips=max_spread_pips,
                         stopout_level=stopout_level, spread_multiplier=spread_multiplier)
    m5: dict[str, deque[Candle]] = {s: deque(maxlen=history_bars) for s in symbols}
    h1 = {s: CompletedH1(archive, s, warmup_start - timedelta(days=2), history_bars)
          for s in symbols}
    previous: dict[str, Candle] = {}
    equity_rows: list[dict] = []
    candles = 0
    active_candles = 0
    first_active: datetime | None = None
    last_active: Candle | None = None
    replay_started = perf_counter()
    for candle in archive.merged_m5(symbols, warmup_start, end):
        candles += 1
        if progress_every and candles % progress_every == 0:
            print(f"Replay scanned {candles:,} M5 bars, through {candle.time.isoformat()}", flush=True)
        symbol = candle.symbol
        now = candle.time
        completed_h1 = h1[symbol].through(now)
        # Only the previous candle can enter model input; current high/low/close not yet known.
        prev = previous.get(symbol)
        if prev is not None:
            m5[symbol].append(prev)
        previous[symbol] = candle
        if now < start:
            continue
        active_candles += 1
        first_active = first_active or now
        last_active = candle
        broker.update_mark(candle, at_open=True)
        broker.check_open_gap(candle)
        if prev is not None:
            obs = Observation(symbol=symbol, decision_time=now,
                              completed_m5=tuple(m5[symbol]), completed_h1=completed_h1,
                              balance_usd=broker.balance, equity_usd=broker.equity,
                              existing_position=(replace(broker.positions[symbol]) if symbol in broker.positions else None))
            signal = strategy(obs)
            if signal is not None:
                if not isinstance(signal, Signal):
                    raise TypeError("Strategy must return Signal or None")
                broker.execute(candle, signal)
        broker.check_range(candle)
        broker.update_mark(candle, at_open=False)
        broker.check_stopout(candle)
        broker.record_equity()
        if active_candles % equity_every == 0:
            equity_rows.append({"time": candle.ends_at.isoformat(), "balance_usd": round(broker.balance, 8),
                                "equity_usd": round(broker.equity, 8), "margin_usd": round(broker.margin_used, 8),
                                "open_positions": len(broker.positions)})
    replay_seconds = max(perf_counter() - replay_started, 0.000001)
    if not active_candles or last_active is None:
        raise ValueError("No historical candles within requested replay interval")
    # The last available closed bar gives last mark; forced liquidation accounts for remaining exposure.
    broker.liquidate_end(last_active.ends_at)
    equity_rows.append({"time": last_active.ends_at.isoformat(), "balance_usd": round(broker.balance, 8),
                        "equity_usd": round(broker.equity, 8), "margin_usd": 0.0, "open_positions": 0})
    trades = broker.trades
    wins = [t for t in trades if t["net_pnl_usd"] > 0]
    profit = sum(t["net_pnl_usd"] for t in wins)
    loss = -sum(t["net_pnl_usd"] for t in trades if t["net_pnl_usd"] < 0)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "trades.csv", trades)
    _write_csv(out / "equity.csv", equity_rows)
    _write_csv(out / "rejections.csv", broker.rejections)
    _write_csv(out / "events.csv", broker.events)
    def hash_file(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    strategy_file = inspect.getsourcefile(strategy)
    summary = {
        "status": "SIMULATION_ONLY_NOT_MT5_DEMO_OR_LIVE",
        "strategy_id": strategy_id,
        "evaluation_label": "USER_DECLARED_POST_LOCK" if lock_time is not None else "EXPLORATORY_NOT_OUT_OF_SAMPLE",
        "warning": ("Model freeze timestamp was user-supplied, not independently verified. "
                    "M5 OHLC has unknown intrabar sequence; spreads are approximated per candle; "
                    "stop-out is tested at bar close; fills and slippage are synthetic."),
        "data_zip": Path(data_zip).name,
        "data_zip_sha256": hash_file(data_zip),
        "strategy_file": Path(strategy_file).name if strategy_file else None,
        "strategy_file_sha256": hash_file(strategy_file) if strategy_file and Path(strategy_file).is_file() else None,
        "broker_profile_captured_at": archive.broker_profile.get("captured_at_utc"),
        "model_freeze_time": lock_time.isoformat() if lock_time else None,
        "requested_from_utc": start.isoformat(),
        "requested_to_exclusive_utc": end.isoformat(),
        "first_active_candle": first_active.isoformat() if first_active else None,
        "last_active_candle": last_active.time.isoformat(),
        "symbols": symbols,
        "candles_processed_including_warmup": candles,
        "candles_replayed": active_candles,
        "replay_loop_wall_seconds": round(replay_seconds, 3),
        "replay_loop_bars_per_second_including_warmup": round(candles / replay_seconds, 2),
        "starting_balance_usd": initial_balance,
        "ending_balance_usd": round(broker.balance, 8),
        "net_profit_usd": round(broker.balance - initial_balance, 8),
        "return_pct": round((broker.balance / initial_balance - 1) * 100, 6),
        "min_equity_usd": round(broker.min_equity, 8),
        "maximum_drawdown_usd": round(broker.max_drawdown_usd, 8),
        "maximum_drawdown_pct": round(broker.max_drawdown_pct, 6),
        "trades_closed": len(trades),
        "entries_filled": broker.entry_count,
        "trades_rejected": len(broker.rejections),
        "win_rate_pct": round(100 * len(wins) / len(trades), 4) if trades else None,
        "profit_factor": round(profit / loss, 6) if loss else (None if not profit else "INF"),
        "gross_winning_trade_profit_usd": round(profit, 8),
        "gross_losing_trade_loss_usd": round(loss, 8),
        "commission_usd": round(sum(t["commission_usd"] for t in trades), 8),
        "leverage_ASSUMED": leverage,
        "contract_size_ASSUMED_UNLESS_PROFILE": 100_000,
        "stopout_level_ASSUMED_pct": stopout_level * 100,
        "max_positions": max_positions,
        "max_lots": max_lots,
        "max_spread_pips": max_spread_pips,
        "slippage_multiplier": slippage_multiplier,
        "spread_multiplier": spread_multiplier,
        "equity_csv_every_n_candles": equity_every,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
