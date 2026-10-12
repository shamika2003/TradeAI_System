"""Command line entry. Run at TradeAI_System repository root."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
from pathlib import Path

from .data import MT5Archive, parse_time
from .runner import import_strategy, run_replay


def parse_bound(value: str, *, end: bool = False):
    if len(value) == 10:
        t = parse_time(value + "T00:00:00+00:00")
        return t + timedelta(days=1) if end else t
    return parse_time(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline, no-orders historical TradeAI replay sandbox")
    subs = parser.add_subparsers(dest="command", required=True)
    inspect = subs.add_parser("inspect", help="Inspect MT5 OHLC archive and broker settings")
    inspect.add_argument("--data", required=True, help="Path to TradeAI_Model_Training_Data.zip")
    run = subs.add_parser("run", help="Stream M5 bars through external strategy and paper broker")
    run.add_argument("--data", required=True)
    run.add_argument("--from", dest="start", required=True, help="UTC date or timestamp, inclusive")
    run.add_argument("--to", dest="end", required=True, help="UTC date inclusive, or timestamp exclusive")
    run.add_argument("--symbols", nargs="+", default=["EURUSD", "GBPUSD", "USDJPY", "USDCNH"])
    run.add_argument("--strategy", required=True, help="Python module path:function")
    run.add_argument("--lock-time", help="REAL training/model freeze timestamp UTC; never guess")
    run.add_argument("--exploratory", action="store_true", help="Label as research only; NOT out-of-sample")
    run.add_argument("--balance", type=float, default=150.0)
    run.add_argument("--leverage", type=float, default=100.0)
    run.add_argument("--max-positions", type=int, default=4)
    run.add_argument("--max-lots", type=float, default=0.10)
    run.add_argument("--max-spread-pips", type=float)
    run.add_argument("--slippage-multiplier", type=float, default=1.0)
    run.add_argument("--spread-multiplier", type=float, default=1.0, help="Stress spread costs; >= 1.0")
    run.add_argument("--stopout-level", type=float, default=50.0, help="Account margin stopout %%")
    run.add_argument("--history-bars", type=int, default=100)
    run.add_argument("--equity-every", type=int, default=12)
    run.add_argument("--progress-every", type=int, default=50000, help="Show progress after N scanned candles; 0 disables")
    run.add_argument("--output", default="artifacts/replay_sandbox/latest")
    args = parser.parse_args()
    if args.command == "inspect":
        archive = MT5Archive(args.data)
        print(json.dumps({"symbols": archive.symbols, "profile": archive.broker_profile,
                          "warning": "M5/H1 candles only: not historical tick quotes"}, indent=2))
        return
    summary = run_replay(
        data_zip=args.data, symbols=args.symbols, start=parse_bound(args.start),
        end=parse_bound(args.end, end=True), strategy=import_strategy(args.strategy),
        output_dir=args.output, initial_balance=args.balance, leverage=args.leverage,
        max_positions=args.max_positions, max_lots=args.max_lots,
        max_spread_pips=args.max_spread_pips,
        slippage_multiplier=args.slippage_multiplier, spread_multiplier=args.spread_multiplier,
        stopout_level=args.stopout_level / 100.0,
        lock_time=parse_bound(args.lock_time) if args.lock_time else None,
        exploratory=args.exploratory, strategy_id=args.strategy,
        history_bars=args.history_bars, equity_every=args.equity_every,
        progress_every=args.progress_every)
    print(json.dumps(summary, indent=2))
    print("Reports:", Path(args.output).resolve())


if __name__ == "__main__":
    main()
