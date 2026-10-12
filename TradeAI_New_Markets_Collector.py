r"""Export broker M5/H1 OHLC and stored spread history for offline TradeAI research.

Usage from TradeAI_System folder:
    python .\TradeAI_New_Markets_Collector.py

READ-ONLY market history/specifications. Does not train, place or modify orders.
Does not export account IDs, credentials, balances, positions, or trading history.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
from zipfile import ZipFile, ZIP_STORED

import pandas as pd

SYMBOLS = ("GOLD", "SILVER", "US100Cash", "US500Cash", "US30Cash")
COLUMNS = ("time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def market_spec(info) -> dict:
    """Instrument public specifications only; never export account identifiers."""
    keys = (
        "name", "description", "digits", "point", "trade_tick_size",
        "trade_tick_value", "trade_tick_value_profit", "trade_tick_value_loss",
        "trade_contract_size", "volume_min", "volume_max", "volume_step",
        "trade_mode", "trade_exemode", "trade_stops_level", "trade_freeze_level",
        "spread", "spread_float", "currency_base", "currency_profit",
        "currency_margin", "swap_mode", "swap_long", "swap_short",
        "margin_initial", "margin_maintenance", "trade_calc_mode",
    )
    data = {}
    for key in keys:
        val = getattr(info, key, None)
        if isinstance(val, (str, int, float, bool)) or val is None:
            data[key] = val
    data["commission_schedule"] = "NOT_AVAILABLE_FROM_MT5_SYMBOL_INFO_VERIFY_MANUALLY"
    return data


def collect_one(mt5, symbol: str, timeframe: int, interval_minutes: int,
                start_year: int, last_year: int, temp: Path) -> tuple[Path | None, dict]:
    output = temp / f"{symbol}_{'M5' if interval_minutes == 5 else 'H1'}.csv.gz"
    summary: dict = {
        "symbol": symbol,
        "timeframe": "M5" if interval_minutes == 5 else "H1",
        "source": "MetaTrader5.copy_rates_range",
        "history_start_requested": f"{start_year}-01-01T00:00:00Z",
        "history_end_exclusive": f"{last_year + 1}-01-01T00:00:00Z",
        "years": {},
        "rows": 0,
        "zero_spread_rows": 0,
        "positive_spread_rows": 0,
        "first_time_utc": None,
        "last_time_utc": None,
        "sha256": None,
    }
    written = 0
    first = last = None
    zero_spreads = positive_spreads = 0
    with gzip.open(output, "wt", encoding="utf-8", newline="", compresslevel=6) as stream:
        for year in range(start_year, last_year + 1):
            lo = datetime(year, 1, 1, tzinfo=timezone.utc)
            hi = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            quotes = mt5.copy_rates_range(symbol, timeframe, lo, hi)
            if quotes is None or len(quotes) == 0:
                summary["years"][str(year)] = {"rows": 0, "mt5_error": str(mt5.last_error())}
                print(f"  {symbol:10s} {summary['timeframe']} {year}: NO HISTORY ({mt5.last_error()})", flush=True)
                continue

            frame = pd.DataFrame(quotes)
            missing = set(COLUMNS).difference(frame.columns)
            if missing:
                raise ValueError(f"{symbol} {year}: MT5 rates missing required columns: {sorted(missing)}")
            frame = frame.loc[:, list(COLUMNS)].copy()
            frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
            frame = frame.loc[(frame.time >= pd.Timestamp(lo)) & (frame.time < pd.Timestamp(hi))]
            frame = frame.sort_values("time").drop_duplicates("time", keep="last")
            if not frame.empty:
                if frame.time.isna().any():
                    raise ValueError(f"{symbol} {year}: null timestamp")
                if (frame["high"] < frame[["open", "close", "low"]].max(axis=1)).any() or \
                   (frame["low"] > frame[["open", "close", "high"]].min(axis=1)).any():
                    raise ValueError(f"{symbol} {year}: malformed OHLC data")
                if (frame["spread"] < 0).any():
                    raise ValueError(f"{symbol} {year}: negative spread")

                n = len(frame)
                first = frame.time.iloc[0] if first is None else min(first, frame.time.iloc[0])
                last = frame.time.iloc[-1] if last is None else max(last, frame.time.iloc[-1])
                zero_spreads += int((frame.spread == 0).sum())
                positive_spreads += int((frame.spread > 0).sum())
                frame.time = frame.time.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                frame.to_csv(stream, header=(written == 0), index=False)
                written += n
                print(f"  {symbol:10s} {summary['timeframe']} {year}: {n:,} bars", flush=True)
            else:
                print(f"  {symbol:10s} {summary['timeframe']} {year}: NO BARS IN YEAR", flush=True)
            summary["years"][str(year)] = {"rows": len(frame)}

    if written == 0:
        output.unlink(missing_ok=True)
        summary["warning"] = "No bars exported. Check MT5 historical availability."
        return None, summary

    summary["rows"] = written
    summary["first_time_utc"] = first.strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["last_time_utc"] = last.strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["zero_spread_rows"] = zero_spreads
    summary["positive_spread_rows"] = positive_spreads
    summary["sha256"] = sha256_file(output)
    summary["archive_member"] = f"market_data/{output.name}"
    if first.year > start_year:
        summary["warning"] = (f"Broker data begins in {first.year}; earlier bars unavailable or not loaded. "
                              "Check MT5 Tools > Options > Charts > Max. bars in chart and server history.")
    return output, summary


def run(mt5, output: Path, start_year: int, end_year: int) -> dict:
    if end_year >= 2026:
        raise ValueError("This research collector intentionally excludes 2026 and later")
    if start_year < 2000 or end_year < start_year:
        raise ValueError("Invalid requested year range")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing research archive: {output}")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    manifest = {
        "purpose": "ISOLATED_RESEARCH_MARKET_DATA_EXPORT_NO_TRADING",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "year_range_inclusive": [start_year, end_year],
        "timestamp_convention": "MT5 epoch seconds represented as UTC; verify broker clock convention",
        "data_limitations": (
            "M5/H1 historical OHLC and bar-associated spread in MT5 points, "
            "NOT bid/ask tick history, tick-accurate stops, fill quotes, "
            "commission schedule, slippage or swap. Zero spreads are preserved, NOT treated as free trading."
        ),
        "account_ids_or_secrets_exported": False,
        "symbols": {},
        "data_files": {},
    }
    try:
        account = mt5.account_info()
        manifest["account_currency_only"] = getattr(account, "currency", None)
        frames = (("M5", mt5.TIMEFRAME_M5, 5), ("H1", mt5.TIMEFRAME_H1, 60))
        with tempfile.TemporaryDirectory(prefix="tradeai_market_collector_") as tmp_name:
            tmp = Path(tmp_name)
            produced = []
            for symbol in SYMBOLS:
                info = mt5.symbol_info(symbol)
                if info is None:
                    print(f"{symbol}: UNAVAILABLE", flush=True)
                    manifest["symbols"][symbol] = {"error": "not present in broker symbol list"}
                    continue
                if not info.visible and not mt5.symbol_select(symbol, True):
                    print(f"{symbol}: cannot enable for history retrieval", flush=True)
                    manifest["symbols"][symbol] = {"error": "MT5 symbol_select failed"}
                    continue
                manifest["symbols"][symbol] = market_spec(info)
                for name, tf, minutes in frames:
                    path, summary = collect_one(mt5, symbol, tf, minutes, start_year, end_year, tmp)
                    manifest["data_files"][f"{symbol}_{name}"] = summary
                    if path is not None:
                        produced.append(path)
            if not produced:
                raise RuntimeError("No history was retrievable. Archive not created.")
            manifest_bytes = json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8")
            with ZipFile(output, mode="x", compression=ZIP_STORED, allowZip64=True) as archive:
                archive.writestr("collection_manifest.json", manifest_bytes)
                for path in produced:
                    archive.write(path, arcname=f"market_data/{path.name}")
    finally:
        mt5.shutdown()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output", type=Path, default=Path("TradeAI_NEW_MARKETS_2020_2025.zip"))
    args = parser.parse_args()

    import MetaTrader5 as mt5

    print("TRADEAI MARKET-DATA EXPORT | READ ONLY | NO TRADING", flush=True)
    print("Symbols:", ", ".join(SYMBOLS), flush=True)
    print("UTC dates:", args.start_year, "through", args.end_year, flush=True)
    manifest = run(mt5, args.output, args.start_year, args.end_year)
    print("\nEXPORT COMPLETE:", args.output.resolve(), flush=True)
    print("FILE SIZE MB:", round(args.output.stat().st_size / 1024**2, 2), flush=True)
    for key, result in manifest["data_files"].items():
        print(f"  {key:15s} {result['rows']:>9,} rows | "
              f"{result['first_time_utc']} .. {result['last_time_utc']}", flush=True)
        if result.get("warning"):
            print("     WARNING:", result["warning"], flush=True)
    print("Upload this ZIP to the ChatGPT TradeAI conversation for local offline research.", flush=True)
    print("No account identifiers, passwords, order history or trading activity exported.", flush=True)


if __name__ == "__main__":
    main()
