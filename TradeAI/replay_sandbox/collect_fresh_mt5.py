"""Collect recent *closed* MT5 M5/H1 bars into a separate replay archive.

This is a DATA-ONLY tool. It never sends orders, starts engines, trains a model,
or modifies an existing archive. MT5 may encode broker-local clock values as
Unix UTC seconds; preserve those values for existing TradeAI time features and
record the observed broker offset plus derived real UTC in the manifest.

    python -m TradeAI.replay_sandbox.collect_fresh_mt5 --data .\\TradeAI_Model_Training_Data.zip
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

UTC = timezone.utc
SYMBOLS = ("EURUSD", "GBPUSD", "USDJPY", "USDCNH")
DEFAULT_OUTPUT = Path("artifacts/replay_sandbox/mt5_fresh_broker_clock.zip")
DEFAULT_MODEL_DIR = Path("artifacts/models/precision_m5")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def detect_offset(ticks: dict[str, object], now_utc: datetime) -> int:
    """Reject stale/mismatched quotes instead of silently assuming a timezone."""
    offsets = []
    for symbol, tick in ticks.items():
        if tick is None:
            raise RuntimeError(f"No MT5 tick available for {symbol}; collection aborted")
        seconds = int(tick.time)
        delta = (seconds - now_utc.timestamp()) / 3600.0
        offset = round(delta)
        age = offset * 3600 - (seconds - now_utc.timestamp())
        if not (-12 <= offset <= 14) or not (-3 <= age <= 120):
            raise RuntimeError(f"{symbol}: timestamp clock offset/quote age is inconsistent (delta={delta:.4f}h, age={age:.1f}s)")
        offsets.append(offset)
    if len(set(offsets)) != 1:
        raise RuntimeError(f"MT5 symbols disagree about broker clock offset: {dict(zip(ticks, offsets))}")
    return offsets[0]


def _time_iso(value) -> str:
    return value.to_pydatetime().astimezone(UTC).isoformat()


def _csv_gz(df) -> bytes:
    """Encode archive CSV with the same columns as the existing TradeAI export."""
    cols = ("time", "symbol", "timeframe", "open", "high", "low", "close", "tick_volume", "spread", "real_volume")
    out = df.loc[:, cols].copy()
    out["time"] = out["time"].dt.strftime("%Y-%m-%d %H:%M:%S+00:00")
    payload = out.to_csv(index=False).encode("utf-8")
    return gzip.compress(payload, compresslevel=6, mtime=0)


def prepare_frame(frame, *, symbol: str, timeframe: str, offset_hours: int, now_utc: datetime, cutoff_utc: datetime):
    """Drop partial/future candles; check 30-day causal feature warmup coverage."""
    import pandas as pd
    if frame.empty:
        raise RuntimeError(f"MT5 returned no closed {symbol} {timeframe} bars")
    frame = frame.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if frame["symbol"].ne(symbol).any() or frame["timeframe"].ne(timeframe).any():
        raise RuntimeError("MT5 returned rows for unexpected symbol or timeframe")
    duration = pd.Timedelta(minutes=5 if timeframe == "M5" else 60)
    corrected = frame["time"] - pd.Timedelta(hours=offset_hours)
    # fetch_recent_closed uses from_pos(...,1,...): this is another independent assertion.
    good = corrected + duration <= pd.Timestamp(now_utc) + pd.Timedelta(seconds=5)
    if not bool(good.all()):
        raise RuntimeError(f"{symbol} {timeframe}: forming/future candles detected; refusing archive")
    if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise RuntimeError(f"{symbol} {timeframe}: timestamp order failure")
    needed_start = cutoff_utc - timedelta(days=30)
    if timeframe == "M5" and corrected.iloc[0].to_pydatetime() > needed_start:
        raise RuntimeError(f"{symbol}: only {corrected.iloc[0]} to {corrected.iloc[-1]} UTC available; need at least {needed_start} for model warmup. Increase MT5 Tools > Options > Charts > Max bars in chart, then retry.")
    if corrected.iloc[-1].to_pydatetime() < cutoff_utc:
        raise RuntimeError(f"{symbol} {timeframe}: no data newer than frozen-model creation UTC")
    return frame, {
        "count": len(frame),
        "first_mt5_broker_clock_encoded_as_utc": _time_iso(frame["time"].iloc[0]),
        "last_mt5_broker_clock_encoded_as_utc": _time_iso(frame["time"].iloc[-1]),
        "first_corrected_utc": _time_iso(corrected.iloc[0]),
        "last_corrected_utc": _time_iso(corrected.iloc[-1]),
    }


def model_creation_cutoff(root: Path) -> tuple[datetime, dict]:
    model_fingerprints = {}
    moments = []
    for s in SYMBOLS:
        d = root / s
        p, m = d / "model.json", d / "metadata.json"
        if not p.is_file() or not m.is_file():
            raise FileNotFoundError(f"Frozen precision model missing: {d}")
        meta = json.loads(m.read_text(encoding="utf-8"))
        t = datetime.fromisoformat(meta["created_utc"].replace("Z", "+00:00"))
        if t.tzinfo is None:
            raise RuntimeError(f"Model metadata for {s} lacks UTC offset")
        t = t.astimezone(UTC)
        moments.append(t)
        model_fingerprints[s] = {"created_utc": t.isoformat(), "model_sha256": _sha256(p), "metadata_sha256": _sha256(m)}
    return max(moments), model_fingerprints


def collect(*, data_zip: Path, output: Path, model_root: Path, m5_count: int, h1_count: int) -> dict:
    if output.resolve() == data_zip.resolve():
        raise ValueError("Output MUST NOT overwrite original historical ZIP")
    if output.exists():
        raise FileExistsError(f"Output exists; preserving prior capture: {output}. Choose another --output filename")
    if m5_count < 10000 or h1_count < 800:
        raise ValueError("Use at least 10000 M5 and 800 H1 bars for causal feature warmup")
    with ZipFile(data_zip) as z:
        entries = [p for p in z.namelist() if p.replace("\\", "/").split("/")[-1] == "broker_profile.json"]
        if len(entries) != 1:
            raise ValueError("Historical ZIP must contain one broker_profile.json")
        broker_json = z.read(entries[0])
        broker = json.loads(broker_json)
    if broker.get("account_currency") != "USD" or any(s not in broker["symbols"] for s in SYMBOLS):
        raise ValueError("Broker profile missing required USD four-symbol settings")

    lock, fingerprints = model_creation_cutoff(model_root)
    # Import MT5-dependent code only after validating paths, models and output.
    from Trade_Bot_Training.data_collector import MarketDataCollector
    import MetaTrader5 as mt5
    client = MarketDataCollector()
    staged = {}
    details = {}
    try:
        client.connect()
        # Use the actual PC UTC clock, never the broker's raw seconds as UTC.
        now_utc = datetime.now(UTC)
        for s in SYMBOLS:
            client._ensure_symbol(s)
        ticks = {s: mt5.symbol_info_tick(s) for s in SYMBOLS}
        offset_hours = detect_offset(ticks, now_utc)
        for s in SYMBOLS:
            details[s] = {}
            for tf, n in (("M5", m5_count), ("H1", h1_count)):
                frame = client.fetch_recent_closed(s, tf, n)
                frame, meta = prepare_frame(frame, symbol=s, timeframe=tf,
                                            offset_hours=offset_hours, now_utc=now_utc,
                                            cutoff_utc=lock)
                staged[f"{s}_{tf}.csv.gz"] = _csv_gz(frame)
                details[s][tf] = meta
                print(f"{s} {tf}: {meta['count']:,} closed bars; last TRUE UTC={meta['last_corrected_utc']}")
    finally:
        client.disconnect()

    manifest = {
        "schema_version": 1,
        "capture_utc": now_utc.isoformat(),
        "clock_warning": "Archive candle times intentionally preserve MT5 broker-clock wall time misrepresented as UTC by MT5. NEVER label these raw values actual UTC. Subtract broker_offset_hours for true UTC and post-creation eligibility; do not convert model input time features without verifying parity.",
        "broker_offset_hours_observed_at_capture": offset_hours,
        "model_cutoff_latest_creation_real_utc": lock.isoformat(),
        "model_fingerprints": fingerprints,
        "source": "MT5.copy_rates_from_pos(pos=1) via TradeAI MarketDataCollector.fetch_recent_closed; read only",
        "original_broker_profile_sha256": hashlib.sha256(broker_json).hexdigest(),
        "original_archive_sha256": _sha256(data_zip),
        "symbols": details,
        "replay_limitation": "Collection only. Existing precision_replay timestamps/--lock-time are broker-clock-encoded; a timezone-aware forward replay wrapper must be installed before validating true-UTC post-model intervals.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Assemble in a temporary file; replace output only on complete success.
    with tempfile.NamedTemporaryFile(prefix="tradeai_capture_", suffix=".zip", dir=output.parent, delete=False) as fh:
        tmp = Path(fh.name)
    try:
        with ZipFile(tmp, "w", compression=ZIP_DEFLATED, compresslevel=6) as z:
            z.writestr("broker_profile.json", broker_json)
            z.writestr("collection_manifest.json", json.dumps(manifest, indent=2) + "\n")
            for name, blob in sorted(staged.items()):
                z.writestr("mt5_fresh/" + name, blob)
        # Fail closed if the archive isn't readable by the existing replay loader.
        from .data import MT5Archive
        archive = MT5Archive(tmp)
        for s in SYMBOLS:
            for tf in ("M5", "H1"):
                if not next(archive.candles(s, tf), None):
                    raise RuntimeError(f"Empty produced archive: {s}_{tf}")
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    manifest["archive_path"] = str(output.resolve())
    manifest["archive_sha256"] = _sha256(output)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description="Read-only MT5 post-model M5/H1 capture; NEVER orders/training")
    p.add_argument("--data", default="TradeAI_Model_Training_Data.zip", help="Existing broker-profile archive (READ ONLY)")
    p.add_argument("--output", default=str(DEFAULT_OUTPUT), help="NEW file, never overwritten")
    p.add_argument("--model-root", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--m5-count", type=int, default=15000)
    p.add_argument("--h1-count", type=int, default=1500)
    a = p.parse_args()
    info = collect(data_zip=Path(a.data), output=Path(a.output), model_root=Path(a.model_root),
                   m5_count=a.m5_count, h1_count=a.h1_count)
    print(json.dumps({k: info[k] for k in ("archive_path", "archive_sha256", "capture_utc", "broker_offset_hours_observed_at_capture", "model_cutoff_latest_creation_real_utc")}, indent=2))
    print("Collection complete; NO trades, NO training. Do NOT run --lock-time replay on raw MT5 clock values without a verified adapter.")


if __name__ == "__main__":
    main()
