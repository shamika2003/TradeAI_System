"""Read historical MT5 candle exports directly from TradeAI_Model_Training_Data.zip.

MT5 export OHLC is assumed BID; the per-candle spread is in broker points.
There are no true historical ticks in this dataset.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
import csv
import gzip
import heapq
import io
import json
from pathlib import Path
from typing import Iterator
from zipfile import ZipFile

UTC = timezone.utc


def parse_time(text: str) -> datetime:
    result = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("Timestamps must include their UTC offset: " + text)
    return result.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class Candle:
    time: datetime  # UTC candle START, not end
    symbol: str
    timeframe: str
    open: float
    high: float
    low: float
    close: float
    spread_points: float
    tick_volume: int = 0

    def __post_init__(self) -> None:
        if self.timeframe not in {"M5", "H1"} or self.time.tzinfo is None:
            raise ValueError("Unsupported timeframe or naive candle time")
        if not all(isfinite(v) for v in (self.open, self.high, self.low, self.close, self.spread_points)):
            raise ValueError("Non-finite candle fields")
        if self.high < max(self.open, self.close, self.low) or self.low > min(self.open, self.close, self.high):
            raise ValueError(f"Invalid OHLC at {self.symbol} {self.time}")
        if self.low <= 0 or self.spread_points < 0:
            raise ValueError("Invalid market price/spread")

    @property
    def ends_at(self) -> datetime:
        return self.time + (timedelta(hours=1) if self.timeframe == "H1" else timedelta(minutes=5))


def candle_from_dict(row: dict[str, str]) -> Candle:
    return Candle(
        time=parse_time(row["time"]),
        symbol=row["symbol"].strip(),
        timeframe=row["timeframe"].strip(),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        spread_points=float(row["spread"] or 0),
        tick_volume=int(float(row.get("tick_volume", "0") or 0)),
    )


class MT5Archive:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        with ZipFile(self.path) as z:
            self.entries = {p.replace("\\", "/").split("/")[-1]: p for p in z.namelist()}
            if "broker_profile.json" not in self.entries:
                raise ValueError("Archive needs broker_profile.json")
            self.broker_profile = json.loads(z.read(self.entries["broker_profile.json"]))
        self.symbols = sorted(self.broker_profile["symbols"])
        if self.broker_profile.get("account_currency") != "USD":
            raise ValueError("This version supports USD accounts only")

    def candles(self, symbol: str, timeframe: str, *, start: datetime | None = None,
                end: datetime | None = None) -> Iterator[Candle]:
        key = f"{symbol}_{timeframe}.csv.gz"
        if key not in self.entries:
            raise ValueError(f"Missing {key}; available: {sorted(self.entries)}")
        # Contexts stay open only while generator is being consumed.
        with ZipFile(self.path) as z:
            with z.open(self.entries[key]) as raw, gzip.open(raw, "rt", encoding="utf-8", newline="") as stream:
                previous = None
                start_lex = start.strftime("%Y-%m-%d %H:%M:%S") if start else None
                end_lex = end.strftime("%Y-%m-%d %H:%M:%S") if end else None
                for row in csv.DictReader(stream):
                    # MT5 export timestamps use lexicographically sortable UTC strings.
                    ts = row["time"][:19]
                    if start_lex and ts < start_lex:
                        continue
                    if end_lex and ts >= end_lex:
                        return
                    c = candle_from_dict(row)
                    if c.symbol != symbol or c.timeframe != timeframe:
                        raise ValueError(f"Wrong symbol/timeframe in {key}")
                    if previous is not None and c.time <= previous:
                        raise ValueError(f"Non-increasing timestamps in {key} at {c.time}")
                    previous = c.time
                    if start is not None and c.time < start:
                        continue
                    if end is not None and c.time >= end:
                        return
                    yield c

    def merged_m5(self, symbols: list[str], start: datetime, end: datetime) -> Iterator[Candle]:
        streams = [self.candles(s, "M5", start=start, end=end) for s in symbols]
        # heapq.merge is stable; M5 bars are sorted per symbol; only O(symbols) working memory.
        yield from heapq.merge(*streams, key=lambda x: (x.time, x.symbol))


class CompletedH1:
    """Expose only hourly candles whose CLOSE timestamp <= the decision timestamp."""
    def __init__(self, archive: MT5Archive, symbol: str, start: datetime, lookback: int):
        self._iter = iter(archive.candles(symbol, "H1", start=start))
        self._next = next(self._iter, None)
        self.history: deque[Candle] = deque(maxlen=lookback)

    def through(self, at: datetime) -> tuple[Candle, ...]:
        while self._next is not None and self._next.ends_at <= at:
            self.history.append(self._next)
            self._next = next(self._iter, None)
        return tuple(self.history)
