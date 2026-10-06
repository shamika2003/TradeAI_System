from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import MetaTrader5 as mt5
import pandas as pd

from Trade_Bot_Training.direction_config import (
    CONTEXT_TIMEFRAME,
    H1_CHUNK_DAYS,
    HISTORY_START_UTC,
    M5_CHUNK_DAYS,
    PRIMARY_TIMEFRAME,
    SYMBOLS,
    ensure_directories,
    raw_path,
)


TIMEFRAME_MAP = {
    "M5": mt5.TIMEFRAME_M5,
    "H1": mt5.TIMEFRAME_H1,
}

TIMEFRAME_DELTA = {
    "M5": pd.Timedelta(minutes=5),
    "H1": pd.Timedelta(hours=1),
}


class MarketDataCollector:
    """MT5 historical collector for the directional model.

    Rules:
    - only fully CLOSED candles are returned
    - timestamps are normalized to UTC
    - duplicates are removed
    - malformed OHLC bars are rejected
    - history is requested in bounded time chunks
    """

    def __init__(self) -> None:
        self.connected = False

    def connect(self) -> None:
        if self.connected:
            return

        print("=" * 78)
        print("TRADEAI DIRECTION MODEL - MT5 DATA COLLECTOR")
        print("=" * 78)

        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")

        if mt5.terminal_info() is None:
            mt5.shutdown()
            raise RuntimeError("MT5 terminal is not reachable.")

        self.connected = True
        print("MT5 connected.")

    def disconnect(self) -> None:
        if self.connected:
            mt5.shutdown()
            self.connected = False
            print("MT5 disconnected.")

    def _ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"MT5 symbol not available: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Unable to enable MT5 symbol: {symbol}")

    @staticmethod
    def _normalize_frame(symbol: str, timeframe_name: str, rates) -> pd.DataFrame:
        if rates is None or len(rates) == 0:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        if "time" not in df.columns:
            return pd.DataFrame()

        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df["symbol"] = symbol
        df["timeframe"] = timeframe_name

        for column in (
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ):
            if column not in df.columns:
                df[column] = 0.0

        df = df[
            [
                "time",
                "symbol",
                "timeframe",
                "open",
                "high",
                "low",
                "close",
                "tick_volume",
                "spread",
                "real_volume",
            ]
        ].copy()

        for column in (
            "open",
            "high",
            "low",
            "close",
            "tick_volume",
            "spread",
            "real_volume",
        ):
            df[column] = pd.to_numeric(df[column], errors="coerce")

        df = df.dropna(subset=["time", "open", "high", "low", "close"])

        valid = (
            (df["open"] > 0)
            & (df["high"] > 0)
            & (df["low"] > 0)
            & (df["close"] > 0)
            & (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        )

        return df.loc[valid].copy()

    def fetch_history(
        self,
        symbol: str,
        timeframe_name: str,
        start_utc: str | datetime = HISTORY_START_UTC,
        end_utc: datetime | None = None,
    ) -> pd.DataFrame:
        self.connect()

        symbol = str(symbol).upper()
        timeframe_name = str(timeframe_name).upper()

        if timeframe_name not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe_name}")

        self._ensure_symbol(symbol)
        timeframe = TIMEFRAME_MAP[timeframe_name]

        if isinstance(start_utc, str):
            start = pd.Timestamp(start_utc, tz="UTC").to_pydatetime()
        else:
            start = start_utc
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            else:
                start = start.astimezone(timezone.utc)

        if end_utc is None:
            end = datetime.now(timezone.utc)
        else:
            end = end_utc
            if end.tzinfo is None:
                end = end.replace(tzinfo=timezone.utc)
            else:
                end = end.astimezone(timezone.utc)

        chunk_days = M5_CHUNK_DAYS if timeframe_name == "M5" else H1_CHUNK_DAYS

        print()
        print("-" * 78)
        print(f"{symbol} {timeframe_name} | {start.isoformat()} -> {end.isoformat()}")
        print("-" * 78)

        chunks: list[pd.DataFrame] = []
        cursor = start

        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)
            rates = mt5.copy_rates_range(symbol, timeframe, cursor, chunk_end)
            chunk = self._normalize_frame(symbol, timeframe_name, rates)

            if not chunk.empty:
                chunks.append(chunk)

            print(f"  {cursor.date()} -> {chunk_end.date()} | {len(chunk):,} bars")
            cursor = chunk_end

        if not chunks:
            raise RuntimeError(f"No usable MT5 history for {symbol} {timeframe_name}")

        df = pd.concat(chunks, ignore_index=True)
        df = (
            df.drop_duplicates(subset=["time"], keep="last")
            .sort_values("time")
            .reset_index(drop=True)
        )

        # MT5 bar timestamps are candle OPEN times. Keep a bar only when its
        # full duration has elapsed, so the model never sees a forming candle.
        now_utc = pd.Timestamp.now(tz="UTC")
        delta = TIMEFRAME_DELTA[timeframe_name]
        df = df.loc[(df["time"] + delta) <= now_utc].copy().reset_index(drop=True)

        print(f"Collected {symbol} {timeframe_name}: {len(df):,} closed bars")
        if not df.empty:
            print(f"Range: {df['time'].min()} -> {df['time'].max()}")

        return df

    def fetch_recent_closed(self, symbol: str, timeframe_name: str, count: int) -> pd.DataFrame:
        self.connect()

        symbol = str(symbol).upper()
        timeframe_name = str(timeframe_name).upper()

        if timeframe_name not in TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe_name}")

        self._ensure_symbol(symbol)

        # MT5 position 0 is the currently-forming candle. Position 1 is the
        # latest fully closed candle.
        rates = mt5.copy_rates_from_pos(
            symbol,
            TIMEFRAME_MAP[timeframe_name],
            1,
            int(count),
        )

        df = self._normalize_frame(symbol, timeframe_name, rates)
        if df.empty:
            raise RuntimeError(f"No recent closed data for {symbol} {timeframe_name}")

        return (
            df.drop_duplicates(subset=["time"])
            .sort_values("time")
            .reset_index(drop=True)
        )

    def save_history(self, symbol: str, timeframe_name: str, df: pd.DataFrame) -> Path:
        ensure_directories()
        path = raw_path(symbol, timeframe_name)
        df.to_csv(path, index=False, compression="gzip")
        print(f"Saved: {path}")
        return path


def collect_all() -> None:
    ensure_directories()
    collector = MarketDataCollector()

    try:
        for symbol in SYMBOLS:
            for timeframe_name in (PRIMARY_TIMEFRAME, CONTEXT_TIMEFRAME):
                df = collector.fetch_history(symbol, timeframe_name)
                collector.save_history(symbol, timeframe_name, df)
    finally:
        collector.disconnect()


if __name__ == "__main__":
    collect_all()