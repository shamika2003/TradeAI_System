# filename: TradeAI/market/mt5_feed.py

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from typing import Callable

import MetaTrader5 as mt5
import pandas as pd

from analytics.logger import log
from config.settings import SYMBOLS


# Canonical feature schema contains a 1000-bar rolling z-score. Keep headroom
# so the latest feature row is valid even after cleaning/warm-up.
MIN_FEATURE_BARS = 1100
_FETCH_LOG_AT: dict[tuple[str, int], float] = {}


def initialize_mt5():
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialization failed: {mt5.last_error()}")

    terminal = mt5.terminal_info()
    if terminal is None:
        raise RuntimeError("MT5 terminal not reachable")

    log("INFO | MT5 Connected Successfully")

    for symbol in SYMBOLS:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")
        if not info.visible and not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Unable to enable symbol: {symbol}")
        log(f"INFO | Symbol enabled: {symbol}")

    return True


def _timeframe_minutes(timeframe: int) -> int:
    if timeframe == mt5.TIMEFRAME_H1:
        return 60
    if timeframe == mt5.TIMEFRAME_M5:
        return 5
    return 5


def _timeframe_name(timeframe: int) -> str:
    if timeframe == mt5.TIMEFRAME_H1:
        return "H1"
    if timeframe == mt5.TIMEFRAME_M5:
        return "M5"
    return str(timeframe)


def _to_frame(symbol: str, rates) -> pd.DataFrame | None:
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    if df.empty or "time" not in df.columns:
        return None
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df["symbol"] = symbol
    df = df.dropna()
    df = df.drop_duplicates(subset=["time"])
    df = df.sort_values("time").reset_index(drop=True)
    return df


def _log_short_data(symbol: str, timeframe: int, got: int, need: int, attempt: int) -> None:
    # Do not flood bot.log every 1 second when a broker is temporarily not
    # serving history. One warning per symbol/timeframe every 15 seconds is enough.
    key = (symbol, int(timeframe))
    now = time.monotonic()
    last = _FETCH_LOG_AT.get(key, 0.0)
    if now - last >= 15.0 or attempt == 1:
        log(
            f"WARNING | MT5 history short {symbol} {_timeframe_name(timeframe)} "
            f"got={got} need>={need} try={attempt}"
        )
        _FETCH_LOG_AT[key] = now


def _cancelled(should_stop: Callable[[], bool] | None) -> bool:
    try:
        return bool(should_stop and should_stop())
    except Exception:
        return False


def _fetch_rates(
    symbol: str,
    timeframe: int,
    n_bars: int,
    retry: int = 3,
    should_stop: Callable[[], bool] | None = None,
):
    """Fetch enough CLOSED MT5 history for the production feature contract.

    copy_rates_from_pos can return only the terminal's currently cached history.
    When that happens, fall back to copy_rates_range so MT5 requests a larger
    history window from the broker rather than retrying the same short cache.
    """
    required = min(int(n_bars), MIN_FEATURE_BARS)
    timeframe_minutes = _timeframe_minutes(timeframe)

    info = mt5.symbol_info(symbol)
    if info is not None and not info.visible:
        mt5.symbol_select(symbol, True)

    for attempt in range(1, retry + 1):
        if _cancelled(should_stop):
            return None

        best: pd.DataFrame | None = None

        try:
            rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, int(n_bars))
            best = _to_frame(symbol, rates)
        except Exception:
            best = None

        if best is not None and len(best) >= required:
            return best.tail(int(n_bars)).reset_index(drop=True)

        got = 0 if best is None else len(best)
        _log_short_data(symbol, timeframe, got, required, attempt)

        if _cancelled(should_stop):
            return None

        # Ask MT5/broker for a time range. Use generous windows because H1 needs
        # >1000 CLOSED bars for the canonical rolling features.
        try:
            end = datetime.now(timezone.utc) - timedelta(minutes=timeframe_minutes)
            lookback_days = 220 if timeframe == mt5.TIMEFRAME_H1 else 35
            start = end - timedelta(days=lookback_days)
            range_rates = mt5.copy_rates_range(symbol, timeframe, start, end)
            ranged = _to_frame(symbol, range_rates)
            if ranged is not None:
                if best is None or len(ranged) > len(best):
                    best = ranged
                if len(best) >= required:
                    return best.tail(int(n_bars)).reset_index(drop=True)
        except Exception as exc:
            if attempt == retry:
                log(
                    f"WARNING | MT5 range history failed {symbol} "
                    f"{_timeframe_name(timeframe)}: {exc}"
                )

        if attempt < retry:
            # Interruptible short wait; never keep retrying after STOP.
            for _ in range(10):
                if _cancelled(should_stop):
                    return None
                time.sleep(0.1)

    got = 0 if best is None else len(best)
    log(
        f"ERROR | Market data unavailable {symbol} {_timeframe_name(timeframe)} "
        f"got={got} need>={required}"
    )
    return None


def get_mtf_data(symbol: str, should_stop: Callable[[], bool] | None = None):
    if _cancelled(should_stop):
        return None, None

    df_m5 = _fetch_rates(symbol, mt5.TIMEFRAME_M5, 2000, should_stop=should_stop)
    if _cancelled(should_stop):
        return None, None

    df_h1 = _fetch_rates(symbol, mt5.TIMEFRAME_H1, 2000, should_stop=should_stop)

    if df_m5 is None or df_h1 is None:
        return None, None
    if df_m5.empty or df_h1.empty:
        return None, None

    return df_m5, df_h1
