import re

import pandas as pd

from analytics.logger import log

from config.settings import (
    BACKTEST_START_DATE,
    BACKTEST_END_DATE,
)


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _start_boundary(value):
    text = str(value or "").strip()
    if not text:
        return None
    return pd.Timestamp(text)


def _end_exclusive_boundary(value):
    """Return an exclusive end boundary.

    UI backtests use YYYY-MM-DD. Treat that as the full calendar day instead
    of midnight at the start of the day. A timestamp with a time component is
    treated as inclusive by advancing one nanosecond.
    """
    text = str(value or "").strip()
    if not text:
        return None

    ts = pd.Timestamp(text)
    if _DATE_ONLY_RE.fullmatch(text):
        return ts + pd.Timedelta(days=1)

    return ts + pd.Timedelta(nanoseconds=1)


class ReplayEngine:
    """Chronological multi-symbol replay with pre-period warmup history.

    Important invariants:
      * HISTORY_SIZE warmup bars come from BEFORE BACKTEST_START_DATE.
      * The first executable candle is the first candle on/after the requested
        backtest start, not HISTORY_SIZE bars into the requested period.
      * A date-only BACKTEST_END_DATE includes the whole end date.
      * Symbols advance by the earliest next timestamp so missing bars on one
        symbol cannot push another symbol into the future.
    """

    def __init__(self, file_path, symbols, history_size=2000):
        self.symbols = list(symbols)
        self.history_size = int(history_size)

        if self.history_size < 1:
            raise ValueError("history_size must be >= 1")

        # =====================================================
        # LOAD FULL DATASET
        # =====================================================
        source = pd.read_csv(file_path)

        if "time" not in source.columns or "symbol" not in source.columns:
            raise RuntimeError("Replay dataset requires time and symbol columns")

        source["time"] = pd.to_datetime(source["time"], errors="coerce")
        source.dropna(subset=["time", "symbol"], inplace=True)
        source = source.sort_values(["symbol", "time"]).reset_index(drop=True)

        original_rows = len(source)

        self.start_time = _start_boundary(BACKTEST_START_DATE)
        self.end_exclusive = _end_exclusive_boundary(BACKTEST_END_DATE)

        if (
            self.start_time is not None
            and self.end_exclusive is not None
            and self.start_time >= self.end_exclusive
        ):
            raise ValueError("Backtest start must be before the end boundary")

        log(
            f"INFO | Backtest replay window "
            f"{BACKTEST_START_DATE or 'BEGIN'}"
            f" -> "
            f"{BACKTEST_END_DATE or 'END'}"
        )
        log(f"INFO | CSV source rows {original_rows}")

        # =====================================================
        # BUILD PER-SYMBOL WARMUP + ACTIVE WINDOWS
        # =====================================================
        self.data = {}
        self.pointer = {}
        self.active_start_pointer = {}
        self.active_total = {}
        self.finished_symbols = set()

        for symbol in self.symbols:
            symbol_df = (
                source[source["symbol"] == symbol]
                .sort_values("time")
                .reset_index(drop=True)
            )

            if symbol_df.empty:
                self.data[symbol] = symbol_df
                self.pointer[symbol] = 0
                self.active_start_pointer[symbol] = 0
                self.active_total[symbol] = 0
                self.finished_symbols.add(symbol)
                log(f"WARNING | {symbol} has no replay data")
                continue

            times = symbol_df["time"]

            if self.start_time is None:
                requested_start_index = 0
            else:
                requested_start_index = int(
                    times.searchsorted(self.start_time, side="left")
                )

            if self.end_exclusive is None:
                active_end_index = len(symbol_df)
            else:
                active_end_index = int(
                    times.searchsorted(self.end_exclusive, side="left")
                )

            active_end_index = max(
                requested_start_index,
                min(active_end_index, len(symbol_df)),
            )

            # Retain only the warmup tail before the requested period instead
            # of carrying the entire multi-year history in ReplayEngine.
            slice_start = max(
                0,
                requested_start_index - self.history_size,
            )
            temp = (
                symbol_df.iloc[slice_start:active_end_index]
                .copy()
                .reset_index(drop=True)
            )

            pointer = requested_start_index - slice_start

            # If the dataset itself does not provide enough warmup candles,
            # fail soft by delaying execution until HISTORY_SIZE is available.
            if pointer < self.history_size:
                missing = self.history_size - pointer
                pointer = min(self.history_size, len(temp))
                if missing > 0:
                    log(
                        f"WARNING | {symbol} has only "
                        f"{self.history_size - missing} pre-start warmup bars; "
                        f"execution delayed by up to {missing} bars"
                    )

            active_count = max(0, len(temp) - pointer)

            self.data[symbol] = temp
            self.pointer[symbol] = pointer
            self.active_start_pointer[symbol] = pointer
            self.active_total[symbol] = active_count

            if pointer >= len(temp):
                self.finished_symbols.add(symbol)

            first_exec = (
                str(temp.iloc[pointer]["time"])
                if pointer < len(temp)
                else "NONE"
            )
            last_exec = (
                str(temp.iloc[-1]["time"])
                if pointer < len(temp)
                else "NONE"
            )

            log(
                f"INFO | {symbol} warmup={pointer} "
                f"active={active_count} "
                f"first_exec={first_exec} "
                f"last_exec={last_exec}"
            )

        # =====================================================
        # STATE
        # =====================================================
        self.current_time = None
        self.current_candles = {}

        log("INFO | Replay Engine initialized")

    # =====================================================
    # SYNCHRONIZED MARKET SNAPSHOT
    # =====================================================
    def next_market_snapshot(self):
        """Advance only symbols whose next candle is the earliest timestamp."""

        next_times = {}

        for symbol in self.symbols:
            dataset = self.data.get(symbol)
            if dataset is None:
                continue

            index = int(self.pointer.get(symbol, 0))
            if index >= len(dataset):
                self.finished_symbols.add(symbol)
                continue

            next_times[symbol] = pd.Timestamp(
                dataset.iloc[index]["time"]
            )

        if not next_times:
            self.current_candles = {}
            return None

        next_time = min(next_times.values())
        market = {}
        self.current_candles = {}

        # Preserve configured symbol order for same-timestamp tie breaking.
        for symbol in self.symbols:
            symbol_time = next_times.get(symbol)
            if symbol_time is None or symbol_time != next_time:
                continue

            dataset = self.data[symbol]
            index = int(self.pointer[symbol])
            start = max(0, index - self.history_size)

            history = dataset.iloc[start:index].copy()
            if len(history) < self.history_size:
                # This should only happen when the source dataset did not
                # contain enough pre-start bars. Advance until enough history
                # exists instead of leaking the execution candle into history.
                self.pointer[symbol] += 1
                if self.pointer[symbol] >= len(dataset):
                    self.finished_symbols.add(symbol)
                continue

            candle = dataset.iloc[index].copy()

            market[symbol] = history
            self.current_candles[symbol] = candle
            self.pointer[symbol] += 1

            if self.pointer[symbol] >= len(dataset):
                self.finished_symbols.add(symbol)

        self.current_time = next_time

        # If every earliest symbol was skipped only because it lacked warmup,
        # continue internally to the next timestamp.
        if not market:
            return self.next_market_snapshot()

        return market

    # =====================================================
    # CURRENT REPLAY TIME
    # =====================================================
    def get_current_time(self):
        return self.current_time

    # =====================================================
    # CURRENT CANDLE
    # =====================================================
    def get_current_candle(self, symbol):
        return self.current_candles.get(symbol)

    # =====================================================
    # HISTORY ACCESS
    # =====================================================
    def get_history(self, symbol):
        dataset = self.data.get(symbol)
        if dataset is None:
            return None

        index = int(self.pointer.get(symbol, 0))
        start = max(0, index - self.history_size)
        return dataset.iloc[start:index].copy()

    # =====================================================
    # STATUS
    # =====================================================
    def finished(self):
        return len(self.finished_symbols) == len(self.symbols)

    # =====================================================
    # PROGRESS
    # =====================================================
    def progress(self):
        result = {}

        for symbol in self.symbols:
            total = int(self.active_total.get(symbol, 0))
            if total <= 0:
                result[symbol] = 100.0
                continue

            start = int(self.active_start_pointer.get(symbol, 0))
            current = int(self.pointer.get(symbol, start))
            completed = min(total, max(0, current - start))
            result[symbol] = round(completed / total * 100.0, 2)

        return result
