import pandas as pd

from analytics.logger import log

from config.settings import (
    BACKTEST_START_DATE,
    BACKTEST_END_DATE
)

class ReplayEngine:

    def __init__(self, file_path, symbols, history_size=2000):

        self.symbols = symbols
        self.history_size = history_size

        # =====================================================
        # LOAD CSV
        # =====================================================
        self.df = pd.read_csv(file_path)

        self.df["time"] = pd.to_datetime(self.df["time"])

        # =====================================================
        # DATE FILTER
        # =====================================================
        original_rows = len(self.df)

        if BACKTEST_START_DATE:
            self.df = self.df[
                self.df["time"] >= pd.to_datetime(BACKTEST_START_DATE)
            ]

        if BACKTEST_END_DATE:
            self.df = self.df[
                self.df["time"] <= pd.to_datetime(BACKTEST_END_DATE)
            ]

        self.df.reset_index(drop=True, inplace=True)

        log(
            f"INFO | Backtest date filter "
            f"{BACKTEST_START_DATE}"
            f" -> "
            f"{BACKTEST_END_DATE}"
        )

        log(
            f"INFO | CSV rows "
            f"{original_rows}"
            f" -> "
            f"{len(self.df)}"
        )

        # =====================================================
        # SORT DATA
        # =====================================================
        self.df = (self.df.sort_values(["time", "symbol"]).reset_index(drop=True))

        # =====================================================
        # SPLIT SYMBOL DATA
        # =====================================================
        self.data = {}

        for symbol in symbols:
            temp = self.df[self.df["symbol"] == symbol].copy()

            temp = (temp.sort_values("time").reset_index(drop=True))

            self.data[symbol] = temp

            log(
                f"INFO | {symbol} candles "
                f"{len(temp)}"
            )

        # =====================================================
        # REPLAY POINTER
        # =====================================================
        self.pointer = {}

        for symbol in symbols:
            if len(self.data[symbol]) <= history_size:

                self.pointer[symbol] = len(self.data[symbol])

                log(
                    f"WARNING | {symbol} "
                    "not enough history"
                )

            else:
                self.pointer[symbol] = history_size

        # =====================================================
        # STATE
        # =====================================================
        self.current_time = None

        self.current_candles = {}

        self.finished_symbols = set()

        log("INFO | Replay Engine initialized")


    # =====================================================
    # SYNCHRONIZED MARKET SNAPSHOT
    # =====================================================
    def next_market_snapshot(self):

        market = {}
        timestamps = []

        for symbol in self.symbols:
            
            dataset = self.data.get(symbol)

            if dataset is None:
                continue

            index = self.pointer[symbol]

            if index >= len(dataset):
                self.finished_symbols.add(symbol)
                continue

            start = max(0, index - self.history_size)

            history = (dataset.iloc[start:index].copy())

            if history.empty:
                continue

            # ---------------------------------
            # CURRENT FUTURE CANDLE
            # ---------------------------------
            candle = (dataset.iloc[index].copy())

            timestamps.append(candle["time"])

            self.current_candles[symbol] = candle

            # only historical candles go to AI
            market[symbol] = history

            # move replay forward
            self.pointer[symbol] += 1

        if len(market) == 0:
            return None

        # synchronized time
        self.current_time = max(timestamps)

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

        index = self.pointer[symbol]

        start = max(0, index-self.history_size)

        return (dataset.iloc[start:index].copy())

    # =====================================================
    # STATUS
    # =====================================================
    def finished(self):

        return (len(self.finished_symbols) == len(self.symbols))

    # =====================================================
    # PROGRESS
    # =====================================================
    def progress(self):

        result = {}

        for symbol in self.symbols:

            total = len(self.data[symbol])

            current = self.pointer[symbol]

            if total == 0:
                result[symbol] = 100

            else:
                result[symbol] = round(current / total * 100, 2)

        return result