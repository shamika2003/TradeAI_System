# filename: data_fetcher_online.py

import MetaTrader5 as mt5
import pandas as pd
import time

from analytics.logger import log
from config.settings import SYMBOLS


# =====================================================
# MT5 INITIALIZATION
# =====================================================
def initialize_mt5():

    if not mt5.initialize():
        raise RuntimeError("MT5 initialization failed")

    terminal = mt5.terminal_info()

    if terminal is None:
        raise RuntimeError("MT5 terminal not reachable")

    log("INFO | MT5 Connected Successfully")

    for symbol in SYMBOLS:

        info = mt5.symbol_info(symbol)

        if info is None:
            raise RuntimeError(f"Symbol not found: {symbol}")

        if not info.visible:
            mt5.symbol_select(symbol, True)

        log(f"INFO | Symbol enabled: {symbol}")

    return True


# =====================================================
# SAFE FETCH CORE
# =====================================================
def _fetch_rates(symbol, timeframe, n_bars, retry=3):

    for attempt in range(retry):

        try:
            rates = mt5.copy_rates_from_pos(
                symbol,
                timeframe,
                1,
                n_bars
            )

            if rates is None or len(rates) < 100:
                raise ValueError("Insufficient market data")

            df = pd.DataFrame(rates)

            df["time"] = pd.to_datetime(df["time"], unit="s")
            df["symbol"] = symbol

            # CLEANING STEP
            df = df.dropna()
            df = df.drop_duplicates(subset=["time"])
            df = df.sort_values("time").reset_index(drop=True)

            return df

        except Exception as e:
            log(f"WARNING | Fetch failed {symbol} try {attempt+1}: {e}")
            time.sleep(1)

    log(f"ERROR | Market data fetch failed: {symbol}")
    return None


# =====================================================
# MULTI TIMEFRAME PIPELINE
# =====================================================
def get_mtf_data(symbol):

    df_m5 = _fetch_rates(symbol, mt5.TIMEFRAME_M5, 2000)
    df_h1 = _fetch_rates(symbol, mt5.TIMEFRAME_H1, 2000)

    if df_m5 is None or df_h1 is None:
        return None, None

    if df_m5.empty or df_h1.empty:
        return None, None

    return df_m5, df_h1