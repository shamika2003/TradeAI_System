# filename: Trade_Bot_Traning/data_collector.py

import time

import MetaTrader5 as mt5
import pandas as pd


class MarketDataCollector:
    def __init__(self):
        self.connected = False

    def connect(self):
        if self.connected:
            return

        print("\n" + "═" * 80)
        print("📡 MARKET DATA ACQUISITION SERVICE")
        print("═" * 80)
        print("🔄 Establishing MetaTrader 5 connection...")

        if not mt5.initialize():
            raise RuntimeError(f"❌ MT5 initialize failed: {mt5.last_error()}")

        self.connected = True
        print("✔ Connection established")
        print("═" * 80)

    def disconnect(self):
        if self.connected:
            print("\n🔌 Closing MetaTrader 5 connection...")
            mt5.shutdown()
            self.connected = False
            print("✔ Connection closed")

    def fetch_history(
        self,
        symbol: str,
        timeframe,
        total_candles: int = 200000,
        chunk_size: int = 5000,
        retry_delay: float = 1.5,
    ) -> pd.DataFrame | None:
        """
        Fetch CLOSED candles only.

        MT5 position 0 is the currently forming candle. Starting at position 1
        removes that incomplete bar from both training timeframes.
        """
        self.connect()

        print("\n" + "─" * 80)
        print(f"📈 DATA REQUEST : {symbol}")
        print(f"📊 Target Candles : {total_candles:,}")
        print(f"📦 Chunk Size     : {chunk_size:,}")
        print("🔒 Closed candles : YES (MT5 start position = 1)")
        print("─" * 80)

        all_data = []

        try:
            for offset in range(0, total_candles, chunk_size):
                start_pos = offset + 1
                request_count = min(chunk_size, total_candles - offset)
                end_pos = start_pos + request_count - 1

                print(
                    f"📥 Fetching MT5 positions "
                    f"{start_pos:,} → {end_pos:,}"
                )

                rates = mt5.copy_rates_from_pos(
                    symbol,
                    timeframe,
                    start_pos,
                    request_count,
                )

                if rates is None or len(rates) == 0:
                    print(f"⚠️ Fetch warning | {symbol} | {mt5.last_error()}")
                    time.sleep(retry_delay)
                    continue

                df_chunk = pd.DataFrame(rates)
                df_chunk["symbol"] = symbol
                all_data.append(df_chunk)

                print(f"✔ Chunk received ({len(df_chunk):,} candles)")

        except Exception as e:
            print(f"❌ Collector failure | {symbol} | {e}")

        if not all_data:
            print(f"⚠️ No usable data acquired for {symbol}")
            return None

        print("\n🧩 Combining data chunks...")
        df = pd.concat(all_data, ignore_index=True)

        print("🧹 Removing duplicate candles...")
        df = df.drop_duplicates(subset=["time"])

        print("🕒 Converting timestamps...")
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df["candle_time"] = df["time"]

        print("📊 Sorting market data...")
        df = df.sort_values("time").reset_index(drop=True)

        print("─" * 80)
        print(f"✅ DATA ACQUISITION COMPLETE | {symbol}")
        print(f"📈 Total Candles : {len(df):,}")

        if not df.empty:
            print(f"🕒 Date Range    : {df['time'].min()} → {df['time'].max()}")

        print("─" * 80)
        return df
