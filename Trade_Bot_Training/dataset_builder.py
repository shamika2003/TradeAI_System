# filename: Trade_Bot_Traning/dataset_builder.py

from __future__ import annotations

import json
from datetime import datetime, timezone

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from config_model import DATA_PATH, DATASET_METADATA_PATH, SYMBOLS, TIMEFRAME_NAME
from data_collector import MarketDataCollector
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from shared.tradeai_core.target_definition import (
    MAX_TARGET_HORIZON_BARS,
    TARGET_VERSION,
    add_training_targets,
    target_contract,
)


def _validate_causal_rows(df: pd.DataFrame, symbol: str) -> None:
    if "h1_source_time" not in df.columns:
        raise RuntimeError(f"{symbol}: causal H1 audit column is missing")
    h1_available = pd.to_datetime(df["h1_source_time"]) + pd.Timedelta(hours=1)
    m5_decision = pd.to_datetime(df["time"]) + pd.Timedelta(minutes=5)
    bad = h1_available > m5_decision
    if bad.any():
        example = df.loc[bad, ["time", "h1_source_time"]].head(3)
        raise RuntimeError(f"{symbol}: H1 look-ahead leakage detected:\n{example.to_string(index=False)}")


def build_dataset():
    print("\n" + "═" * 80)
    print("📡 MARKET DATASET CONSTRUCTION ENGINE — PROFIT TARGET V5")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🔐 Feature hash   : {FEATURE_HASH}")
    print(f"🎯 Target version : {TARGET_VERSION}")
    print(f"⏱ Barrier horizon: {MAX_TARGET_HORIZON_BARS} M5 bars")

    collector = MarketDataCollector()
    transformer = FeatureTransformer()
    all_symbol_data = []
    symbol_ranges = {}

    try:
        for symbol in SYMBOLS:
            print("\n" + "═" * 80)
            print(f"🌍 PROCESSING SYMBOL: {symbol}")
            print("═" * 80)

            print("📥 Acquiring CLOSED M5 historical data...")
            df_m5 = collector.fetch_history(symbol=symbol, timeframe=mt5.TIMEFRAME_M5, total_candles=500000)
            if df_m5 is None or df_m5.empty:
                print(f"❌ M5 acquisition failed for {symbol}")
                continue
            df_m5 = df_m5.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            print(f"✔ M5 loaded | Candles: {len(df_m5):,}")

            print("📥 Acquiring CLOSED H1 historical data...")
            df_h1 = collector.fetch_history(symbol=symbol, timeframe=mt5.TIMEFRAME_H1, total_candles=40000)
            if df_h1 is None or df_h1.empty:
                print(f"❌ H1 acquisition failed for {symbol}")
                continue
            df_h1 = df_h1.sort_values("time").drop_duplicates("time").reset_index(drop=True)
            print(f"✔ H1 loaded | Candles: {len(df_h1):,}")

            print("🧮 Building canonical point-in-time features...")
            df = transformer.build_multi_timeframe_features(df_m5, df_h1)
            if df is None or df.empty:
                print(f"❌ Feature generation failed for {symbol}")
                continue

            print("🔎 Running H1 causality audit...")
            _validate_causal_rows(df, symbol)
            print("✔ No future H1 information detected")

            df["symbol"] = symbol
            print("🎯 Constructing profit/barrier labels...")
            df = add_training_targets(df, symbol=symbol)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=["target_class", "target_buy_r", "target_sell_r", "target_best_r"], inplace=True)
            df["target_class"] = df["target_class"].astype("int8")
            df.drop(columns=["h1_source_time"], inplace=True, errors="ignore")
            df.reset_index(drop=True, inplace=True)

            if df.empty:
                print(f"⚠️ Empty dataset after cleaning for {symbol}")
                continue

            counts = df["target_class"].value_counts().sort_index().to_dict()
            symbol_ranges[symbol] = {
                "rows": int(len(df)),
                "start": str(df["time"].min()),
                "end": str(df["time"].max()),
                "class_counts": {str(k): int(v) for k, v in counts.items()},
            }
            print(f"✅ SYMBOL COMPLETE: {symbol} | rows={len(df):,} | classes={counts}")
            all_symbol_data.append(df)
    finally:
        print("\n🔌 Disconnecting market data source...")
        collector.disconnect()

    if not all_symbol_data:
        raise RuntimeError("Dataset build failed for all symbols")

    final_df = pd.concat(all_symbol_data, ignore_index=True)
    final_df = final_df.sort_values(["time", "symbol"]).reset_index(drop=True)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(DATA_PATH, index=False)

    metadata = {
        "dataset_contract_version": "tradeai_dataset_v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "target_contract": target_contract(),
        "timeframe": TIMEFRAME_NAME,
        "symbols": list(SYMBOLS),
        "rows": int(len(final_df)),
        "max_target_horizon_bars": MAX_TARGET_HORIZON_BARS,
        "closed_candles_only": True,
        "causal_h1_alignment": True,
        "symbol_ranges": symbol_ranges,
    }
    with open(DATASET_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "═" * 80)
    print("📊 DATASET BUILD REPORT")
    print("═" * 80)
    print(f"📈 Total Rows     : {len(final_df):,}")
    print(f"🧩 Symbols Loaded : {final_df['symbol'].nunique()}")
    print(f"💾 Dataset        : {DATA_PATH}")
    print(f"🧾 Metadata       : {DATASET_METADATA_PATH}")
    print("🔐 Causality      : PASS")
    print("🎯 Profit target  : PASS")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    build_dataset()
