from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config_model import DATA_PATH, DATASET_METADATA_PATH, SYMBOLS, TIMEFRAME_NAME
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from shared.tradeai_core.target_definition import MAX_TARGET_HORIZON_BARS, TARGET_VERSION, add_training_targets, target_contract


def _aggregate_h1(m5: pd.DataFrame) -> pd.DataFrame:
    x = m5.copy().sort_values("time")
    x["__hour"] = x["time"].dt.floor("h")
    spread_agg = "mean" if "spread" in x.columns else "first"
    spec = {
        "open": "first", "high": "max", "low": "min", "close": "last",
        "tick_volume": "sum",
    }
    if "spread" in x.columns:
        spec["spread"] = spread_agg
    h1 = x.groupby("__hour", sort=True, as_index=False).agg(spec).rename(columns={"__hour": "time"})
    if "spread" not in h1.columns:
        h1["spread"] = 0.0
    return h1


def rebuild():
    print("\n" + "═" * 80)
    print("🧬 TRADEAI DATASET REBUILD — STRUCTURAL META FEATURES V7")
    print("═" * 80)
    print(f"Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"Feature hash   : {FEATURE_HASH}")
    print(f"Target         : {TARGET_VERSION}")
    if not DATA_PATH.exists():
        raise RuntimeError(f"Existing causal M5 dataset missing: {DATA_PATH}")

    source = pd.read_csv(DATA_PATH)
    source["time"] = pd.to_datetime(source["time"], errors="coerce")
    raw_required = ["symbol", "time", "open", "high", "low", "close", "tick_volume"]
    missing = [c for c in raw_required if c not in source.columns]
    if missing:
        raise RuntimeError(
            "Cannot rebuild features from the existing dataset because raw M5 columns are missing: "
            + ", ".join(missing)
            + ". Run dataset_builder.py once to reacquire broker history."
        )
    if "spread" not in source.columns:
        source["spread"] = 0.0

    transformer = FeatureTransformer()
    parts, ranges = [], {}
    for symbol in SYMBOLS:
        print("\n" + "─" * 80)
        print(f"📈 REBUILD {symbol}")
        raw = source[source["symbol"].astype(str).str.upper() == symbol][
            ["time", "open", "high", "low", "close", "tick_volume", "spread"]
        ].copy()
        raw.dropna(subset=["time", "open", "high", "low", "close"], inplace=True)
        raw.sort_values("time", inplace=True)
        raw.drop_duplicates("time", inplace=True)
        raw.reset_index(drop=True, inplace=True)
        if len(raw) < 10_000:
            raise RuntimeError(f"{symbol}: insufficient raw M5 rows for rebuild: {len(raw):,}")
        h1 = _aggregate_h1(raw)
        df = transformer.build_multi_timeframe_features(raw, h1)
        FeatureTransformer.assert_causal_alignment(df)
        df["symbol"] = symbol
        df = add_training_targets(df, symbol=symbol)
        required_targets = [
            "target_class", "target_buy_r", "target_sell_r", "target_best_r",
            "target_buy_tp_hit", "target_sell_tp_hit",
            "target_buy_quality_hit", "target_sell_quality_hit",
        ]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=transformer.get_feature_list() + required_targets, inplace=True)
        df["target_class"] = df["target_class"].astype("int8")
        df.drop(columns=["h1_source_time"], inplace=True, errors="ignore")
        df.reset_index(drop=True, inplace=True)
        if df.empty:
            raise RuntimeError(f"{symbol}: empty rebuilt dataset")
        ranges[symbol] = {
            "rows": int(len(df)), "start": str(df["time"].min()), "end": str(df["time"].max()),
            "class_counts": {str(k): int(v) for k, v in df["target_class"].value_counts().sort_index().to_dict().items()},
            "buy_tp_hits": int(df["target_buy_tp_hit"].sum()),
            "sell_tp_hits": int(df["target_sell_tp_hit"].sum()),
            "buy_quality_hits": int(df["target_buy_quality_hit"].sum()),
            "sell_quality_hits": int(df["target_sell_quality_hit"].sum()),
        }
        print(
            f"✔ rows={len(df):,} "
            f"BUY_Q={ranges[symbol]['buy_quality_hits']:,} SELL_Q={ranges[symbol]['sell_quality_hits']:,} "
            f"BUY_TP={ranges[symbol]['buy_tp_hits']:,} SELL_TP={ranges[symbol]['sell_tp_hits']:,}"
        )
        parts.append(df)

    final = pd.concat(parts, ignore_index=True).sort_values(["time", "symbol"]).reset_index(drop=True)
    tmp = DATA_PATH.with_suffix(".opportunity.tmp.csv")
    final.to_csv(tmp, index=False)
    os.replace(tmp, DATA_PATH)

    metadata = {
        "dataset_contract_version": "tradeai_dataset_v4_quality",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "target_contract": target_contract(),
        "timeframe": TIMEFRAME_NAME,
        "symbols": list(SYMBOLS),
        "rows": int(len(final)),
        "max_target_horizon_bars": int(MAX_TARGET_HORIZON_BARS),
        "closed_candles_only": True,
        "causal_h1_alignment": True,
        "feature_rebuild_source": "existing_causal_m5_aggregated_to_h1",
        "symbol_ranges": ranges,
    }
    meta_tmp = DATASET_METADATA_PATH.with_suffix(".opportunity.tmp.json")
    with open(meta_tmp, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    os.replace(meta_tmp, DATASET_METADATA_PATH)
    print("\n✅ OPPORTUNITY DATASET READY")
    print(f"Rows: {len(final):,}")
    print(f"Dataset: {DATA_PATH}")


if __name__ == "__main__":
    rebuild()
