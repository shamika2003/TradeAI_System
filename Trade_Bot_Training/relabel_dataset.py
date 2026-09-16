# filename: Trade_Bot_Traning/relabel_dataset.py

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config_model import DATA_PATH, DATASET_METADATA_PATH, SYMBOLS, TIMEFRAME_NAME
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION
from shared.tradeai_core.target_definition import (
    MAX_TARGET_HORIZON_BARS,
    TARGET_VERSION,
    add_training_targets,
    target_contract,
)


def _load_metadata() -> dict:
    if not DATASET_METADATA_PATH.exists():
        raise RuntimeError(f"Dataset metadata missing: {DATASET_METADATA_PATH}")
    with open(DATASET_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise RuntimeError("Feature schema mismatch; rebuild Stage 2 dataset")
    if metadata.get("feature_hash") != FEATURE_HASH:
        raise RuntimeError("Feature hash mismatch; rebuild Stage 2 dataset")
    if metadata.get("closed_candles_only") is not True:
        raise RuntimeError("Dataset is not certified closed-candle-only")
    if metadata.get("causal_h1_alignment") is not True:
        raise RuntimeError("Dataset is not certified causal H1")
    if metadata.get("timeframe") != TIMEFRAME_NAME:
        raise RuntimeError("Dataset timeframe mismatch")
    return metadata


def relabel():
    print("\n" + "═" * 80)
    print("🎯 TRADEAI PROFIT/BARRIER RELABEL — STAGE 3")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🔐 Feature hash   : {FEATURE_HASH}")
    print(f"🎯 New target     : {TARGET_VERSION}")

    metadata = _load_metadata()
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset missing: {DATA_PATH}")

    print(f"📂 Loading causal dataset: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    print(f"✔ Rows loaded: {len(df):,}")

    output = []
    ranges = {}

    for symbol in SYMBOLS:
        print("\n" + "─" * 80)
        print(f"📈 RELABEL: {symbol}")
        symbol_df = df[df["symbol"] == symbol].copy()
        symbol_df = symbol_df.sort_values("time").reset_index(drop=True)
        if symbol_df.empty:
            raise RuntimeError(f"No rows for {symbol}")

        labelled = add_training_targets(symbol_df, symbol=symbol)
        labelled.replace([np.inf, -np.inf], np.nan, inplace=True)
        labelled.dropna(
            subset=["target_class", "target_buy_r", "target_sell_r", "target_best_r"],
            inplace=True,
        )
        labelled["target_class"] = labelled["target_class"].astype("int8")
        labelled.reset_index(drop=True, inplace=True)

        counts = labelled["target_class"].value_counts().sort_index().to_dict()
        print(f"✔ Rows        : {len(labelled):,}")
        print(f"✔ Class counts: {counts}")

        ranges[symbol] = {
            "rows": int(len(labelled)),
            "start": str(labelled["time"].min()),
            "end": str(labelled["time"].max()),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
        }
        output.append(labelled)

    final_df = pd.concat(output, ignore_index=True)
    final_df = final_df.sort_values(["time", "symbol"]).reset_index(drop=True)

    temp_path = DATA_PATH.with_suffix(".stage3.tmp.csv")
    print(f"\n💾 Writing relabelled dataset: {temp_path}")
    final_df.to_csv(temp_path, index=False)
    temp_path.replace(DATA_PATH)

    metadata.update(
        {
            "dataset_contract_version": "tradeai_dataset_v2",
            "target_version": TARGET_VERSION,
            "max_target_horizon_bars": MAX_TARGET_HORIZON_BARS,
            "target_contract": target_contract(),
            "rows": int(len(final_df)),
            "symbol_ranges": ranges,
            "stage3_relabelled_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    metadata.pop("primary_horizon_bars", None)

    with open(DATASET_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "═" * 80)
    print("📊 STAGE 3 RELABEL REPORT")
    print("═" * 80)
    print(f"📈 Total Rows : {len(final_df):,}")
    print(f"🧩 Symbols    : {final_df['symbol'].nunique()}")
    print(f"🎯 Target     : {TARGET_VERSION}")
    print(f"💾 Dataset    : {DATA_PATH}")
    print("✔ PROFIT/BARRIER LABEL CONTRACT READY")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    relabel()
