from __future__ import annotations

import json
import joblib
import numpy as np
import pandas as pd

from config_model import DATA_PATH, DATASET_METADATA_PATH, MODEL_PARAMS, MODEL_PATH, SYMBOLS, TIMEFRAME_NAME, TRAINING_CUTOFF_DATE
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from training_utils import create_opportunity_bundle, fit_opportunity_bundle
from shared.tradeai_core.model_contract import create_model_artifact
from shared.tradeai_core.target_definition import MAX_TARGET_HORIZON_BARS, TARGET_VERSION
from shared.tradeai_core.training_window import apply_supervised_training_cutoff


def _load_dataset_metadata() -> dict:
    if not DATASET_METADATA_PATH.exists():
        raise RuntimeError(f"Dataset metadata missing: {DATASET_METADATA_PATH}")
    with open(DATASET_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    checks = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "timeframe": TIMEFRAME_NAME,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"Dataset contract mismatch for {key}: expected {expected!r}, got {metadata.get(key)!r}")
    if set(metadata.get("symbols") or []) != set(SYMBOLS):
        raise RuntimeError("Dataset symbol contract mismatch")
    if metadata.get("closed_candles_only") is not True or metadata.get("causal_h1_alignment") is not True:
        raise RuntimeError("Dataset causality certification missing")
    return metadata


def load_data():
    metadata = _load_dataset_metadata()
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset missing: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    features = FeatureTransformer().get_feature_list()
    required = features + [
        "symbol", "time", "target_class", "target_buy_r", "target_sell_r", "target_best_r",
        "target_buy_tp_hit", "target_sell_tp_hit",
        "target_buy_quality_hit", "target_sell_quality_hit",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {missing}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required, inplace=True)
    df = df.sort_values(["time", "symbol"]).reset_index(drop=True)
    df, window = apply_supervised_training_cutoff(
        df,
        cutoff_exclusive=TRAINING_CUTOFF_DATE,
        label_horizon_bars=MAX_TARGET_HORIZON_BARS,
    )
    effective = dict(metadata)
    effective["training_cutoff_date"] = TRAINING_CUTOFF_DATE
    effective["training_window"] = window.as_dict()
    return df, effective, features, window.as_dict()


def train():
    print("\n" + "═" * 80)
    print("🧠 TRADEAI STRUCTURAL META-OPPORTUNITY ENSEMBLE — PRE-VALIDATION REFIT")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🔐 Feature hash   : {FEATURE_HASH}")
    print(f"🎯 Target version : {TARGET_VERSION}")
    print("🧠 Architecture   : causal structural setup gate + BUY/SELL quality classifiers + expected-net-R regressors")
    print("🔒 This step fits the final pre-holdout model. Stage 11 independently qualifies sparse-stable policy/risk and exact deployment models on separated chronological zones.")

    df, metadata, features, training_window = load_data()
    print(f"✔ Dataset loaded | Rows: {len(df):,}")
    print(
        f"🔒 Training window : {training_window['mode']} | fit_end={training_window['fit_end']} | "
        f"backtest_safe_from={training_window['backtest_safe_from'] or 'FORWARD ONLY'}"
    )

    models, report = {}, {}
    for symbol in SYMBOLS:
        data = df[df["symbol"] == symbol].sort_values("time").reset_index(drop=True)
        if len(data) < 5000:
            raise RuntimeError(f"Insufficient rows for {symbol}: {len(data):,}")
        print("\n" + "─" * 80)
        print(f"📈 FINAL PRE-HOLDOUT FIT: {symbol}")
        print(
            f"Rows={len(data):,} BUY_Q={int(data['target_buy_quality_hit'].sum()):,} "
            f"SELL_Q={int(data['target_sell_quality_hit'].sum()):,} "
            f"BUY_TP={int(data['target_buy_tp_hit'].sum()):,} SELL_TP={int(data['target_sell_tp_hit'].sum()):,}"
        )
        bundle = create_opportunity_bundle()
        fit_opportunity_bundle(bundle, data[features], data)
        models[symbol] = bundle
        report[symbol] = {
            "rows": int(len(data)),
            "buy_quality_hits": int(data["target_buy_quality_hit"].sum()),
            "sell_quality_hits": int(data["target_sell_quality_hit"].sum()),
            "buy_tp_hits": int(data["target_buy_tp_hit"].sum()),
            "sell_tp_hits": int(data["target_sell_tp_hit"].sum()),
            "start": str(data["time"].min()),
            "end": str(data["time"].max()),
            "meta_training": dict(bundle.get("training_meta") or {}),
        }
        print("✔ Four-model opportunity bundle fitted")

    artifact = create_model_artifact(
        models=models,
        symbols=SYMBOLS,
        timeframe=TIMEFRAME_NAME,
        training_metadata=metadata,
        metrics=report,
        model_params=MODEL_PARAMS,
        training_window=training_window,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print("\n" + "═" * 80)
    print(f"✔ PRE-HOLDOUT DEPLOYMENT ARTIFACT SAVED: {MODEL_PATH}")
    print("Stage 11 replaces these provisional bundles with exact score-calibrated structural meta bundles only after validation passes.")
    print("The untouched June runtime backtest remains the final authority.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    train()
