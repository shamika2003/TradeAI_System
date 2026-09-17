# filename: Trade_Bot_Traning/trainer.py

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config_model import (
    DATA_PATH,
    DATASET_METADATA_PATH,
    MODEL_PARAMS,
    MODEL_PATH,
    SYMBOLS,
    TIMEFRAME_NAME,
    TRAINING_CUTOFF_DATE,
)
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from training_utils import aggregate_fold_metrics, compute_weights, create_model, evaluate_probabilities
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
    required = features + ["symbol", "time", "target_class", "target_buy_r", "target_sell_r", "target_best_r"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Dataset missing required columns: {missing}")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required, inplace=True)
    df["target_class"] = df["target_class"].astype(np.int32)
    df = df.sort_values(["time", "symbol"]).reset_index(drop=True)

    df, training_window = apply_supervised_training_cutoff(
        df,
        cutoff_exclusive=TRAINING_CUTOFF_DATE,
        label_horizon_bars=MAX_TARGET_HORIZON_BARS,
    )

    effective_metadata = dict(metadata)
    effective_metadata["training_cutoff_date"] = TRAINING_CUTOFF_DATE
    effective_metadata["training_window"] = training_window.as_dict()
    return df, effective_metadata, features, training_window.as_dict()


def _print_metrics(prefix: str, m: dict):
    print(
        f"{prefix} BAL_ACC={m['balanced_acc']:.4f} "
        f"F1={m['macro_f1']:.4f} "
        f"TRADES={m['trades']:,} "
        f"COVER={m['coverage']*100:.1f}% "
        f"WIN={m['win_rate']*100:.1f}% "
        f"AVG_R={m['avg_r']:.4f} "
        f"PF_R={m['profit_factor_r']:.3f} "
        f"TOTAL_R={m['total_r']:.1f}"
    )


def train():
    print("\n" + "═" * 80)
    print("🧠 QUANT TRAINING ENGINE — PROFIT/BARRIER V2")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🔐 Feature hash   : {FEATURE_HASH}")
    print(f"🎯 Target version : {TARGET_VERSION}")

    df, dataset_metadata, features, training_window = load_data()
    print(f"✔ Dataset loaded | Rows: {len(df):,}")
    print(
        f"🔒 Training window : {training_window['mode']} | "
        f"fit_end={training_window['fit_end']} | "
        f"backtest_safe_from={training_window['backtest_safe_from'] or 'FORWARD ONLY'}"
    )

    models = {}
    report = {}

    for symbol in SYMBOLS:
        print("\n" + "═" * 80)
        print(f"📈 SYMBOL PIPELINE: {symbol}")
        print("═" * 80)
        data = df[df["symbol"] == symbol].sort_values("time").reset_index(drop=True)
        if len(data) < 5000:
            raise RuntimeError(f"Insufficient rows for {symbol}: {len(data):,}")

        X = data[features]
        y = data["target_class"].to_numpy(dtype=np.int32)
        best_r = data["target_best_r"].to_numpy(dtype=np.float64)

        counts = data["target_class"].value_counts().sort_index().to_dict()
        print(f"📊 Rows         : {len(data):,}")
        print(f"🎯 Class counts : {counts}")

        splitter = TimeSeriesSplit(n_splits=5, gap=MAX_TARGET_HORIZON_BARS)
        fold_metrics = []

        for i, (train_idx, val_idx) in enumerate(splitter.split(X), 1):
            print("\n" + "·" * 60)
            print(f"🔁 FOLD {i}/5 | purge={MAX_TARGET_HORIZON_BARS} bars")
            print("·" * 60)

            model = create_model()
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                sample_weight=compute_weights(y[train_idx], best_r[train_idx]),
                verbose=False,
            )
            proba = model.predict_proba(X.iloc[val_idx])
            metrics = evaluate_probabilities(
                model,
                y[val_idx],
                proba,
                data["target_buy_r"].to_numpy()[val_idx],
                data["target_sell_r"].to_numpy()[val_idx],
            )
            fold_metrics.append(metrics)
            _print_metrics("📊", metrics)

        avg = aggregate_fold_metrics(fold_metrics)
        _print_metrics("✅ CV POOLED", avg)

        final_model = create_model()
        final_model.fit(
            X,
            y,
            sample_weight=compute_weights(y, best_r),
            verbose=False,
        )
        models[symbol] = final_model
        report[symbol] = {
            **avg,
            "rows": int(len(data)),
            "class_counts": {str(k): int(v) for k, v in counts.items()},
            "start": str(data["time"].min()),
            "end": str(data["time"].max()),
        }

    artifact = create_model_artifact(
        models=models,
        symbols=SYMBOLS,
        timeframe=TIMEFRAME_NAME,
        training_metadata=dataset_metadata,
        metrics=report,
        model_params=MODEL_PARAMS,
        training_window=training_window,
    )
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print("\n" + "═" * 80)
    print("📊 FINAL SYSTEM REPORT")
    print("═" * 80)
    for symbol, stats in report.items():
        print(
            f"{symbol:<8} | BAL_ACC={stats['balanced_acc']:.4f} "
            f"F1={stats['macro_f1']:.4f} WIN={stats['win_rate']*100:.1f}% "
            f"AVG_R={stats['avg_r']:.4f} PF_R={stats['profit_factor_r']:.3f}"
        )
    print(f"\n✔ CONTRACT MODEL SAVED: {MODEL_PATH}")
    print("NOTE: R metrics are label-simulator units. Production backtest is the money-P/L authority.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    train()
