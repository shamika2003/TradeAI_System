# filename: Trade_Bot_Traning/holdout_test.py

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config_model import DATA_PATH, DATASET_METADATA_PATH, SYMBOLS
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from training_utils import compute_weights, create_model, evaluate_probabilities
from shared.tradeai_core.target_definition import MAX_TARGET_HORIZON_BARS, TARGET_VERSION


def _validate_metadata():
    with open(DATASET_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise RuntimeError("Holdout feature schema mismatch")
    if metadata.get("feature_hash") != FEATURE_HASH:
        raise RuntimeError("Holdout feature hash mismatch")
    if metadata.get("target_version") != TARGET_VERSION:
        raise RuntimeError("Holdout target version mismatch; run relabel_dataset.py first")


def _print_metrics(m: dict):
    print(f"🎯 Balanced ACC : {m['balanced_acc']:.4f}")
    print(f"🧠 Macro F1     : {m['macro_f1']:.4f}")
    print(f"📈 Trades       : {m['trades']:,} ({m['coverage']*100:.2f}% coverage)")
    print(f"🏆 Win rate     : {m['win_rate']*100:.2f}%")
    print(f"💰 Avg R/trade  : {m['avg_r']:.4f}")
    print(f"⚖ Profit factor: {m['profit_factor_r']:.3f}")
    print(f"Σ Total R       : {m['total_r']:.1f}")


def run_holdout():
    print("\n" + "═" * 80)
    print("🧪 UNSEEN HOLDOUT — PROFIT/BARRIER TARGET")
    print("═" * 80)
    _validate_metadata()

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    features = FeatureTransformer().get_feature_list()
    required = features + ["target_class", "target_buy_r", "target_sell_r", "target_best_r", "symbol", "time"]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required, inplace=True)

    summary = {}
    for symbol in SYMBOLS:
        print("\n" + "═" * 80)
        print(f"📈 SYMBOL HOLDOUT: {symbol}")
        print("═" * 80)
        data = df[df["symbol"] == symbol].sort_values("time").reset_index(drop=True)
        split = int(len(data) * 0.80)
        train_end = split - MAX_TARGET_HORIZON_BARS
        train = data.iloc[:train_end]
        test = data.iloc[split:]

        print(f"📊 Train rows : {len(train):,}")
        print(f"🧹 Purge gap  : {MAX_TARGET_HORIZON_BARS} bars")
        print(f"📊 Test rows  : {len(test):,}")
        print(f"🕒 Test range : {test['time'].min()} → {test['time'].max()}")

        model = create_model()
        y_train = train["target_class"].to_numpy(dtype=np.int32)
        model.fit(
            train[features],
            y_train,
            sample_weight=compute_weights(y_train, train["target_best_r"].to_numpy()),
            verbose=False,
        )
        proba = model.predict_proba(test[features])
        metrics = evaluate_probabilities(
            model,
            test["target_class"].to_numpy(dtype=np.int32),
            proba,
            test["target_buy_r"].to_numpy(),
            test["target_sell_r"].to_numpy(),
        )
        _print_metrics(metrics)
        summary[symbol] = metrics

    print("\n" + "═" * 80)
    print("📊 FINAL HOLDOUT SUMMARY")
    print("═" * 80)
    for symbol, m in summary.items():
        print(
            f"{symbol:<8} | WIN={m['win_rate']*100:5.1f}% | AVG_R={m['avg_r']:+.4f} | "
            f"PF_R={m['profit_factor_r']:.3f} | TRADES={m['trades']:,}"
        )
    print("\nThese are target-simulator R metrics, not final dollar profit.")
    print("The next production backtest includes real TradeAI sizing, spread, slippage and commission.")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    run_holdout()
