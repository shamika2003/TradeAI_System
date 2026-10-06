from __future__ import annotations

import json
import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier

from Trade_Bot_Training.precision_config import (
    ACTION_CONFIDENCE,
    CLASS_DOWN,
    CLASS_HOLD,
    CLASS_NAMES,
    CLASS_UP,
    EARLY_STOPPING_ROUNDS,
    MODEL_PARAMS,
    PRODUCTION_MIN_ACTION_ROWS,
    PRODUCTION_MIN_DIRECTIONAL_ACCURACY,
    PRODUCTION_MIN_GOOD_MONTH_FRACTION,
    PRODUCTION_MIN_MEAN_SIGNED_MOVE_ATR,
    PRODUCTION_MIN_SIDE_ACCURACY,
    PRODUCTION_MIN_SIDE_ROWS,
    PRODUCTION_MONTH_MIN_ACCURACY,
    PRODUCTION_MONTH_MIN_ROWS,
    PURGE_BARS,
    SYMBOLS,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    ensure_directories,
    model_dir,
    report_dir,
)
from Trade_Bot_Training.precision_feature_engine import precision_feature_columns
from Trade_Bot_Training.precision_dataset_builder import build_symbol_frame


def _split(df: pd.DataFrame):
    n = len(df)
    train_end = int(n * TRAIN_FRACTION)
    validation_end = train_end + int(n * VALIDATION_FRACTION)

    train = df.iloc[: max(0, train_end - PURGE_BARS)].copy()
    validation = df.iloc[train_end : max(train_end, validation_end - PURGE_BARS)].copy()
    test = df.iloc[validation_end:].copy()
    if min(len(train), len(validation), len(test)) <= 0:
        raise RuntimeError("Chronological split produced an empty partition.")
    return train, validation, test


def _action_metrics(rows: pd.DataFrame, probs: np.ndarray) -> tuple[dict, pd.DataFrame]:
    pred = probs.argmax(axis=1).astype(np.int8)
    confidence = probs.max(axis=1)
    action = (pred != CLASS_HOLD) & (confidence >= ACTION_CONFIDENCE)
    move = rows["future_move_atr"].to_numpy(dtype=float)
    endpoint_direction = np.where(move >= 0.0, CLASS_UP, CLASS_DOWN)

    scored = pd.DataFrame(
        {
            "decision_time": pd.to_datetime(rows["decision_time"], utc=True),
            "target_class": rows["target_class"].to_numpy(dtype=np.int8),
            "future_move_atr": move,
            "p_down": probs[:, CLASS_DOWN],
            "p_hold": probs[:, CLASS_HOLD],
            "p_up": probs[:, CLASS_UP],
            "predicted_class": pred,
            "confidence": confidence,
            "action": action,
        }
    )

    overall = {
        "rows": int(len(rows)),
        "three_class_accuracy": float(accuracy_score(rows["target_class"], pred)),
        "three_class_balanced_accuracy": float(
            balanced_accuracy_score(rows["target_class"], pred)
        ),
        "confusion_matrix": confusion_matrix(
            rows["target_class"], pred, labels=[CLASS_DOWN, CLASS_HOLD, CLASS_UP]
        ).tolist(),
    }

    action_rows = int(action.sum())
    if action_rows:
        signed = np.where(pred[action] == CLASS_UP, move[action], -move[action])
        overall.update(
            {
                "action_rows": action_rows,
                "action_coverage": float(action.mean()),
                "directional_accuracy": float((pred[action] == endpoint_direction[action]).mean()),
                "mean_signed_move_atr": float(np.mean(signed)),
                "median_signed_move_atr": float(np.median(signed)),
            }
        )
    else:
        overall.update(
            {
                "action_rows": 0,
                "action_coverage": 0.0,
                "directional_accuracy": None,
                "mean_signed_move_atr": None,
                "median_signed_move_atr": None,
            }
        )

    side_stats = {}
    for cls in (CLASS_DOWN, CLASS_UP):
        mask = action & (pred == cls)
        n_side = int(mask.sum())
        side_stats[CLASS_NAMES[cls]] = {
            "rows": n_side,
            "directional_accuracy": (
                float((pred[mask] == endpoint_direction[mask]).mean()) if n_side else None
            ),
        }
    overall["sides"] = side_stats

    scored["month"] = scored["decision_time"].dt.to_period("M").astype(str)
    monthly_records = []
    for month, part in scored[scored["action"]].groupby("month"):
        pr = part["predicted_class"].to_numpy(dtype=np.int8)
        mv = part["future_move_atr"].to_numpy(dtype=float)
        truth = np.where(mv >= 0.0, CLASS_UP, CLASS_DOWN)
        signed = np.where(pr == CLASS_UP, mv, -mv)
        monthly_records.append(
            {
                "month": month,
                "rows": int(len(part)),
                "directional_accuracy": float((pr == truth).mean()),
                "mean_signed_move_atr": float(np.mean(signed)),
            }
        )
    overall["monthly"] = monthly_records
    return overall, scored


def _production_gate(metrics: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics["action_rows"] < PRODUCTION_MIN_ACTION_ROWS:
        reasons.append(f"action rows {metrics['action_rows']} < {PRODUCTION_MIN_ACTION_ROWS}")
    if (metrics["directional_accuracy"] or 0.0) < PRODUCTION_MIN_DIRECTIONAL_ACCURACY:
        reasons.append(
            f"directional accuracy {metrics['directional_accuracy']} < {PRODUCTION_MIN_DIRECTIONAL_ACCURACY}"
        )
    if (metrics["mean_signed_move_atr"] or -999.0) < PRODUCTION_MIN_MEAN_SIGNED_MOVE_ATR:
        reasons.append(
            f"mean signed move {metrics['mean_signed_move_atr']} ATR < {PRODUCTION_MIN_MEAN_SIGNED_MOVE_ATR}"
        )

    for side in ("DOWN", "UP"):
        s = metrics["sides"][side]
        if s["rows"] < PRODUCTION_MIN_SIDE_ROWS:
            reasons.append(f"{side} rows {s['rows']} < {PRODUCTION_MIN_SIDE_ROWS}")
        if (s["directional_accuracy"] or 0.0) < PRODUCTION_MIN_SIDE_ACCURACY:
            reasons.append(
                f"{side} accuracy {s['directional_accuracy']} < {PRODUCTION_MIN_SIDE_ACCURACY}"
            )

    qualifying = [m for m in metrics["monthly"] if m["rows"] >= PRODUCTION_MONTH_MIN_ROWS]
    if qualifying:
        good = [m for m in qualifying if m["directional_accuracy"] >= PRODUCTION_MONTH_MIN_ACCURACY]
        fraction = len(good) / len(qualifying)
        if fraction < PRODUCTION_MIN_GOOD_MONTH_FRACTION:
            reasons.append(
                f"good-month fraction {fraction:.3f} < {PRODUCTION_MIN_GOOD_MONTH_FRACTION}"
            )
    else:
        reasons.append("no month has enough high-confidence actions for stability testing")

    return not reasons, reasons


def train_symbol(symbol: str) -> dict:
    symbol = symbol.upper()
    print("\n" + "=" * 78)
    print(f"TRAIN SELECTIVE PRECISION MODEL: {symbol}")
    print("=" * 78)

    df, dataset_summary = build_symbol_frame(symbol, verbose=False)
    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")
    feature_columns = precision_feature_columns(df)
    train, validation, test = _split(df)

    X_train = train[feature_columns].astype("float32")
    y_train = train["target_class"].astype("int8")
    X_validation = validation[feature_columns].astype("float32")
    y_validation = validation["target_class"].astype("int8")

    tune_params = dict(MODEL_PARAMS)
    tune_params["early_stopping_rounds"] = EARLY_STOPPING_ROUNDS
    tuning = XGBClassifier(**tune_params)
    tuning.fit(
        X_train,
        y_train,
        eval_set=[(X_validation, y_validation)],
        verbose=False,
    )
    selected_trees = int(getattr(tuning, "best_iteration", MODEL_PARAMS["n_estimators"] - 1)) + 1
    print(f"Early-stop selected    : {selected_trees} trees")

    # Keep the early-stopped chronological TRAIN model as the evaluated candidate.
    # Validation selected only the tree count; the final TEST remains unseen by fit.
    final_model = tuning
    X_test = test[feature_columns].astype("float32")
    probs = final_model.predict_proba(X_test)
    del X_test
    gc.collect()

    metrics, scored = _action_metrics(test, probs)
    production_ready, reasons = _production_gate(metrics)

    mdir = model_dir(symbol)
    rdir = report_dir(symbol)
    mdir.mkdir(parents=True, exist_ok=True)
    rdir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(mdir / "model.json")

    metadata = {
        "symbol": symbol,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_type": "three_class_selective_xgboost",
        "class_map": {"DOWN": CLASS_DOWN, "HOLD": CLASS_HOLD, "UP": CLASS_UP},
        "action_confidence": ACTION_CONFIDENCE,
        "selected_trees": selected_trees,
        "feature_columns": feature_columns,
        "split": {
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "test_start": str(test["decision_time"].iloc[0]),
            "test_end": str(test["decision_time"].iloc[-1]),
            "purge_bars": PURGE_BARS,
        },
        "final_test": metrics,
        "production_ready_by_historical_gate": production_ready,
        "production_gate_failures": reasons,
        "important": (
            "Historical gate success is not a substitute for post-training forward validation. "
            "Do not promote to automatic live execution until new post-lock data also passes."
        ),
    }
    (mdir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (rdir / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    scored.to_csv(rdir / "test_predictions.csv.gz", index=False, compression="gzip")
    pd.DataFrame(metrics["monthly"]).to_csv(rdir / "test_monthly.csv", index=False)

    print(f"Selected trees         : {selected_trees}")
    print(f"Final test period      : {metadata['split']['test_start']} -> {metadata['split']['test_end']}")
    print(f"High-conf actions      : {metrics['action_rows']:,} ({metrics['action_coverage']:.3%})")
    print(f"Directional accuracy   : {metrics['directional_accuracy']:.2%}")
    print(f"Mean signed move       : {metrics['mean_signed_move_atr']:+.3f} ATR")
    print(f"DOWN accuracy / rows   : {metrics['sides']['DOWN']['directional_accuracy']:.2%} / {metrics['sides']['DOWN']['rows']:,}")
    print(f"UP accuracy / rows     : {metrics['sides']['UP']['directional_accuracy']:.2%} / {metrics['sides']['UP']['rows']:,}")
    print(f"Historical prod gate   : {'PASS' if production_ready else 'FAIL'}")
    if reasons:
        for reason in reasons:
            print("  -", reason)
    print("Model                  :", mdir / "model.json")
    return metadata


def train_all() -> None:
    ensure_directories()
    all_results = {symbol: train_symbol(symbol) for symbol in SYMBOLS}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": {
            s: {
                "directional_accuracy": r["final_test"]["directional_accuracy"],
                "action_rows": r["final_test"]["action_rows"],
                "action_coverage": r["final_test"]["action_coverage"],
                "mean_signed_move_atr": r["final_test"]["mean_signed_move_atr"],
                "historical_gate": r["production_ready_by_historical_gate"],
                "gate_failures": r["production_gate_failures"],
            }
            for s, r in all_results.items()
        },
    }
    path = report_dir(SYMBOLS[0]).parent / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSummary:", path)


if __name__ == "__main__":
    train_all()
