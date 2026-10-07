from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from Trade_Bot_Training.direction_config import (
    CONFIDENCE_LEVELS,
    HORIZON_MINUTES,
    REPORT_ROOT,
    SYMBOLS,
    ensure_directories,
    symbol_report_dir,
)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def _load_predictions(symbol: str) -> pd.DataFrame:
    path = symbol_report_dir(symbol) / "test_predictions.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing test predictions: {path}\nRun model training first."
        )

    df = pd.read_csv(path, compression="gzip")
    if "prediction" not in df.columns and "direction" in df.columns:
        df["prediction"] = df["direction"]
    required = {
        "decision_time",
        "target_up",
        "future_move_atr",
        "p_up",
        "p_down",
        "prediction",
        "confidence",
        "expected_move_atr",
    }
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"{symbol}: missing prediction columns: " + ", ".join(sorted(missing))
        )

    df["decision_time"] = pd.to_datetime(df["decision_time"], utc=True, errors="coerce")
    numeric = [
        "target_up",
        "future_move_atr",
        "p_up",
        "p_down",
        "confidence",
        "expected_move_atr",
    ]
    for column in numeric:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["decision_time", "prediction", *numeric]).copy()
    df["target_up"] = df["target_up"].astype(int)
    df["prediction"] = df["prediction"].astype(str).str.upper()
    df["predicted_up"] = (df["prediction"] == "UP").astype(int)
    df["correct"] = df["predicted_up"] == df["target_up"]
    df["signed_realized_move_atr"] = np.where(
        df["predicted_up"] == 1,
        df["future_move_atr"],
        -df["future_move_atr"],
    )
    df["month"] = df["decision_time"].dt.to_period("M").astype(str)
    return df


def _threshold_stats(df: pd.DataFrame, total_rows: int, threshold: float) -> dict:
    selected = df.loc[df["confidence"] >= threshold].copy()
    if selected.empty:
        return {
            "confidence_threshold": float(threshold),
            "rows": 0,
            "coverage": 0.0,
            "accuracy": None,
            "mean_signed_realized_move_atr": None,
            "median_signed_realized_move_atr": None,
            "mean_expected_move_atr": None,
        }

    return {
        "confidence_threshold": float(threshold),
        "rows": int(len(selected)),
        "coverage": float(len(selected) / total_rows if total_rows else 0.0),
        "accuracy": float(selected["correct"].mean()),
        "mean_signed_realized_move_atr": float(selected["signed_realized_move_atr"].mean()),
        "median_signed_realized_move_atr": float(selected["signed_realized_move_atr"].median()),
        "mean_expected_move_atr": float(selected["expected_move_atr"].mean()),
    }


def _month_stats(month: str, df: pd.DataFrame) -> dict:
    try:
        auc = float(roc_auc_score(df["target_up"], df["p_up"]))
    except ValueError:
        auc = None

    return {
        "month": month,
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(df["target_up"], df["predicted_up"])),
        "roc_auc": auc,
        "mean_signed_realized_move_atr": float(df["signed_realized_move_atr"].mean()),
        "mean_expected_move_atr": float(df["expected_move_atr"].mean()),
    }


def validate_symbol(symbol: str) -> dict:
    symbol = str(symbol).upper()

    print()
    print("=" * 78)
    print(f"DIRECTION MODEL VALIDATION: {symbol}")
    print("=" * 78)

    df = _load_predictions(symbol)
    total_rows = len(df)

    threshold_rows = [
        {"symbol": symbol, **_threshold_stats(df, total_rows, threshold)}
        for threshold in CONFIDENCE_LEVELS
    ]
    monthly_rows = [
        {"symbol": symbol, **_month_stats(month, month_df)}
        for month, month_df in df.groupby("month", sort=True)
    ]

    try:
        overall_auc = float(roc_auc_score(df["target_up"], df["p_up"]))
    except ValueError:
        overall_auc = None

    correlation = None
    if len(df) > 1 and df["future_move_atr"].std() > 0 and df["expected_move_atr"].std() > 0:
        correlation = float(
            np.corrcoef(df["future_move_atr"], df["expected_move_atr"])[0, 1]
        )

    summary = {
        "symbol": symbol,
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_minutes": HORIZON_MINUTES,
        "test_rows": int(total_rows),
        "accuracy": float(df["correct"].mean()),
        "roc_auc": overall_auc,
        "mean_signed_realized_move_atr": float(df["signed_realized_move_atr"].mean()),
        "expected_vs_realized_move_correlation": correlation,
        "confidence": threshold_rows,
        "monthly": monthly_rows,
        "important": (
            "This validates the untouched fixed-horizon model test only. "
            "Confidence rows are diagnostics, not a trading policy and not tuned here."
        ),
    }

    report_dir = symbol_report_dir(symbol)
    report_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(threshold_rows).to_csv(
        report_dir / "confidence_validation.csv",
        index=False,
    )
    pd.DataFrame(monthly_rows).to_csv(
        report_dir / "monthly_validation.csv",
        index=False,
    )
    (report_dir / "validation_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )

    print(f"Rows               : {total_rows:,}")
    print(f"Accuracy           : {summary['accuracy']:.4f}")
    print(f"ROC-AUC            : {summary['roc_auc']}")
    print(f"Signed move        : {summary['mean_signed_realized_move_atr']:+.4f} ATR")
    print(f"Move correlation   : {summary['expected_vs_realized_move_correlation']}")
    print(f"Summary            : {report_dir / 'validation_summary.json'}")

    return summary


def validate_all() -> None:
    ensure_directories()
    summaries = {symbol: validate_symbol(symbol) for symbol in SYMBOLS}

    confidence_frames = []
    monthly_frames = []
    for symbol in SYMBOLS:
        report_dir = symbol_report_dir(symbol)
        confidence_frames.append(pd.read_csv(report_dir / "confidence_validation.csv"))
        monthly_frames.append(pd.read_csv(report_dir / "monthly_validation.csv"))

    if confidence_frames:
        pd.concat(confidence_frames, ignore_index=True).to_csv(
            REPORT_ROOT / "confidence_validation_all.csv",
            index=False,
        )
    if monthly_frames:
        pd.concat(monthly_frames, ignore_index=True).to_csv(
            REPORT_ROOT / "monthly_validation_all.csv",
            index=False,
        )

    payload = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "horizon_minutes": HORIZON_MINUTES,
        "symbols": summaries,
    }
    (REPORT_ROOT / "validation_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("DIRECTION MODEL VALIDATION COMPLETE")
    print("=" * 78)
    print(f"Summary: {REPORT_ROOT / 'validation_summary.json'}")


if __name__ == "__main__":
    validate_all()
