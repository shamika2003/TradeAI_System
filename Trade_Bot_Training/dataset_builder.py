from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from Trade_Bot_Training.direction_config import (
    DATASET_METADATA_PATH,
    DATASET_VERSION,
    HORIZON_BARS,
    HORIZON_MINUTES,
    M5_MINUTES,
    MIN_RAW_H1_ROWS,
    MIN_RAW_M5_ROWS,
    MODEL_VERSION,
    PREDICTION_CONTRACT,
    SYMBOLS,
    ensure_directories,
    processed_path,
    raw_path,
)
from Trade_Bot_Training.feature_engine import (
    build_feature_frame,
    model_feature_columns,
)


EXPECTED_M5_DELTA = pd.Timedelta(minutes=M5_MINUTES)
EXPECTED_HORIZON_DELTA = pd.Timedelta(minutes=HORIZON_MINUTES)


def _load_raw(symbol: str, timeframe: str) -> pd.DataFrame:
    path = raw_path(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw dataset: {path}\n"
            "Run the collection step first."
        )

    df = pd.read_csv(path, compression="gzip")
    if "time" not in df.columns:
        raise RuntimeError(f"{path} has no time column.")

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    return df


def _feature_hash(columns: list[str]) -> str:
    return hashlib.sha256("\n".join(columns).encode("utf-8")).hexdigest()


def _build_executable_labels(m5: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Create labels that match the earliest bar-level executable entry.

    At the close of bar t the model becomes available. The first executable
    bar-level proxy is therefore the open of t+1. With HORIZON_BARS=6, the
    target endpoint is the close of t+6, exactly 30 minutes after entry.

    No weekend, holiday or missing-feed gap is allowed inside the target
    window: the t+6 bar must begin exactly 30 minutes after bar t begins.
    """

    required = {"time", "open", "close"}
    missing = required.difference(m5.columns)
    if missing:
        raise RuntimeError(
            "Raw M5 data is missing columns: " + ", ".join(sorted(missing))
        )

    labels = m5[["time", "open", "close"]].copy()

    labels["entry_time"] = labels["time"].shift(-1)
    labels["entry_open"] = labels["open"].shift(-1)

    labels["exit_bar_time"] = labels["time"].shift(-HORIZON_BARS)
    labels["exit_time"] = labels["exit_bar_time"] + EXPECTED_M5_DELTA
    labels["exit_close"] = labels["close"].shift(-HORIZON_BARS)

    labels["entry_gap_minutes"] = (
        (labels["entry_time"] - labels["time"]) / pd.Timedelta(minutes=1)
    )
    labels["horizon_gap_minutes"] = (
        (labels["exit_bar_time"] - labels["time"]) / pd.Timedelta(minutes=1)
    )

    labels["is_contiguous_horizon"] = (
        (labels["entry_time"] - labels["time"] == EXPECTED_M5_DELTA)
        & (labels["exit_bar_time"] - labels["time"] == EXPECTED_HORIZON_DELTA)
    )

    valid_prices = (
        labels["entry_open"].notna()
        & labels["exit_close"].notna()
        & (labels["entry_open"] > 0.0)
    )

    raw_return = labels["exit_close"] / labels["entry_open"] - 1.0
    labels["future_return"] = raw_return.where(
        labels["is_contiguous_horizon"] & valid_prices,
        np.nan,
    )

    labels["future_move_price"] = (
        labels["exit_close"] - labels["entry_open"]
    ).where(labels["is_contiguous_horizon"] & valid_prices, np.nan)

    labels["target_up"] = np.where(
        labels["future_return"] > 0.0,
        1.0,
        np.where(labels["future_return"] < 0.0, 0.0, np.nan),
    )

    metadata = {
        "raw_rows": int(len(labels)),
        "non_contiguous_horizon_rows": int((~labels["is_contiguous_horizon"]).sum()),
        "zero_move_rows": int((labels["future_return"] == 0.0).sum()),
    }
    return labels, metadata


def build_symbol_dataset(symbol: str) -> dict:
    symbol = str(symbol).upper()

    print()
    print("=" * 78)
    print(f"BUILDING EXECUTABLE DIRECTION DATASET: {symbol}")
    print("=" * 78)

    m5 = _load_raw(symbol, "M5")
    h1 = _load_raw(symbol, "H1")

    if len(m5) < MIN_RAW_M5_ROWS:
        raise RuntimeError(
            f"{symbol}: only {len(m5):,} M5 bars. Need at least {MIN_RAW_M5_ROWS:,}."
        )
    if len(h1) < MIN_RAW_H1_ROWS:
        raise RuntimeError(
            f"{symbol}: only {len(h1):,} H1 bars. Need at least {MIN_RAW_H1_ROWS:,}."
        )

    m5 = (
        m5.dropna(subset=["time"])
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )
    h1 = (
        h1.dropna(subset=["time"])
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )

    labels, label_meta = _build_executable_labels(m5)
    features = build_feature_frame(m5, h1)

    dataset = features.merge(
        labels[
            [
                "time",
                "entry_time",
                "entry_open",
                "exit_bar_time",
                "exit_time",
                "exit_close",
                "entry_gap_minutes",
                "horizon_gap_minutes",
                "is_contiguous_horizon",
                "future_return",
                "future_move_price",
                "target_up",
            ]
        ],
        how="left",
        on="time",
        validate="one_to_one",
    )

    feature_columns = model_feature_columns(dataset)
    if not feature_columns:
        raise RuntimeError("Feature engine produced no model features.")
    if "m5_atr_pct_14" not in dataset.columns:
        raise RuntimeError("Feature engine did not produce m5_atr_pct_14.")

    # Convert the executable future move into current-ATR units. This becomes
    # the companion regression target. It is not used as an input feature.
    atr_price = dataset["close"].astype(float) * dataset["m5_atr_pct_14"].astype(float)
    dataset["future_move_atr"] = (
        dataset["future_move_price"].astype(float)
        / atr_price.replace(0.0, np.nan)
    )
    dataset["future_abs_move_atr"] = dataset["future_move_atr"].abs()
    dataset["future_move_bps"] = dataset["future_return"].astype(float) * 10_000.0

    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    rows_before_cleaning = len(dataset)

    required_training_columns = feature_columns + [
        "target_up",
        "future_return",
        "future_move_atr",
        "entry_time",
        "exit_time",
    ]
    dataset = dataset.dropna(subset=required_training_columns).copy()

    # Decision time must equal the executable entry proxy timestamp. This is
    # the key contract that the previous target did not satisfy economically.
    decision_time = pd.to_datetime(dataset["decision_time"], utc=True)
    entry_time = pd.to_datetime(dataset["entry_time"], utc=True)
    bad_entry_alignment = int((decision_time != entry_time).sum())
    if bad_entry_alignment:
        raise RuntimeError(
            f"{symbol}: {bad_entry_alignment} rows have decision_time != entry_time."
        )

    invalid_horizon_rows = int((dataset["horizon_gap_minutes"] != HORIZON_MINUTES).sum())
    if invalid_horizon_rows:
        raise RuntimeError(
            f"{symbol}: {invalid_horizon_rows} non-contiguous target windows survived."
        )

    dataset["target_up"] = dataset["target_up"].astype("int8")
    dataset["is_contiguous_horizon"] = dataset["is_contiguous_horizon"].astype(bool)

    float_columns = feature_columns + [
        "entry_open",
        "exit_close",
        "entry_gap_minutes",
        "horizon_gap_minutes",
        "future_return",
        "future_move_price",
        "future_move_atr",
        "future_abs_move_atr",
        "future_move_bps",
    ]
    for column in float_columns:
        if column in dataset.columns:
            dataset[column] = dataset[column].astype("float32")

    output_path = processed_path(symbol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False, compression="gzip")

    up_rows = int((dataset["target_up"] == 1).sum())
    down_rows = int((dataset["target_up"] == 0).sum())
    dropped_rows = rows_before_cleaning - len(dataset)

    summary = {
        "symbol": symbol,
        "raw_merged_rows": int(rows_before_cleaning),
        "rows": int(len(dataset)),
        "dropped_rows": int(dropped_rows),
        "retained_fraction": float(len(dataset) / rows_before_cleaning if rows_before_cleaning else 0.0),
        "non_contiguous_horizon_rows": label_meta["non_contiguous_horizon_rows"],
        "zero_move_rows": label_meta["zero_move_rows"],
        "invalid_horizon_rows_after_cleaning": invalid_horizon_rows,
        "bad_entry_alignment_rows": bad_entry_alignment,
        "up_rows": up_rows,
        "down_rows": down_rows,
        "up_fraction": float(up_rows / len(dataset) if len(dataset) else 0.0),
        "mean_abs_move_atr": float(dataset["future_abs_move_atr"].mean()),
        "median_abs_move_atr": float(dataset["future_abs_move_atr"].median()),
        "start_decision_time": str(dataset["decision_time"].min()),
        "end_decision_time": str(dataset["decision_time"].max()),
        "processed_path": str(output_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "feature_hash": _feature_hash(feature_columns),
    }

    print(f"Rows          : {len(dataset):,}")
    print(f"Dropped       : {dropped_rows:,}")
    print(f"Gap windows   : {label_meta['non_contiguous_horizon_rows']:,}")
    print(f"Bad alignment : {bad_entry_alignment:,}")
    print(f"DOWN          : {down_rows:,}")
    print(f"UP            : {up_rows:,}")
    print(f"UP fraction   : {summary['up_fraction']:.4f}")
    print(f"Median |move| : {summary['median_abs_move_atr']:.3f} ATR")
    print(f"Features      : {len(feature_columns)}")
    print(f"Saved         : {output_path}")

    return summary


def build_all() -> None:
    ensure_directories()

    symbol_summaries: dict[str, dict] = {}
    canonical_features: list[str] | None = None
    canonical_hash: str | None = None

    for symbol in SYMBOLS:
        summary = build_symbol_dataset(symbol)
        symbol_summaries[symbol] = summary

        if canonical_features is None:
            canonical_features = summary["feature_columns"]
            canonical_hash = summary["feature_hash"]
        else:
            if summary["feature_columns"] != canonical_features:
                raise RuntimeError("Feature columns differ between symbols.")
            if summary["feature_hash"] != canonical_hash:
                raise RuntimeError("Feature hash differs between symbols.")

    metadata = {
        "dataset_version": DATASET_VERSION,
        "model_version": MODEL_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "prediction_contract": PREDICTION_CONTRACT,
        "horizon_bars": HORIZON_BARS,
        "horizon_minutes": HORIZON_MINUTES,
        "entry_definition": "Open of the immediately following contiguous M5 candle.",
        "exit_definition": (
            f"Close after {HORIZON_BARS} contiguous M5 bars / {HORIZON_MINUTES} minutes from entry."
        ),
        "target": {"DOWN": 0, "UP": 1},
        "target_definition": (
            "UP when exit_close > executable entry_open; DOWN when exit_close < executable "
            "entry_open. Exact zero moves and non-contiguous target windows are removed."
        ),
        "regression_target": (
            "Signed executable entry-to-exit price move normalized by the causal current M5 ATR."
        ),
        "sample_weight_definition": "None. Direction classification is unweighted.",
        "historical_spread_features_in_model": False,
        "feature_count": len(canonical_features or []),
        "feature_columns": canonical_features or [],
        "feature_hash": canonical_hash,
        "symbols": symbol_summaries,
    }

    DATASET_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print()
    print("=" * 78)
    print("EXECUTABLE DIRECTION DATASET BUILD COMPLETE")
    print("=" * 78)
    print(f"Metadata: {DATASET_METADATA_PATH}")


if __name__ == "__main__":
    build_all()
