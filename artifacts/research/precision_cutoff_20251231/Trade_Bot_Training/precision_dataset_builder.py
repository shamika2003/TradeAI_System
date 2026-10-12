from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from Trade_Bot_Training.precision_config import (
    CLASS_DOWN,
    CLASS_HOLD,
    CLASS_UP,
    HORIZON_BARS,
    HORIZON_MINUTES,
    MIN_MEANINGFUL_MOVE_ATR,
    REPORT_ROOT,
    SYMBOLS,
    ensure_directories,
    raw_path,
)
from Trade_Bot_Training.precision_feature_engine import (
    _clean_m5,
    build_precision_features,
    precision_feature_columns,
)


def build_symbol_frame(symbol: str, *, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    symbol = symbol.upper()
    path = raw_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw M5 data: {path}")

    if verbose:
        print("\n" + "=" * 78)
        print(f"BUILD SELECTIVE PRECISION FRAME: {symbol}")
        print("=" * 78)

    raw = pd.read_csv(
        path,
        compression="gzip",
        usecols=["time", "open", "high", "low", "close", "tick_volume"],
    )
    clean = _clean_m5(raw)
    frame = build_precision_features(raw)
    if len(clean) != len(frame) or not clean["time"].reset_index(drop=True).equals(frame["time"].reset_index(drop=True)):
        raise RuntimeError("Feature/label row alignment mismatch after M5 cleaning.")

    entry_time = clean["time"].shift(-1)
    entry_open = clean["open"].shift(-1)
    exit_bar_time = clean["time"].shift(-HORIZON_BARS)
    exit_close = clean["close"].shift(-HORIZON_BARS)
    contiguous = (
        (entry_time - clean["time"] == pd.Timedelta(minutes=5))
        & (exit_bar_time - clean["time"] == pd.Timedelta(minutes=HORIZON_MINUTES))
    )

    move_atr = ((exit_close - entry_open) / frame["current_atr"]).astype("float32")
    feature_columns = precision_feature_columns(frame)
    valid = (
        contiguous.to_numpy(dtype=bool)
        & np.isfinite(move_atr.to_numpy(dtype=float))
        & frame[feature_columns].notna().all(axis=1).to_numpy(dtype=bool)
    )

    X = frame.loc[valid, ["decision_time"] + feature_columns].copy().reset_index(drop=True)
    move = move_atr.loc[valid].reset_index(drop=True)
    target = np.full(len(X), CLASS_HOLD, dtype=np.int8)
    target[move.to_numpy() <= -MIN_MEANINGFUL_MOVE_ATR] = CLASS_DOWN
    target[move.to_numpy() >= MIN_MEANINGFUL_MOVE_ATR] = CLASS_UP
    X["future_move_atr"] = move.to_numpy(dtype=np.float32)
    X["target_class"] = target

    counts = pd.Series(target).value_counts().to_dict()
    feature_hash = hashlib.sha256("\n".join(feature_columns).encode("utf-8")).hexdigest()
    summary = {
        "symbol": symbol,
        "rows": int(len(X)),
        "feature_count": int(len(feature_columns)),
        "feature_hash": feature_hash,
        "feature_columns": feature_columns,
        "min_meaningful_move_atr": MIN_MEANINGFUL_MOVE_ATR,
        "class_counts": {
            "DOWN": int(counts.get(CLASS_DOWN, 0)),
            "HOLD": int(counts.get(CLASS_HOLD, 0)),
            "UP": int(counts.get(CLASS_UP, 0)),
        },
        "start": str(X["decision_time"].iloc[0]),
        "end": str(X["decision_time"].iloc[-1]),
    }
    if verbose:
        print(f"Rows         : {summary['rows']:,}")
        print(f"Features     : {summary['feature_count']}")
        print(
            "DOWN/HOLD/UP: "
            f"{summary['class_counts']['DOWN']:,} / "
            f"{summary['class_counts']['HOLD']:,} / "
            f"{summary['class_counts']['UP']:,}"
        )
    return X, summary


def build_all() -> None:
    """Validate all raw data and write compact metadata only.

    Full feature matrices are intentionally not written as giant CSV files. The
    train step rebuilds each symbol causally, trains it, then releases memory.
    """
    ensure_directories()
    summaries = {}
    for symbol in SYMBOLS:
        frame, summary = build_symbol_frame(symbol, verbose=True)
        summaries[symbol] = summary
        del frame

    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "label_contract": (
            "DOWN <= -0.60 ATR, HOLD inside +/-0.60 ATR, UP >= +0.60 ATR, "
            "measured from next contiguous M5 open to the close six bars later."
        ),
        "symbols": summaries,
    }
    path = REPORT_ROOT / "dataset_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nDataset audit:", path)


if __name__ == "__main__":
    build_all()
