from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from Trade_Bot_Training.precision_config import (
    ACTION_CONFIDENCE,
    CLASS_DOWN,
    CLASS_HOLD,
    CLASS_NAMES,
    CLASS_UP,
    model_dir,
)


class PrecisionDirectionModel:
    def __init__(self, symbol: str, path: str | Path | None = None) -> None:
        self.symbol = symbol.upper()
        self.path = Path(path) if path is not None else model_dir(self.symbol)
        metadata_path = self.path / "metadata.json"
        model_path = self.path / "model.json"
        if not metadata_path.exists() or not model_path.exists():
            raise FileNotFoundError(f"Precision model not trained for {self.symbol}: {self.path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_columns = list(self.metadata["feature_columns"])
        self.action_confidence = float(self.metadata.get("action_confidence", ACTION_CONFIDENCE))
        self.historical_gate_pass = bool(self.metadata.get("production_ready_by_historical_gate", False))
        self.model = XGBClassifier()
        self.model.load_model(model_path)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in frame.columns]
        if missing:
            raise ValueError("Missing precision model features: " + ", ".join(missing[:20]))
        X = frame[self.feature_columns].astype("float32")
        if not np.isfinite(X.to_numpy()).all():
            raise ValueError("Precision feature matrix contains NaN or infinite values.")

        probs = self.model.predict_proba(X)
        pred = probs.argmax(axis=1).astype(np.int8)
        conf = probs.max(axis=1)
        tradeable = (pred != CLASS_HOLD) & (conf >= self.action_confidence) & self.historical_gate_pass
        signal = np.array([CLASS_NAMES[int(v)] for v in pred], dtype=object)
        signal = np.where(tradeable, signal, "HOLD")

        return pd.DataFrame(
            {
                "symbol": self.symbol,
                "p_down": probs[:, CLASS_DOWN],
                "p_hold": probs[:, CLASS_HOLD],
                "p_up": probs[:, CLASS_UP],
                "raw_class": [CLASS_NAMES[int(v)] for v in pred],
                "signal": signal,
                "confidence": conf,
                "tradeable": tradeable,
                "historical_gate_pass": self.historical_gate_pass,
            },
            index=frame.index,
        )
