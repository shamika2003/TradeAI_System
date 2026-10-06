from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

from Trade_Bot_Training.direction_config import (
    symbol_model_dir,
)


def _probability_logit(
    probability: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        p
        / (
            1.0 - p
        )
    ).reshape(
        -1,
        1,
    )


class DirectionModel:
    """Read-only active TradeAI direction ensemble."""

    def __init__(
        self,
        symbol: str,
        model_dir: str | Path | None = None,
    ) -> None:
        self.symbol = str(
            symbol
        ).upper()

        self.model_dir = (
            Path(
                model_dir
            )
            if model_dir is not None
            else symbol_model_dir(
                self.symbol
            )
        )

        self.metadata_path = (
            self.model_dir
            / "metadata.json"
        )

        self.calibrator_path = (
            self.model_dir
            / "calibrator.joblib"
        )

        self.move_model_path = (
            self.model_dir
            / "move_model.json"
        )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Missing model metadata: "
                f"{self.metadata_path}"
            )

        self.metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8"
            )
        )

        if (
            str(
                self.metadata.get(
                    "symbol",
                    "",
                )
            ).upper()
            != self.symbol
        ):
            raise RuntimeError(
                "Model symbol mismatch."
            )

        self.feature_columns = list(
            self.metadata.get(
                "feature_columns",
                [],
            )
        )

        if not self.feature_columns:
            raise RuntimeError(
                "Model metadata contains "
                "no feature columns."
            )

        self.horizon_minutes = int(
            self.metadata.get(
                "prediction_horizon_minutes",
                0,
            )
        )

        ensemble_path = (
            self.model_dir
            / "ensemble.json"
        )

        self.models = []
        self.weights = []

        if ensemble_path.exists():
            payload = json.loads(
                ensemble_path.read_text(
                    encoding="utf-8"
                )
            )

            for member in payload.get(
                "members",
                [],
            ):
                model = XGBClassifier()
                model.load_model(
                    self.model_dir
                    / member[
                        "file"
                    ]
                )

                self.models.append(
                    model
                )

                self.weights.append(
                    float(
                        member.get(
                            "weight",
                            1.0,
                        )
                    )
                )

        if not self.models:
            fallback = (
                self.model_dir
                / "model.json"
            )

            model = XGBClassifier()
            model.load_model(
                fallback
            )

            self.models = [
                model
            ]
            self.weights = [
                1.0
            ]

        weights = np.asarray(
            self.weights,
            dtype=float,
        )

        if (
            not np.isfinite(
                weights
            ).all()
            or weights.sum() <= 0
        ):
            raise RuntimeError(
                "Invalid ensemble weights."
            )

        self.weights = (
            weights
            / weights.sum()
        )

        self.calibrator = joblib.load(
            self.calibrator_path
        )

        self.move_model = XGBRegressor()
        self.move_model.load_model(
            self.move_model_path
        )

    def _matrix(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        missing = [
            column
            for column
            in self.feature_columns
            if column not in features.columns
        ]

        if missing:
            raise ValueError(
                "Missing model features: "
                + ", ".join(
                    missing[:20]
                )
            )

        X = (
            features[
                self.feature_columns
            ]
            .copy()
            .astype(
                "float32"
            )
        )

        if not np.isfinite(
            X.to_numpy()
        ).all():
            raise ValueError(
                "Feature matrix contains "
                "NaN or infinite values."
            )

        return X

    def predict_proba(
        self,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        X = self._matrix(
            features
        )

        member_probabilities = (
            np.column_stack(
                [
                    model.predict_proba(
                        X
                    )[:, 1]
                    for model
                    in self.models
                ]
            )
        )

        raw_p_up = np.average(
            member_probabilities,
            axis=1,
            weights=self.weights,
        )

        p_up = (
            self.calibrator.predict_proba(
                _probability_logit(
                    raw_p_up
                )
            )[:, 1]
        )

        p_up = np.clip(
            np.asarray(
                p_up,
                dtype=float,
            ),
            0.0,
            1.0,
        )

        p_down = (
            1.0 - p_up
        )

        expected_move_atr = (
            self.move_model.predict(
                X
            ).astype(float)
        )

        if (
            "m5_atr_pct_14"
            not in features.columns
        ):
            raise ValueError(
                "m5_atr_pct_14 is required."
            )

        atr_pct = pd.to_numeric(
            features.loc[
                X.index,
                "m5_atr_pct_14",
            ],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        expected_move_return = (
            expected_move_atr
            * atr_pct
        )

        return pd.DataFrame(
            {
                "symbol": self.symbol,
                "horizon_minutes": (
                    self.horizon_minutes
                ),
                "p_down": p_down,
                "p_up": p_up,
                "direction": np.where(
                    p_up >= 0.5,
                    "UP",
                    "DOWN",
                ),
                "confidence": np.maximum(
                    p_up,
                    p_down,
                ),
                "expected_move_atr": (
                    expected_move_atr
                ),
                "expected_move_return": (
                    expected_move_return
                ),
                "expected_move_bps": (
                    expected_move_return
                    * 10_000.0
                ),
                "p_up_raw": raw_p_up,
                "ensemble_disagreement": (
                    member_probabilities.std(
                        axis=1
                    )
                ),
            },
            index=features.index,
        )
