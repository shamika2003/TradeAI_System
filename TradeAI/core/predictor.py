# filename: TradeAI/core/predictor.py

from __future__ import annotations

import os

import joblib
import numpy as np

from analytics.logger import log
from config.settings import (
    MIN_CONFIDENCE,
    MODEL_PATH,
    REQUIRE_CALIBRATED_POLICY,
    REQUIRE_CALIBRATED_RISK_POLICY,
    SIGNAL_THRESHOLD,
    SYMBOLS,
    TIMEFRAME,
)
from core.feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from shared.tradeai_core.model_contract import (
    require_decision_policy,
    require_risk_policy,
    validate_model_artifact,
)
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, SELL_CLASS, TARGET_VERSION


class Predictor:
    """Production multiclass barrier predictor with strict contract validation."""

    _artifact_cache = {}

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.models = None
        self.artifact = None
        self.decision_policy = None
        self.risk_policy = None
        self.feature_engine = FeatureTransformer()
        self.feature_list = self.feature_engine.get_feature_list()
        self.loaded = False
        self._load_model()

    def _load_model(self):
        try:
            path_text = os.fspath(self.model_path)
            absolute_key = os.path.abspath(path_text)
            if not os.path.exists(path_text):
                raise RuntimeError(f"Model missing: {path_text}. Run the training/validation pipeline first.")

            if absolute_key in Predictor._artifact_cache:
                artifact = Predictor._artifact_cache[absolute_key]
            else:
                artifact = joblib.load(path_text)
                validate_model_artifact(
                    artifact,
                    expected_symbols=SYMBOLS,
                    expected_timeframe=f"M{TIMEFRAME}",
                    expected_target_version=TARGET_VERSION,
                )
                Predictor._artifact_cache[absolute_key] = artifact

            if REQUIRE_CALIBRATED_POLICY:
                policy = require_decision_policy(artifact)
            else:
                try:
                    policy = require_decision_policy(artifact)
                except Exception:
                    policy = {
                        "symbols": {
                            symbol: {
                                "enabled": True,
                                "min_confidence": MIN_CONFIDENCE,
                                "signal_threshold": SIGNAL_THRESHOLD,
                            }
                            for symbol in SYMBOLS
                        }
                    }

            if REQUIRE_CALIBRATED_RISK_POLICY:
                risk_policy = require_risk_policy(artifact)
            else:
                try:
                    risk_policy = require_risk_policy(artifact)
                except Exception:
                    risk_policy = {
                        "version": "fallback",
                        "risk_percent": 1.0,
                        "source": "settings_fallback",
                    }

            self.artifact = artifact
            self.models = artifact["models"]
            self.decision_policy = policy
            self.risk_policy = risk_policy
            self.loaded = True
            log(
                f"INFO | MODEL CONTRACT OK schema={FEATURE_SCHEMA_VERSION} "
                f"hash={FEATURE_HASH[:12]}... target={TARGET_VERSION} "
                f"policy={'CALIBRATED' if REQUIRE_CALIBRATED_POLICY else 'OPTIONAL'}"
            )
        except Exception as e:
            self.models = None
            self.artifact = None
            self.decision_policy = None
            self.risk_policy = None
            self.loaded = False
            log(f"ERROR | Model contract/load failed: {e}")
            raise RuntimeError(f"TradeAI model startup validation failed: {e}") from e

    def prepare_features(self, df):
        if df is None or df.empty:
            return None
        missing = [f for f in self.feature_list if f not in df.columns]
        if missing:
            log(f"ERROR | Missing features {missing}")
            return None
        X = df[self.feature_list].copy()
        for col in X.columns:
            X[col] = np.asarray(X[col], dtype=np.float32)
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        return X

    def get_model(self, symbol):
        if not isinstance(self.models, dict):
            return None
        model = self.models.get(symbol)
        if model is None:
            log(f"ERROR | No model for {symbol}")
        return model

    def get_symbol_policy(self, symbol):
        symbol_map = (self.decision_policy or {}).get("symbols", {})
        policy = symbol_map.get(symbol)
        if policy is None:
            if REQUIRE_CALIBRATED_POLICY:
                raise RuntimeError(f"No calibrated decision policy for {symbol}")
            return {
                "enabled": True,
                "min_confidence": MIN_CONFIDENCE,
                "signal_threshold": SIGNAL_THRESHOLD,
            }
        return policy

    def get_risk_policy(self):
        if self.risk_policy is None:
            raise RuntimeError("No calibrated production risk policy loaded")
        return dict(self.risk_policy)

    @staticmethod
    def _class_probabilities(model, row_proba):
        class_index = {int(cls): i for i, cls in enumerate(model.classes_)}
        for cls in (SELL_CLASS, HOLD_CLASS, BUY_CLASS):
            if cls not in class_index:
                raise RuntimeError(f"Model missing probability class {cls}")
        return (
            float(row_proba[class_index[SELL_CLASS]]),
            float(row_proba[class_index[HOLD_CLASS]]),
            float(row_proba[class_index[BUY_CLASS]]),
        )

    def predict(self, df, symbol):
        try:
            if not self.loaded:
                return None
            X = self.prepare_features(df)
            if X is None or X.empty:
                return None
            model = self.get_model(symbol)
            if model is None:
                return None

            policy = self.get_symbol_policy(symbol)
            if not bool(policy.get("enabled", True)):
                return {
                    "signal": 0.0,
                    "confidence": 0.0,
                    "p_buy": 0.0,
                    "p_hold": 1.0,
                    "p_sell": 0.0,
                    "signal_threshold": float(policy["signal_threshold"]),
                    "min_confidence": float(policy["min_confidence"]),
                    "policy_enabled": False,
                }

            proba = model.predict_proba(X.tail(1))[0]
            p_sell, p_hold, p_buy = self._class_probabilities(model, proba)

            signal = p_buy - p_sell
            confidence = max(p_buy, p_sell)

            # HOLD dominance remains an unconditional no-trade decision.
            if p_hold >= confidence:
                signal = 0.0

            return {
                "signal": float(signal),
                "confidence": float(confidence),
                "p_buy": p_buy,
                "p_hold": p_hold,
                "p_sell": p_sell,
                "signal_threshold": float(policy["signal_threshold"]),
                "min_confidence": float(policy["min_confidence"]),
                "policy_enabled": True,
            }
        except Exception as e:
            log(f"ERROR | Predictor failed {symbol}: {e}")
            return None
