from __future__ import annotations

import os
import joblib
import numpy as np

from analytics.logger import log
from config.settings import MODEL_PATH, REQUIRE_CALIBRATED_POLICY, REQUIRE_CALIBRATED_RISK_POLICY, SYMBOLS, TIMEFRAME
from core.feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from shared.tradeai_core.model_contract import require_decision_policy, require_risk_policy, validate_model_artifact
from shared.tradeai_core.target_definition import TARGET_VERSION


class Predictor:
    """Directional opportunity predictor used identically in BACKTEST/DEMO/LIVE."""

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
        path = os.fspath(self.model_path)
        key = os.path.abspath(path)
        if not os.path.exists(path):
            raise RuntimeError(f"Model missing: {path}. Run the training/validation pipeline first.")
        artifact = Predictor._artifact_cache.get(key)
        if artifact is None:
            artifact = joblib.load(path)
            validate_model_artifact(
                artifact,
                expected_symbols=SYMBOLS,
                expected_timeframe=f"M{TIMEFRAME}",
                expected_target_version=TARGET_VERSION,
            )
            Predictor._artifact_cache[key] = artifact
        self.artifact = artifact
        self.models = artifact["models"]
        self.decision_policy = require_decision_policy(artifact) if REQUIRE_CALIBRATED_POLICY else artifact.get("decision_policy")
        self.risk_policy = require_risk_policy(artifact) if REQUIRE_CALIBRATED_RISK_POLICY else artifact.get("risk_policy", {"risk_percent": 0.30})
        self.loaded = True
        log(f"INFO | OPPORTUNITY MODEL OK schema={FEATURE_SCHEMA_VERSION} hash={FEATURE_HASH[:12]} target={TARGET_VERSION}")

    def prepare_features(self, df):
        if df is None or df.empty:
            return None
        missing = [f for f in self.feature_list if f not in df.columns]
        if missing:
            log(f"ERROR | Missing features {missing}")
            return None
        X = df[self.feature_list].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.fillna(0, inplace=True)
        return X.astype(np.float32)

    def get_model(self, symbol):
        return (self.models or {}).get(symbol)

    def get_symbol_policy(self, symbol):
        p = ((self.decision_policy or {}).get("symbols") or {}).get(symbol)
        if p is None:
            raise RuntimeError(f"No calibrated opportunity policy for {symbol}")
        return p

    def get_risk_policy(self):
        if self.risk_policy is None:
            raise RuntimeError("No calibrated production risk policy loaded")
        return dict(self.risk_policy)

    @staticmethod
    def _bundle_predict(bundle, X):
        row = X.tail(1)
        p_buy = float(bundle["buy_classifier"].predict_proba(row)[0, 1])
        p_sell = float(bundle["sell_classifier"].predict_proba(row)[0, 1])
        ev_buy = float(bundle["buy_regressor"].predict(row)[0])
        ev_sell = float(bundle["sell_regressor"].predict(row)[0])

        cal = bundle.get("calibration") or {}
        def _cal_p(value, spec):
            if not isinstance(spec, dict) or spec.get("kind") != "platt":
                return value
            value = min(max(float(value), 1e-6), 1.0 - 1e-6)
            logit = np.log(value / (1.0 - value))
            z = float(spec.get("a", 1.0)) * logit + float(spec.get("b", 0.0))
            z = float(np.clip(z, -40.0, 40.0))
            return float(1.0 / (1.0 + np.exp(-z)))

        def _cal_ev(value, spec):
            if not isinstance(spec, dict) or spec.get("kind") != "linear":
                return value
            return float(spec.get("slope", 1.0)) * float(value) + float(spec.get("intercept", 0.0))

        p_buy = _cal_p(p_buy, cal.get("buy_probability"))
        p_sell = _cal_p(p_sell, cal.get("sell_probability"))
        ev_buy = float(np.clip(_cal_ev(ev_buy, cal.get("buy_expected_r")), -1.25, 3.50))
        ev_sell = float(np.clip(_cal_ev(ev_sell, cal.get("sell_expected_r")), -1.25, 3.50))
        buy_score = ev_buy * (0.50 + p_buy)
        sell_score = ev_sell * (0.50 + p_sell)
        return p_buy, p_sell, ev_buy, ev_sell, buy_score, sell_score

    def predict(self, df, symbol):
        try:
            if not self.loaded:
                return None
            X = self.prepare_features(df)
            if X is None or X.empty:
                return None
            bundle = self.get_model(symbol)
            if not isinstance(bundle, dict) or bundle.get("architecture") not in {"directional_opportunity_v1", "directional_opportunity_v2", "structural_meta_opportunity_v3"}:
                raise RuntimeError(f"Invalid opportunity bundle for {symbol}")
            policy = self.get_symbol_policy(symbol)
            if not bool(policy.get("enabled", True)):
                return {
                    "signal": 0.0, "confidence": 0.0, "direction": None,
                    "trade_eligible": False, "policy_enabled": False,
                    "p_buy": 0.0, "p_sell": 0.0, "p_hold": 1.0,
                    "buy_expected_r": 0.0, "sell_expected_r": 0.0,
                    "selected_expected_r": 0.0,
                    "setup_buy_score": 0.0, "setup_sell_score": 0.0,
                    "min_setup_score": float(policy.get("min_setup_score", 1.0)),
                    "min_setup_gap": float(policy.get("min_setup_gap", 1.0)),
                    "allow_buy": bool(policy.get("allow_buy", True)),
                    "allow_sell": bool(policy.get("allow_sell", True)),
                    "min_confidence": float(policy.get("min_tp_probability", 0.99)),
                    "signal_threshold": float(policy.get("min_ev_gap", 9.0)),
                }

            p_buy, p_sell, ev_buy, ev_sell, buy_score, sell_score = self._bundle_predict(bundle, X)
            ev_gap = buy_score - sell_score
            prob_gap = p_buy - p_sell
            min_p = float(policy["min_tp_probability"])
            min_ev = float(policy["min_expected_r"])
            min_gap = float(policy["min_ev_gap"])
            min_pgap = float(policy["min_probability_gap"])
            min_setup = float(policy.get("min_setup_score", 0.0))
            min_setup_gap = float(policy.get("min_setup_gap", 0.0))

            last = X.tail(1).iloc[0]
            setup_buy = float(last.get("setup_buy_score", 0.0))
            setup_sell = float(last.get("setup_sell_score", 0.0))
            setup_gap = setup_buy - setup_sell

            allow_buy = bool(policy.get("allow_buy", True))
            allow_sell = bool(policy.get("allow_sell", True))
            buy_setup_ok = allow_buy and setup_buy >= min_setup and setup_gap >= min_setup_gap
            sell_setup_ok = allow_sell and setup_sell >= min_setup and setup_gap <= -min_setup_gap

            buy_ok = buy_setup_ok and p_buy >= min_p and ev_buy >= min_ev
            sell_ok = sell_setup_ok and p_sell >= min_p and ev_sell >= min_ev

            # Cross-direction conflict gates are meaningful only when both sides
            # of this symbol independently qualified during validation.  For a
            # one-sided policy, an unqualified opposite model cannot veto the
            # proven side at runtime.
            if allow_buy and allow_sell:
                buy_ok = buy_ok and ev_gap >= min_gap and prob_gap >= min_pgap
                sell_ok = sell_ok and ev_gap <= -min_gap and prob_gap <= -min_pgap
            direction = "BUY" if buy_ok else ("SELL" if sell_ok else None)
            confidence = p_buy if direction == "BUY" else (p_sell if direction == "SELL" else max(p_buy, p_sell))
            selected_ev = ev_buy if direction == "BUY" else (ev_sell if direction == "SELL" else max(ev_buy, ev_sell))
            signal = ev_gap if direction is not None else 0.0
            return {
                "signal": float(signal),
                "confidence": float(confidence),
                "direction": direction,
                "trade_eligible": direction is not None,
                "policy_enabled": True,
                "p_buy": float(p_buy), "p_sell": float(p_sell),
                "p_hold": float(max(0.0, 1.0 - max(p_buy, p_sell))),
                "buy_expected_r": float(ev_buy), "sell_expected_r": float(ev_sell),
                "buy_score": float(buy_score), "sell_score": float(sell_score),
                "selected_expected_r": float(selected_ev),
                "setup_buy_score": setup_buy, "setup_sell_score": setup_sell,
                "setup_score_gap": setup_gap,
                "min_tp_probability": min_p, "min_expected_r": min_ev,
                "min_ev_gap": min_gap, "min_probability_gap": min_pgap,
                "min_setup_score": min_setup, "min_setup_gap": min_setup_gap,
                "allow_buy": allow_buy, "allow_sell": allow_sell,
                "min_confidence": min_p, "signal_threshold": min_gap,
            }
        except Exception as e:
            log(f"ERROR | Opportunity predictor failed {symbol}: {e}")
            return None
