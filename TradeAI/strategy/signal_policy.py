from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from Trade_Bot_Training.direction_model import DirectionModel
from TradeAI.strategy.cost_model import current_all_in_cost_pips, load_symbol_cost_spec
from TradeAI.strategy.strategy_config import POLICY_VERSION, symbol_policy_dir


@dataclass(frozen=True)
class SignalDecision:
    symbol: str
    action: str
    reason: str
    p_up: float
    p_down: float
    direction_confidence: float

    # Compatibility fields for the current runtime/UI.
    meta_probability_profitable: float
    predicted_net_atr: float
    long_probability_profitable: float
    short_probability_profitable: float
    long_predicted_net_atr: float
    short_predicted_net_atr: float
    policy_threshold: float

    spread_pips: float
    all_in_cost_pips: float

    def to_dict(self) -> dict:
        return asdict(self)


class SignalPolicy:
    """Precision-first entry policy. No sizing, SL/TP or trade management."""

    def __init__(self, symbol: str, policy_dir: str | Path | None = None) -> None:
        self.symbol = str(symbol).upper()
        self.policy_dir = (
            Path(policy_dir)
            if policy_dir is not None
            else symbol_policy_dir(self.symbol)
        )
        self.policy_path = self.policy_dir / "policy.json"

        if not self.policy_path.exists():
            raise FileNotFoundError(f"Missing policy: {self.policy_path}")

        self.payload = json.loads(self.policy_path.read_text(encoding="utf-8"))

        if self.payload.get("policy_version") != POLICY_VERSION:
            raise RuntimeError("Policy version mismatch. Rebuild the strategy policy.")

        if str(self.payload.get("symbol", "")).upper() != self.symbol:
            raise RuntimeError("Policy symbol mismatch.")

        self.direction_model = DirectionModel(self.symbol)
        self.cost_spec = load_symbol_cost_spec(self.symbol)

        self.trade_enabled = bool(self.payload.get("trade_enabled", False))
        self.live_enabled = bool(self.payload.get("live_enabled", False))
        self.live_disabled_reason = self.payload.get("live_disabled_reason")

        self.minimum_confidence = self.payload.get("minimum_confidence")
        self.minimum_edge_atr = self.payload.get("minimum_edge_atr")

        if self.trade_enabled:
            self.minimum_confidence = float(self.minimum_confidence)
            self.minimum_edge_atr = float(self.minimum_edge_atr)

    def score_frame(self, features: pd.DataFrame, spread_pips) -> pd.DataFrame:
        pred = self.direction_model.predict_proba(features).reset_index(drop=True)

        spread = np.asarray(spread_pips, dtype=float)
        if spread.ndim == 0:
            spread = np.full(len(features), float(spread), dtype=float)

        if len(spread) != len(features):
            raise ValueError("spread_pips length mismatch.")

        if not np.isfinite(spread).all() or (spread < 0.0).any():
            raise ValueError("spread_pips must be finite and non-negative.")

        p_up = pred["p_up"].to_numpy(dtype=float)
        p_down = pred["p_down"].to_numpy(dtype=float)
        confidence = pred["confidence"].to_numpy(dtype=float)
        expected_move = pred["expected_move_atr"].to_numpy(dtype=float)
        disagreement = pred["ensemble_disagreement"].to_numpy(dtype=float)

        choose_long = p_up >= 0.5
        side_sign = np.where(choose_long, 1.0, -1.0)
        expected_side_atr = side_sign * expected_move

        all_in = current_all_in_cost_pips(self.symbol, spread)

        close = pd.to_numeric(features["close"], errors="coerce").to_numpy(dtype=float)
        atr_pct = pd.to_numeric(
            features["m5_atr_pct_14"],
            errors="coerce",
        ).to_numpy(dtype=float)

        atr_pips = close * atr_pct / self.cost_spec.pip_size
        cost_atr = all_in / atr_pips
        edge_proxy_atr = expected_side_atr - cost_atr

        finite = (
            np.isfinite(confidence)
            & np.isfinite(edge_proxy_atr)
            & np.isfinite(all_in)
        )

        if self.trade_enabled:
            take = (
                finite
                & (confidence >= self.minimum_confidence)
                & (edge_proxy_atr >= self.minimum_edge_atr)
            )
        else:
            take = np.zeros(len(features), dtype=bool)

        action = np.where(
            take,
            np.where(choose_long, "BUY_CANDIDATE", "SELL_CANDIDATE"),
            "NO_TRADE",
        )

        if not self.trade_enabled:
            reason = np.full(
                len(features),
                "NO_TRADE_POLICY_DISABLED",
                dtype=object,
            )
        else:
            reason = np.where(
                ~finite,
                "NO_TRADE_INVALID_INPUT",
                np.where(
                    confidence < self.minimum_confidence,
                    "NO_TRADE_CONFIDENCE",
                    np.where(
                        edge_proxy_atr < self.minimum_edge_atr,
                        "NO_TRADE_NET_EDGE",
                        "POLICY_ACCEPTED",
                    ),
                ),
            )

        selected_probability = np.where(choose_long, p_up, p_down)

        return pd.DataFrame(
            {
                "symbol": self.symbol,
                "action": action,
                "reason": reason,
                "selected_side": np.where(choose_long, "LONG", "SHORT"),
                "p_up": p_up,
                "p_down": p_down,
                "direction_confidence": confidence,
                "ensemble_disagreement": disagreement,
                "expected_move_atr": expected_move,
                "expected_side_atr": expected_side_atr,
                "edge_proxy_atr": edge_proxy_atr,
                "cost_atr": cost_atr,
                "meta_probability_profitable": selected_probability,
                "predicted_net_atr": edge_proxy_atr,
                "long_probability_profitable": p_up,
                "short_probability_profitable": p_down,
                "long_predicted_net_atr": expected_move - cost_atr,
                "short_predicted_net_atr": -expected_move - cost_atr,
                "policy_threshold": np.full(
                    len(features),
                    self.minimum_confidence if self.trade_enabled else 1.01,
                    dtype=float,
                ),
                "spread_pips": spread,
                "all_in_cost_pips": all_in,
            },
            index=features.index,
        )

    def decide(self, features: pd.DataFrame, *, spread_pips: float) -> SignalDecision:
        if len(features) != 1:
            raise ValueError("decide() expects exactly one row.")

        row = self.score_frame(features, spread_pips).iloc[0]
        action = str(row["action"])
        reason = str(row["reason"])

        if not self.live_enabled:
            action = "NO_TRADE"
            reason = "NO_TRADE_LIVE_DISABLED"
            if self.live_disabled_reason:
                reason += ": " + str(self.live_disabled_reason)

        return SignalDecision(
            symbol=self.symbol,
            action=action,
            reason=reason,
            p_up=float(row["p_up"]),
            p_down=float(row["p_down"]),
            direction_confidence=float(row["direction_confidence"]),
            meta_probability_profitable=float(row["meta_probability_profitable"]),
            predicted_net_atr=float(row["predicted_net_atr"]),
            long_probability_profitable=float(row["long_probability_profitable"]),
            short_probability_profitable=float(row["short_probability_profitable"]),
            long_predicted_net_atr=float(row["long_predicted_net_atr"]),
            short_predicted_net_atr=float(row["short_predicted_net_atr"]),
            policy_threshold=float(row["policy_threshold"]),
            spread_pips=float(row["spread_pips"]),
            all_in_cost_pips=float(row["all_in_cost_pips"]),
        )
