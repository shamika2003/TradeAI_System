# filename: Trade_Bot_Traning/training_utils.py

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score
from xgboost import XGBClassifier

from config_model import MIN_CONFIDENCE, MODEL_PARAMS, SIGNAL_THRESHOLD
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, SELL_CLASS


def create_model():
    return XGBClassifier(**MODEL_PARAMS)


def compute_weights(y, best_r=None):
    y = np.asarray(y, dtype=np.int32)
    n = len(y)
    if n == 0:
        return np.array([], dtype=np.float64)

    counts = {cls: max(1, int(np.sum(y == cls))) for cls in (SELL_CLASS, HOLD_CLASS, BUY_CLASS)}
    weights = np.ones(n, dtype=np.float64)

    for cls in (SELL_CLASS, HOLD_CLASS, BUY_CLASS):
        weights[y == cls] = n / (3.0 * counts[cls])

    if best_r is not None:
        strength = np.asarray(best_r, dtype=np.float64)
        strength = np.nan_to_num(strength, nan=0.0, posinf=2.0, neginf=0.0)
        strength = np.clip(np.abs(strength), 0.0, 2.0)
        weights *= 1.0 + 0.20 * strength

    return np.clip(weights, 0.25, 4.0)


def probability_columns(model, proba):
    classes = list(model.classes_)
    index = {int(cls): i for i, cls in enumerate(classes)}
    required = (SELL_CLASS, HOLD_CLASS, BUY_CLASS)
    missing = [cls for cls in required if cls not in index]
    if missing:
        raise RuntimeError(f"Model probability output missing classes: {missing}")

    return (
        proba[:, index[SELL_CLASS]],
        proba[:, index[HOLD_CLASS]],
        proba[:, index[BUY_CLASS]],
    )


def actions_from_probabilities(
    model,
    proba,
    *,
    min_confidence: float | None = None,
    signal_threshold: float | None = None,
):
    p_sell, p_hold, p_buy = probability_columns(model, proba)
    edge = p_buy - p_sell
    directional_conf = np.maximum(p_buy, p_sell)

    if min_confidence is None:
        min_confidence = MIN_CONFIDENCE
    if signal_threshold is None:
        signal_threshold = SIGNAL_THRESHOLD

    actions = np.full(len(proba), HOLD_CLASS, dtype=np.int32)
    directional = (
        (directional_conf > p_hold)
        & (directional_conf >= float(min_confidence))
        & (np.abs(edge) >= float(signal_threshold))
    )
    actions[directional & (edge > 0)] = BUY_CLASS
    actions[directional & (edge < 0)] = SELL_CLASS

    return actions, edge, directional_conf, p_hold


def evaluate_probabilities(
    model,
    y_true,
    proba,
    buy_r,
    sell_r,
    *,
    min_confidence: float | None = None,
    signal_threshold: float | None = None,
):
    y_true = np.asarray(y_true, dtype=np.int32)
    buy_r = np.asarray(buy_r, dtype=np.float64)
    sell_r = np.asarray(sell_r, dtype=np.float64)

    actions, _, _, _ = actions_from_probabilities(
        model,
        proba,
        min_confidence=min_confidence,
        signal_threshold=signal_threshold,
    )

    balanced_acc = float(balanced_accuracy_score(y_true, actions))
    macro_f1 = float(f1_score(y_true, actions, average="macro", zero_division=0))

    trade_mask = actions != HOLD_CLASS
    trades = int(trade_mask.sum())
    samples = int(len(actions))
    coverage = float(trades / samples) if samples else 0.0

    rewards = np.zeros(samples, dtype=np.float64)
    rewards[actions == BUY_CLASS] = buy_r[actions == BUY_CLASS]
    rewards[actions == SELL_CLASS] = sell_r[actions == SELL_CLASS]
    trade_rewards = rewards[trade_mask]

    if trades:
        wins = trade_rewards[trade_rewards > 0]
        losses = trade_rewards[trade_rewards < 0]
        gross_win = float(wins.sum()) if len(wins) else 0.0
        gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0
        win_count = int(np.sum(trade_rewards > 0))
        win_rate = float(win_count / trades)
        avg_r = float(np.mean(trade_rewards))
        total_r = float(np.sum(trade_rewards))
    else:
        gross_win = 0.0
        gross_loss = 0.0
        win_count = 0
        profit_factor = 0.0
        win_rate = 0.0
        avg_r = 0.0
        total_r = 0.0

    return {
        "balanced_acc": balanced_acc,
        "macro_f1": macro_f1,
        "samples": samples,
        "trades": trades,
        "wins": win_count,
        "coverage": coverage,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "total_r": total_r,
        "gross_win_r": gross_win,
        "gross_loss_r": gross_loss,
        "profit_factor_r": float(profit_factor),
    }


def aggregate_fold_metrics(folds: list[dict]) -> dict:
    if not folds:
        return {
            "balanced_acc": 0.0,
            "macro_f1": 0.0,
            "samples": 0,
            "trades": 0,
            "wins": 0,
            "coverage": 0.0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
            "gross_win_r": 0.0,
            "gross_loss_r": 0.0,
            "profit_factor_r": 0.0,
        }

    samples = int(sum(m.get("samples", 0) for m in folds))
    trades = int(sum(m.get("trades", 0) for m in folds))
    wins = int(sum(m.get("wins", 0) for m in folds))
    total_r = float(sum(m.get("total_r", 0.0) for m in folds))
    gross_win = float(sum(m.get("gross_win_r", 0.0) for m in folds))
    gross_loss = float(sum(m.get("gross_loss_r", 0.0) for m in folds))

    # Classification metrics are sample-weighted; trading metrics are pooled
    # from the actual combined trade population rather than averaged by fold.
    def weighted(key):
        if samples <= 0:
            return 0.0
        return float(sum(m.get(key, 0.0) * m.get("samples", 0) for m in folds) / samples)

    return {
        "balanced_acc": weighted("balanced_acc"),
        "macro_f1": weighted("macro_f1"),
        "samples": samples,
        "trades": trades,
        "wins": wins,
        "coverage": float(trades / samples) if samples else 0.0,
        "win_rate": float(wins / trades) if trades else 0.0,
        "avg_r": float(total_r / trades) if trades else 0.0,
        "total_r": total_r,
        "gross_win_r": gross_win,
        "gross_loss_r": gross_loss,
        "profit_factor_r": float(gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0),
    }
