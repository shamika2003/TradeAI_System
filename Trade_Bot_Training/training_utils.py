from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

from config_model import (
    CLASSIFIER_PARAMS,
    META_MIN_TRAIN_ROWS,
    META_TRAIN_SETUP_MIN_GAP,
    META_TRAIN_SETUP_MIN_SCORE,
    RECENCY_HALF_LIFE_ROWS,
    RECENCY_MIN_WEIGHT,
    REGRESSOR_PARAMS,
)
from shared.tradeai_core.setup_engine import setup_candidate_masks
from shared.tradeai_core.decision_policy import SIGNAL_EPISODE_RESET_BARS
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, SELL_CLASS


@dataclass(frozen=True)
class OpportunityPredictions:
    buy_tp_probability: np.ndarray
    sell_tp_probability: np.ndarray
    buy_expected_r: np.ndarray
    sell_expected_r: np.ndarray


def _recency_weights(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=np.float64)
    age = (n - 1) - np.arange(n, dtype=np.float64)
    half = max(1.0, float(RECENCY_HALF_LIFE_ROWS))
    w = np.exp(math.log(0.5) * age / half)
    return np.clip(w, float(RECENCY_MIN_WEIGHT), 1.0)


def _binary_weights(y: np.ndarray) -> np.ndarray:
    return _recency_weights(len(y))


def _regression_weights(r: np.ndarray) -> np.ndarray:
    return _recency_weights(len(r))


def _safe_direction_mask(X, y, direction: str) -> np.ndarray:
    buy_mask, sell_mask = setup_candidate_masks(
        X,
        min_score=float(META_TRAIN_SETUP_MIN_SCORE),
        min_gap=float(META_TRAIN_SETUP_MIN_GAP),
    )
    mask = buy_mask if str(direction).upper() == "BUY" else sell_mask
    y = np.asarray(y)
    # A meta-model trained on too few rows or a one-class sample is unstable.
    # Fail open to all structurally scored rows for training only; runtime still
    # requires the calibrated setup gate before a trade can be taken.
    if int(mask.sum()) < int(META_MIN_TRAIN_ROWS) or len(np.unique(y[mask])) < 2:
        return np.ones(len(y), dtype=bool)
    return mask


def create_opportunity_bundle() -> dict:
    return {
        "architecture": "structural_meta_opportunity_v3",
        "buy_classifier": XGBClassifier(**CLASSIFIER_PARAMS),
        "sell_classifier": XGBClassifier(**CLASSIFIER_PARAMS),
        "buy_regressor": XGBRegressor(**REGRESSOR_PARAMS),
        "sell_regressor": XGBRegressor(**REGRESSOR_PARAMS),
        "calibration": {},
        "training_meta": {},
    }


def fit_opportunity_bundle(bundle: dict, X, data) -> dict:
    buy_hit = np.asarray(data["target_buy_quality_hit"], dtype=np.int8)
    sell_hit = np.asarray(data["target_sell_quality_hit"], dtype=np.int8)
    buy_r = np.asarray(data["target_buy_r"], dtype=np.float64)
    sell_r = np.asarray(data["target_sell_r"], dtype=np.float64)

    buy_mask = _safe_direction_mask(X, buy_hit, "BUY")
    sell_mask = _safe_direction_mask(X, sell_hit, "SELL")

    bundle["buy_classifier"].fit(
        X.loc[buy_mask], buy_hit[buy_mask],
        sample_weight=_binary_weights(buy_hit[buy_mask]), verbose=False,
    )
    bundle["sell_classifier"].fit(
        X.loc[sell_mask], sell_hit[sell_mask],
        sample_weight=_binary_weights(sell_hit[sell_mask]), verbose=False,
    )
    bundle["buy_regressor"].fit(
        X.loc[buy_mask], buy_r[buy_mask],
        sample_weight=_regression_weights(buy_r[buy_mask]), verbose=False,
    )
    bundle["sell_regressor"].fit(
        X.loc[sell_mask], sell_r[sell_mask],
        sample_weight=_regression_weights(sell_r[sell_mask]), verbose=False,
    )
    bundle["training_meta"] = {
        "buy_rows": int(buy_mask.sum()),
        "sell_rows": int(sell_mask.sum()),
        "buy_positive_rate": float(np.mean(buy_hit[buy_mask])) if buy_mask.any() else 0.0,
        "sell_positive_rate": float(np.mean(sell_hit[sell_mask])) if sell_mask.any() else 0.0,
        "setup_min_score": float(META_TRAIN_SETUP_MIN_SCORE),
        "setup_min_gap": float(META_TRAIN_SETUP_MIN_GAP),
    }
    return bundle


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    x = np.clip(np.asarray(x, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def _apply_probability_calibration(p, spec):
    if not isinstance(spec, dict) or spec.get("kind") != "platt":
        return np.asarray(p, dtype=np.float64)
    return _sigmoid(float(spec.get("a", 1.0)) * _logit(p) + float(spec.get("b", 0.0)))


def _apply_ev_calibration(x, spec):
    x = np.asarray(x, dtype=np.float64)
    if not isinstance(spec, dict) or spec.get("kind") != "linear":
        return x
    return float(spec.get("slope", 1.0)) * x + float(spec.get("intercept", 0.0))


def _fit_platt(raw_p, y):
    raw_p = np.asarray(raw_p, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    if len(raw_p) < 100 or len(np.unique(y)) < 2:
        return {"kind": "identity"}
    X = _logit(raw_p).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
    model.fit(X, y)
    return {
        "kind": "platt",
        "a": float(model.coef_[0, 0]),
        "b": float(model.intercept_[0]),
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
    }


def _fit_linear(raw, y):
    raw = np.asarray(raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(raw) & np.isfinite(y)
    raw, y = raw[mask], y[mask]
    if len(raw) < 100 or float(np.var(raw)) < 1e-10:
        return {"kind": "identity"}
    slope = float(np.cov(raw, y, ddof=0)[0, 1] / max(np.var(raw), 1e-12))
    slope = float(np.clip(slope, 0.0, 2.0))
    intercept = float(np.mean(y) - slope * np.mean(raw))
    return {
        "kind": "linear",
        "slope": slope,
        "intercept": intercept,
        "rows": int(len(y)),
    }


def fit_prediction_calibration(bundle: dict, X, data) -> dict:
    raw = predict_opportunities(bundle, X, apply_calibration=False)
    buy_hit = np.asarray(data["target_buy_quality_hit"], dtype=np.int8)
    sell_hit = np.asarray(data["target_sell_quality_hit"], dtype=np.int8)
    buy_mask = _safe_direction_mask(X, buy_hit, "BUY")
    sell_mask = _safe_direction_mask(X, sell_hit, "SELL")

    bundle["calibration"] = {
        "buy_probability": _fit_platt(raw.buy_tp_probability[buy_mask], buy_hit[buy_mask]),
        "sell_probability": _fit_platt(raw.sell_tp_probability[sell_mask], sell_hit[sell_mask]),
        "buy_expected_r": _fit_linear(raw.buy_expected_r[buy_mask], np.asarray(data["target_buy_r"], dtype=np.float64)[buy_mask]),
        "sell_expected_r": _fit_linear(raw.sell_expected_r[sell_mask], np.asarray(data["target_sell_r"], dtype=np.float64)[sell_mask]),
    }
    return bundle


def predict_opportunities(bundle: dict, X, *, apply_calibration: bool = True) -> OpportunityPredictions:
    p_buy = bundle["buy_classifier"].predict_proba(X)[:, 1].astype(np.float64)
    p_sell = bundle["sell_classifier"].predict_proba(X)[:, 1].astype(np.float64)
    buy_r = bundle["buy_regressor"].predict(X).astype(np.float64)
    sell_r = bundle["sell_regressor"].predict(X).astype(np.float64)
    if apply_calibration:
        cal = bundle.get("calibration") or {}
        p_buy = _apply_probability_calibration(p_buy, cal.get("buy_probability"))
        p_sell = _apply_probability_calibration(p_sell, cal.get("sell_probability"))
        buy_r = _apply_ev_calibration(buy_r, cal.get("buy_expected_r"))
        sell_r = _apply_ev_calibration(sell_r, cal.get("sell_expected_r"))
    buy_r = np.clip(buy_r, -1.25, 3.00)
    sell_r = np.clip(sell_r, -1.25, 3.00)
    return OpportunityPredictions(p_buy, p_sell, buy_r, sell_r)


def choose_opportunities(
    pred: OpportunityPredictions,
    *,
    min_tp_probability: float,
    min_expected_r: float,
    min_ev_gap: float,
    min_probability_gap: float,
    setup_frame=None,
    min_setup_score: float = 0.0,
    min_setup_gap: float = 0.0,
    allow_buy: bool = True,
    allow_sell: bool = True,
):
    p_buy = pred.buy_tp_probability
    p_sell = pred.sell_tp_probability
    ev_buy = pred.buy_expected_r
    ev_sell = pred.sell_expected_r

    if setup_frame is None:
        setup_buy = np.ones(len(p_buy), dtype=bool)
        setup_sell = np.ones(len(p_buy), dtype=bool)
    else:
        setup_buy, setup_sell = setup_candidate_masks(
            setup_frame,
            min_score=float(min_setup_score),
            min_gap=float(min_setup_gap),
        )

    buy_ok = bool(allow_buy) & setup_buy & (p_buy >= min_tp_probability) & (ev_buy >= min_expected_r)
    sell_ok = bool(allow_sell) & setup_sell & (p_sell >= min_tp_probability) & (ev_sell >= min_expected_r)

    # Probability is used as a confidence modifier rather than as a substitute
    # for expected payoff.  A side with positive EV but structurally weak setup
    # cannot pass because the setup gate is applied above.
    buy_score = ev_buy * (0.55 + p_buy)
    sell_score = ev_sell * (0.55 + p_sell)
    ev_gap = buy_score - sell_score
    prob_gap = p_buy - p_sell

    actions = np.full(len(p_buy), HOLD_CLASS, dtype=np.int32)

    # When both directions are qualified, require a genuine cross-direction
    # advantage.  When policy validation has proved only one side of a symbol,
    # do not let an unqualified opposite model veto that otherwise valid setup.
    if allow_buy and allow_sell:
        buy_choice = buy_ok & (ev_gap >= min_ev_gap) & (prob_gap >= min_probability_gap)
        sell_choice = sell_ok & (ev_gap <= -min_ev_gap) & (prob_gap <= -min_probability_gap)
    elif allow_buy:
        buy_choice = buy_ok
        sell_choice = np.zeros(len(p_buy), dtype=bool)
    elif allow_sell:
        buy_choice = np.zeros(len(p_buy), dtype=bool)
        sell_choice = sell_ok
    else:
        buy_choice = np.zeros(len(p_buy), dtype=bool)
        sell_choice = np.zeros(len(p_buy), dtype=bool)

    actions[buy_choice] = BUY_CLASS
    actions[sell_choice] = SELL_CLASS
    return actions, buy_score, sell_score, ev_gap



def decluster_actions(actions, reset_bars: int = SIGNAL_EPISODE_RESET_BARS):
    """Keep one decision per continuous setup episode.

    Repeated BUY/SELL predictions on adjacent M5 bars describe the same market
    opportunity, not independent statistical evidence.  A same-direction
    episode becomes eligible again only after ``reset_bars`` consecutive HOLD
    decisions.  A direct direction reversal starts a new episode immediately.
    """
    src = np.asarray(actions, dtype=np.int32)
    out = np.full(len(src), HOLD_CLASS, dtype=np.int32)
    active_direction = HOLD_CLASS
    hold_run = int(reset_bars)
    reset_bars = max(1, int(reset_bars))

    for i, action in enumerate(src):
        action = int(action)
        if action == HOLD_CLASS:
            hold_run += 1
            if hold_run >= reset_bars:
                active_direction = HOLD_CLASS
            continue

        if active_direction == HOLD_CLASS or action != active_direction:
            out[i] = action
            active_direction = action
        hold_run = 0

    return out

def evaluate_opportunities(
    data,
    pred: OpportunityPredictions,
    *,
    min_tp_probability: float,
    min_expected_r: float,
    min_ev_gap: float,
    min_probability_gap: float,
    min_setup_score: float = 0.0,
    min_setup_gap: float = 0.0,
    allow_buy: bool = True,
    allow_sell: bool = True,
    decluster: bool = False,
    episode_reset_bars: int = SIGNAL_EPISODE_RESET_BARS,
) -> dict:
    actions, buy_score, sell_score, ev_gap = choose_opportunities(
        pred,
        min_tp_probability=min_tp_probability,
        min_expected_r=min_expected_r,
        min_ev_gap=min_ev_gap,
        min_probability_gap=min_probability_gap,
        setup_frame=data,
        min_setup_score=min_setup_score,
        min_setup_gap=min_setup_gap,
        allow_buy=allow_buy,
        allow_sell=allow_sell,
    )
    if decluster:
        actions = decluster_actions(actions, episode_reset_bars)

    buy_r = np.asarray(data["target_buy_r"], dtype=np.float64)
    sell_r = np.asarray(data["target_sell_r"], dtype=np.float64)
    rewards = np.zeros(len(actions), dtype=np.float64)
    rewards[actions == BUY_CLASS] = buy_r[actions == BUY_CLASS]
    rewards[actions == SELL_CLASS] = sell_r[actions == SELL_CLASS]
    mask = actions != HOLD_CLASS
    trade_rewards = rewards[mask]
    wins = trade_rewards[trade_rewards > 0]
    losses = trade_rewards[trade_rewards < 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = abs(float(losses.sum())) if len(losses) else 0.0
    pf = gross_win / gross_loss if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)

    def _auc(y, p, candidate_mask=None):
        y = np.asarray(y, dtype=np.int8)
        p = np.asarray(p, dtype=np.float64)
        if candidate_mask is not None:
            y, p = y[candidate_mask], p[candidate_mask]
        if len(y) < 2 or len(np.unique(y)) < 2:
            return 0.5
        return float(roc_auc_score(y, p))

    broad_buy, broad_sell = setup_candidate_masks(
        data,
        min_score=float(META_TRAIN_SETUP_MIN_SCORE),
        min_gap=float(META_TRAIN_SETUP_MIN_GAP),
    )
    trades = int(mask.sum())
    return {
        "samples": int(len(actions)),
        "trades": trades,
        "coverage": float(trades / len(actions)) if len(actions) else 0.0,
        "wins": int(np.sum(trade_rewards > 0)) if trades else 0,
        "win_rate": float(np.mean(trade_rewards > 0)) if trades else 0.0,
        "avg_r": float(np.mean(trade_rewards)) if trades else 0.0,
        "total_r": float(np.sum(trade_rewards)) if trades else 0.0,
        "profit_factor_r": float(pf),
        "gross_win_r": gross_win,
        "gross_loss_r": gross_loss,
        "buy_auc": _auc(data["target_buy_quality_hit"], pred.buy_tp_probability, broad_buy),
        "sell_auc": _auc(data["target_sell_quality_hit"], pred.sell_tp_probability, broad_sell),
        "mean_abs_ev_gap": float(np.mean(np.abs(ev_gap))) if len(ev_gap) else 0.0,
        "setup_buy_candidates": int(broad_buy.sum()),
        "setup_sell_candidates": int(broad_sell.sum()),
    }


def aggregate_opportunity_metrics(folds: list[dict]) -> dict:
    if not folds:
        return {
            "samples": 0, "trades": 0, "coverage": 0.0, "wins": 0,
            "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0,
            "profit_factor_r": 0.0, "buy_auc": 0.5, "sell_auc": 0.5,
        }
    samples = sum(int(x.get("samples", 0)) for x in folds)
    trades = sum(int(x.get("trades", 0)) for x in folds)
    wins = sum(int(x.get("wins", 0)) for x in folds)
    total_r = sum(float(x.get("total_r", 0.0)) for x in folds)
    gw = sum(float(x.get("gross_win_r", 0.0)) for x in folds)
    gl = sum(float(x.get("gross_loss_r", 0.0)) for x in folds)

    def weighted(k, d=0.0):
        return (
            sum(float(x.get(k, d)) * int(x.get("samples", 0)) for x in folds) / samples
            if samples else d
        )

    return {
        "samples": samples,
        "trades": trades,
        "coverage": float(trades / samples) if samples else 0.0,
        "wins": wins,
        "win_rate": float(wins / trades) if trades else 0.0,
        "avg_r": float(total_r / trades) if trades else 0.0,
        "total_r": float(total_r),
        "profit_factor_r": float(gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0),
        "gross_win_r": float(gw),
        "gross_loss_r": float(gl),
        "buy_auc": weighted("buy_auc", 0.5),
        "sell_auc": weighted("sell_auc", 0.5),
    }


# ---------------------------------------------------------------------------
# Legacy multiclass helpers retained only so older diagnostics/tests import.
# ---------------------------------------------------------------------------
def create_model():
    return create_opportunity_bundle()


def compute_weights(y, best_r=None):
    y = np.asarray(y, dtype=np.int32)
    if len(y) == 0:
        return np.array([], dtype=np.float64)
    counts = {cls: max(1, int(np.sum(y == cls))) for cls in (SELL_CLASS, HOLD_CLASS, BUY_CLASS)}
    max_count = max(counts.values())
    weights = np.ones(len(y), dtype=np.float64)
    for cls in (SELL_CLASS, HOLD_CLASS, BUY_CLASS):
        mult = min(2.0, max(1.0, math.sqrt(max_count / counts[cls])))
        if cls == HOLD_CLASS:
            mult = 1.0
        weights[y == cls] = mult
    return np.clip(weights, 0.5, 2.5)


def probability_columns(model, proba):
    classes = list(model.classes_)
    index = {int(cls): i for i, cls in enumerate(classes)}
    return proba[:, index[SELL_CLASS]], proba[:, index[HOLD_CLASS]], proba[:, index[BUY_CLASS]]


def actions_from_probabilities(model, proba, *, min_confidence=0.55, signal_threshold=0.10, hold_margin=0.0):
    p_sell, p_hold, p_buy = probability_columns(model, proba)
    edge = p_buy - p_sell
    conf = np.maximum(p_buy, p_sell)
    actions = np.full(len(proba), HOLD_CLASS, dtype=np.int32)
    directional = (
        (conf >= p_hold + float(hold_margin))
        & (conf >= float(min_confidence))
        & (np.abs(edge) >= float(signal_threshold))
    )
    actions[directional & (edge > 0)] = BUY_CLASS
    actions[directional & (edge < 0)] = SELL_CLASS
    return actions, edge, conf, p_hold


def evaluate_probabilities(model, y_true, proba, buy_r, sell_r, *, min_confidence=0.55, signal_threshold=0.10, hold_margin=0.0):
    actions, _, _, _ = actions_from_probabilities(
        model, proba,
        min_confidence=min_confidence,
        signal_threshold=signal_threshold,
        hold_margin=hold_margin,
    )
    buy_r = np.asarray(buy_r, dtype=np.float64)
    sell_r = np.asarray(sell_r, dtype=np.float64)
    rewards = np.zeros(len(actions), dtype=np.float64)
    rewards[actions == BUY_CLASS] = buy_r[actions == BUY_CLASS]
    rewards[actions == SELL_CLASS] = sell_r[actions == SELL_CLASS]
    mask = actions != HOLD_CLASS
    tr = rewards[mask]
    gw = float(tr[tr > 0].sum()) if len(tr) else 0.0
    gl = abs(float(tr[tr < 0].sum())) if len(tr) else 0.0
    return {
        "balanced_acc": 0.0, "macro_f1": 0.0, "samples": int(len(actions)),
        "trades": int(mask.sum()), "wins": int(np.sum(tr > 0)) if len(tr) else 0,
        "coverage": float(mask.mean()) if len(actions) else 0.0,
        "win_rate": float(np.mean(tr > 0)) if len(tr) else 0.0,
        "avg_r": float(np.mean(tr)) if len(tr) else 0.0,
        "total_r": float(np.sum(tr)) if len(tr) else 0.0,
        "gross_win_r": gw, "gross_loss_r": gl,
        "profit_factor_r": float(gw / gl) if gl > 0 else (999.0 if gw > 0 else 0.0),
    }


def aggregate_fold_metrics(folds):
    return aggregate_opportunity_metrics(folds)
