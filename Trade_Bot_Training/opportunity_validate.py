from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import stage4_production_validate as s4
from config_model import (
    ACCEPT_NORMAL_MAX_DRAWDOWN_PERCENT,
    ACCEPT_NORMAL_MIN_PROFIT_FACTOR,
    ACCEPT_NORMAL_MIN_TRADES,
    ACCEPT_STRESS_MAX_DRAWDOWN_PERCENT,
    ACCEPT_STRESS_MIN_PROFIT_FACTOR,
    ACCEPT_STRESS_MIN_TRADES,
    CALIBRATION_MAX_COVERAGE,
    QUALIFICATION_TRAIN_FRACTION,
    QUALIFICATION_SCORE_CAL_END_FRACTION,
    POLICY_SELECTION_END_FRACTION,
    DEPLOYMENT_CALIBRATION_END_FRACTION,
    DATA_PATH,
    DATASET_METADATA_PATH,
    MODEL_PATH,
    OPPORTUNITY_EV_GAP_GRID,
    OPPORTUNITY_EXPECTED_R_GRID,
    OPPORTUNITY_PROBABILITY_GRID,
    OPPORTUNITY_PROB_GAP_GRID,
    OPPORTUNITY_SETUP_SCORE_GRID,
    OPPORTUNITY_SETUP_GAP_GRID,
    OPPORTUNITY_DIRECTION_MODES,
    POLICY_CALIBRATION_BALANCE,
    POLICY_CALIBRATION_MIN_PAYOFF_RATIO,
    POLICY_CALIBRATION_MIN_TRADES,
    POLICY_CALIBRATION_MIN_EXECUTED_TRADES,
    POLICY_CALIBRATION_BLOCKS,
    POLICY_CALIBRATION_MIN_ACTIVE_BLOCKS,
    POLICY_CALIBRATION_MIN_TRADES_PER_ACTIVE_BLOCK,
    POLICY_CALIBRATION_MIN_POSITIVE_BLOCK_FRACTION,
    POLICY_CALIBRATION_NORMAL_MAX_DRAWDOWN_PERCENT,
    POLICY_CALIBRATION_NORMAL_MIN_PROFIT_FACTOR,
    POLICY_CALIBRATION_RISK_PERCENT,
    POLICY_CALIBRATION_STRESS_MAX_DRAWDOWN_PERCENT,
    POLICY_CALIBRATION_STRESS_MIN_PROFIT_FACTOR,
    PRIMARY_ACCEPTANCE_BALANCES,
    REPORT_DIR,
    STAGE5_CAL_MIN_TRADES,
    STAGE5_CAL_NORMAL_MAX_DRAWDOWN_PERCENT,
    STAGE5_CAL_NORMAL_MIN_PROFIT_FACTOR,
    STAGE5_CAL_STRESS_MAX_DRAWDOWN_PERCENT,
    STAGE5_CAL_STRESS_MIN_PROFIT_FACTOR,
    STAGE5_RISK_GRID,
    SYMBOLS,
    TIMEFRAME_NAME,
    VALIDATION_BALANCES,
)
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from training_utils import (
    choose_opportunities,
    decluster_actions,
    create_opportunity_bundle,
    evaluate_opportunities,
    fit_opportunity_bundle,
    fit_prediction_calibration,
    predict_opportunities,
)
from shared.tradeai_core.decision_policy import DECISION_POLICY_VERSION
from shared.tradeai_core.model_contract import (
    attach_decision_policy,
    attach_risk_policy,
    require_training_window,
    validate_model_artifact,
)
from shared.tradeai_core.risk_policy import RISK_POLICY_VERSION
from shared.tradeai_core.target_definition import BUY_CLASS, HOLD_CLASS, MAX_TARGET_HORIZON_BARS, SELL_CLASS, TARGET_VERSION
from shared.tradeai_core.training_window import apply_supervised_training_cutoff
from shared.tradeai_core.validation_execution import ValidationExecutionContext


REPORT_JSON = REPORT_DIR / "stage5_risk_validation.json"
RISK_POLICY_JSON = REPORT_DIR / "stage5_risk_policy.json"
DECISION_POLICY_JSON = REPORT_DIR / "stage5_decision_policy.json"
TRADES_CSV = REPORT_DIR / "stage5_primary_trades.csv"


class SettingsProxy:
    def __init__(self, base, risk_percent: float):
        self._base = base
        self.RISK_PERCENT = float(risk_percent)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _validate_dataset_metadata():
    with open(DATASET_METADATA_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    checks = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "timeframe": TIMEFRAME_NAME,
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"Dataset contract mismatch {key}: expected {expected!r}, got {meta.get(key)!r}")
    return meta


def _load_data(cutoff_exclusive):
    _validate_dataset_metadata()
    features = FeatureTransformer().get_feature_list()
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    required = features + [
        "symbol", "time", "open", "high", "low", "close", "atr",
        "target_class", "target_buy_r", "target_sell_r", "target_best_r",
        "target_buy_tp_hit", "target_sell_tp_hit",
        "target_buy_quality_hit", "target_sell_quality_hit",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Opportunity validation dataset missing columns: {missing}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required, inplace=True)
    df, window = apply_supervised_training_cutoff(
        df,
        cutoff_exclusive=cutoff_exclusive,
        label_horizon_bars=MAX_TARGET_HORIZON_BARS,
    )
    return df, features, window.as_dict()


def _with_exact_deployment_models(artifact: dict, deployment_models: dict, deployment_validation: dict) -> dict:
    contract_symbols = set((artifact.get("contract") or {}).get("symbols") or [])
    if set(deployment_models or {}) != contract_symbols:
        raise RuntimeError("Exact deployment model symbols do not match artifact contract")
    promoted = dict(artifact)
    promoted["models"] = dict(deployment_models)
    metadata = dict(promoted.get("training_metadata") or {})
    validation_meta = dict(deployment_validation or {})
    validation_meta["exact_runtime_model"] = True
    metadata["deployment_validation"] = validation_meta
    promoted["training_metadata"] = metadata
    return promoted


def _policy_actions(data, pred, policy):
    return choose_opportunities(
        pred,
        min_tp_probability=float(policy["min_tp_probability"]),
        min_expected_r=float(policy["min_expected_r"]),
        min_ev_gap=float(policy["min_ev_gap"]),
        min_probability_gap=float(policy["min_probability_gap"]),
        setup_frame=data,
        min_setup_score=float(policy.get("min_setup_score", 0.0)),
        min_setup_gap=float(policy.get("min_setup_gap", 0.0)),
        allow_buy=bool(policy.get("allow_buy", True)),
        allow_sell=bool(policy.get("allow_sell", True)),
    )


def _attach_management_state(data: pd.DataFrame, pred, policy: dict) -> pd.DataFrame:
    out = data.copy().reset_index(drop=True)
    _, buy_score, sell_score, ev_gap = _policy_actions(out, pred, policy)
    p_buy = pred.buy_tp_probability
    p_sell = pred.sell_tp_probability
    ev_buy = pred.buy_expected_r
    ev_sell = pred.sell_expected_r
    conf = np.maximum(p_buy, p_sell)

    out["_mgmt_signal"] = ev_gap
    out["_mgmt_confidence"] = conf
    out["_mgmt_min_confidence"] = float(policy.get("min_tp_probability", 1.0))
    out["_mgmt_signal_threshold"] = float(policy.get("min_ev_gap", 999.0))
    out["_mgmt_policy_enabled"] = bool(policy.get("enabled", True))
    out["_mgmt_buy_tp_probability"] = p_buy
    out["_mgmt_sell_tp_probability"] = p_sell
    out["_mgmt_buy_expected_r"] = ev_buy
    out["_mgmt_sell_expected_r"] = ev_sell
    out["_mgmt_min_expected_r"] = float(policy.get("min_expected_r", 999.0))
    out["_mgmt_min_tp_probability"] = float(policy.get("min_tp_probability", 1.0))
    out["_mgmt_setup_buy_score"] = out["setup_buy_score"].to_numpy(dtype=np.float64)
    out["_mgmt_setup_sell_score"] = out["setup_sell_score"].to_numpy(dtype=np.float64)
    out["_mgmt_min_setup_score"] = float(policy.get("min_setup_score", 1.0))
    out["_mgmt_min_setup_gap"] = float(policy.get("min_setup_gap", 1.0))
    return out


def _build_candidates(symbol: str, data: pd.DataFrame, pred, policy: dict) -> list[dict]:
    if not policy.get("enabled", True):
        return []
    actions, buy_score, sell_score, ev_gap = _policy_actions(data, pred, policy)
    actions = decluster_actions(actions)
    candidates = []
    max_idx = len(data) - MAX_TARGET_HORIZON_BARS
    for idx in np.flatnonzero(actions != HOLD_CLASS):
        if idx >= max_idx:
            continue
        action = int(actions[idx])
        direction = "BUY" if action == BUY_CLASS else "SELL"
        confidence = float(pred.buy_tp_probability[idx] if action == BUY_CLASS else pred.sell_tp_probability[idx])
        expected_r = float(pred.buy_expected_r[idx] if action == BUY_CLASS else pred.sell_expected_r[idx])
        candidates.append({
            "time": pd.Timestamp(data.iloc[idx]["time"]),
            "symbol": symbol,
            "index": int(idx),
            "direction": direction,
            "edge": float(abs(ev_gap[idx])),
            "confidence": confidence,
            "expected_r": expected_r,
            "p_buy": float(pred.buy_tp_probability[idx]),
            "p_sell": float(pred.sell_tp_probability[idx]),
            "buy_score": float(buy_score[idx]),
            "sell_score": float(sell_score[idx]),
            "setup_buy_score": float(data.iloc[idx]["setup_buy_score"]),
            "setup_sell_score": float(data.iloc[idx]["setup_sell_score"]),
        })
    return candidates


def _direction_flags(mode: str) -> tuple[bool, bool]:
    mode = str(mode).upper()
    if mode == "BUY":
        return True, False
    if mode == "SELL":
        return False, True
    return True, True


def _block_stability(data, pred, policy, blocks=None):
    """Sparse-aware stability check.

    Zero-trade blocks are not treated as failures.  A quality-first strategy may
    legitimately remain flat for weeks.  Stability is judged only on blocks
    that contain enough actual trades, while still requiring the edge to appear
    in multiple separated periods.
    """
    blocks = int(POLICY_CALIBRATION_BLOCKS if blocks is None else blocks)
    idx_groups = np.array_split(np.arange(len(data)), blocks)
    metrics = []
    active = []

    for idx in idx_groups:
        if len(idx) == 0:
            continue
        sub_pred = type(pred)(
            pred.buy_tp_probability[idx], pred.sell_tp_probability[idx],
            pred.buy_expected_r[idx], pred.sell_expected_r[idx],
        )
        m = evaluate_opportunities(
            data.iloc[idx], sub_pred,
            min_tp_probability=policy["min_tp_probability"],
            min_expected_r=policy["min_expected_r"],
            min_ev_gap=policy["min_ev_gap"],
            min_probability_gap=policy["min_probability_gap"],
            min_setup_score=policy["min_setup_score"],
            min_setup_gap=policy["min_setup_gap"],
            allow_buy=policy.get("allow_buy", True),
            allow_sell=policy.get("allow_sell", True),
            decluster=True,
        )
        metrics.append(m)
        if m["trades"] >= int(POLICY_CALIBRATION_MIN_TRADES_PER_ACTIVE_BLOCK):
            active.append(m)

    positive = [
        m for m in active
        if m["avg_r"] > 0.0 and m["profit_factor_r"] >= 1.0
    ]
    active_count = len(active)
    positive_count = len(positive)
    positive_fraction = (
        float(positive_count / active_count) if active_count else 0.0
    )
    stable = (
        active_count >= int(POLICY_CALIBRATION_MIN_ACTIVE_BLOCKS)
        and positive_fraction >= float(POLICY_CALIBRATION_MIN_POSITIVE_BLOCK_FRACTION)
    )
    return {
        "stable": bool(stable),
        "active_blocks": int(active_count),
        "positive_blocks": int(positive_count),
        "positive_fraction": float(positive_fraction),
        "metrics": metrics,
    }


def _disabled_policy(reason: str, diagnostics: dict | None = None) -> dict:
    return {
        "enabled": False,
        "min_tp_probability": 0.99,
        "min_expected_r": 5.0,
        "min_ev_gap": 5.0,
        "min_probability_gap": 0.99,
        "min_setup_score": 0.99,
        "min_setup_gap": 0.99,
        "allow_buy": False,
        "allow_sell": False,
        "calibration": {
            "reason": reason,
            "diagnostics": diagnostics or {},
        },
    }


def _calibrate_symbol_policy(symbol, cal, pred, settings, execution, normal, stress):
    # First screen on execution-aware R across a long policy-selection window.
    # Only the best sparse/stable candidates are sent through the slower broker
    # money simulator.
    screened = []
    closest_label = None

    for mode in OPPORTUNITY_DIRECTION_MODES:
        allow_buy, allow_sell = _direction_flags(mode)
        # Cross-direction gap thresholds matter only when BOTH directions have
        # qualified.  A one-sided policy is intentionally independent from an
        # opposite model that did not prove executable edge.
        ev_gaps = OPPORTUNITY_EV_GAP_GRID if mode == "BOTH" else [0.0]
        probability_gaps = OPPORTUNITY_PROB_GAP_GRID if mode == "BOTH" else [0.0]
        for setup_score in OPPORTUNITY_SETUP_SCORE_GRID:
            for setup_gap in OPPORTUNITY_SETUP_GAP_GRID:
                for p in OPPORTUNITY_PROBABILITY_GRID:
                    for ev in OPPORTUNITY_EXPECTED_R_GRID:
                        for gap in ev_gaps:
                            for pgap in probability_gaps:
                                policy = {
                                    "enabled": True,
                                    "allow_buy": bool(allow_buy),
                                    "allow_sell": bool(allow_sell),
                                    "min_setup_score": float(setup_score),
                                    "min_setup_gap": float(setup_gap),
                                    "min_tp_probability": float(p),
                                    "min_expected_r": float(ev),
                                    "min_ev_gap": float(gap),
                                    "min_probability_gap": float(pgap),
                                }
                                metrics = evaluate_opportunities(
                                    cal, pred,
                                    min_tp_probability=p,
                                    min_expected_r=ev,
                                    min_ev_gap=gap,
                                    min_probability_gap=pgap,
                                    min_setup_score=setup_score,
                                    min_setup_gap=setup_gap,
                                    allow_buy=allow_buy,
                                    allow_sell=allow_sell,
                                    decluster=True,
                                )

                                if metrics["trades"] > 0:
                                    near_score = (
                                        metrics["avg_r"] * math.log1p(metrics["trades"])
                                        + 0.20 * min(metrics["profit_factor_r"], 3.0)
                                        - 0.10 * max(0.0, metrics["coverage"] - CALIBRATION_MAX_COVERAGE) * 100.0
                                    )
                                    if closest_label is None or near_score > closest_label[0]:
                                        closest_label = (near_score, policy, metrics)

                                if metrics["trades"] < POLICY_CALIBRATION_MIN_TRADES:
                                    continue
                                if metrics["coverage"] > CALIBRATION_MAX_COVERAGE:
                                    continue
                                if metrics["avg_r"] <= 0 or metrics["profit_factor_r"] < 1.05:
                                    continue

                                stability = _block_stability(cal, pred, policy)
                                if not stability["stable"]:
                                    continue

                                label_score = (
                                    metrics["avg_r"] * math.log1p(metrics["trades"])
                                    + 0.15 * min(metrics["profit_factor_r"], 3.0)
                                    + 0.08 * min(metrics["buy_auc"] + metrics["sell_auc"], 1.5)
                                    + 0.20 * stability["positive_fraction"]
                                )
                                screened.append((label_score, policy, metrics, stability))

    if not screened:
        diagnostics = {}
        if closest_label is not None:
            _, p0, m0 = closest_label
            diagnostics = {
                "best_near_miss_policy": p0,
                "best_near_miss_metrics": m0,
            }
        return _disabled_policy(
            "no_structural_meta_policy_passed_label_stability",
            diagnostics,
        )

    screened.sort(key=lambda x: x[0], reverse=True)
    proxy = SettingsProxy(settings, POLICY_CALIBRATION_RISK_PERCENT)
    best = None
    closest_money = None
    enriched_cache = {}

    for _, policy, metrics, stability in screened[:50]:
        key = tuple(
            policy[k] for k in (
                "allow_buy", "allow_sell",
                "min_setup_score", "min_setup_gap",
                "min_tp_probability", "min_expected_r",
                "min_ev_gap", "min_probability_gap",
            )
        )
        enriched = enriched_cache.get(key)
        if enriched is None:
            enriched = _attach_management_state(cal, pred, policy)
            enriched_cache[key] = enriched

        candidates = _build_candidates(symbol, enriched, pred, policy)
        if len(candidates) < POLICY_CALIBRATION_MIN_TRADES:
            continue

        by_symbol = {symbol: enriched}
        normal_result = s4._portfolio_backtest(
            candidates, by_symbol, POLICY_CALIBRATION_BALANCE,
            normal, proxy, execution=execution,
        )
        stress_result = s4._portfolio_backtest(
            candidates, by_symbol, POLICY_CALIBRATION_BALANCE,
            stress, proxy, execution=execution,
        )

        checks = {
            "normal_profit": normal_result["net_profit"] > 0,
            "stress_profit": stress_result["net_profit"] >= 0,
            "normal_pf": normal_result["profit_factor"] >= POLICY_CALIBRATION_NORMAL_MIN_PROFIT_FACTOR,
            "stress_pf": stress_result["profit_factor"] >= POLICY_CALIBRATION_STRESS_MIN_PROFIT_FACTOR,
            "normal_payoff": normal_result.get("payoff_ratio", 0.0) >= POLICY_CALIBRATION_MIN_PAYOFF_RATIO,
            "normal_trades": normal_result["trades"] >= POLICY_CALIBRATION_MIN_EXECUTED_TRADES,
            "stress_trades": stress_result["trades"] >= POLICY_CALIBRATION_MIN_EXECUTED_TRADES,
            "normal_dd": normal_result["max_drawdown_pct"] <= POLICY_CALIBRATION_NORMAL_MAX_DRAWDOWN_PERCENT,
            "stress_dd": stress_result["max_drawdown_pct"] <= POLICY_CALIBRATION_STRESS_MAX_DRAWDOWN_PERCENT,
        }

        failed_count = sum(1 for value in checks.values() if not value)
        near_score = (
            -failed_count
            + min(normal_result["profit_factor"], 3.0)
            + min(stress_result["profit_factor"], 3.0)
            + 0.20 * normal_result.get("payoff_ratio", 0.0)
            + 0.05 * min(normal_result["return_pct"], stress_result["return_pct"])
        )
        if closest_money is None or near_score > closest_money[0]:
            closest_money = (
                near_score,
                policy,
                metrics,
                stability,
                s4._public_result(normal_result),
                s4._public_result(stress_result),
                checks,
            )

        if not all(checks.values()):
            continue

        worst_return = min(normal_result["return_pct"], stress_result["return_pct"])
        worst_pf = min(normal_result["profit_factor"], stress_result["profit_factor"])
        worst_dd = max(normal_result["max_drawdown_pct"], stress_result["max_drawdown_pct"])
        score = (
            worst_return
            + 2.0 * (worst_pf - 1.0)
            - 0.20 * worst_dd
            + 0.25 * normal_result.get("payoff_ratio", 0.0)
            + 0.25 * stability["positive_fraction"]
        )
        candidate = {
            **policy,
            "score": float(score),
            "calibration": {
                **metrics,
                "stability": stability,
                "normal_money": s4._public_result(normal_result),
                "stress_money": s4._public_result(stress_result),
                "checks": checks,
            },
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    if best is None:
        diagnostics = {}
        if closest_money is not None:
            _, p0, m0, s0, n0, st0, checks0 = closest_money
            diagnostics = {
                "best_near_miss_policy": p0,
                "best_near_miss_metrics": m0,
                "best_near_miss_stability": s0,
                "best_near_miss_normal": n0,
                "best_near_miss_stress": st0,
                "failed_checks": [k for k, v in checks0.items() if not v],
            }
        return _disabled_policy(
            "no_structural_meta_policy_passed_execution_money_gates",
            diagnostics,
        )

    return best


def _risk_checks(result, stress=False):
    if stress:
        return {
            "net_profit_nonnegative": result["net_profit"] >= 0,
            "profit_factor": result["profit_factor"] >= STAGE5_CAL_STRESS_MIN_PROFIT_FACTOR,
            "max_drawdown": result["max_drawdown_pct"] <= STAGE5_CAL_STRESS_MAX_DRAWDOWN_PERCENT,
            "trade_count": result["trades"] >= STAGE5_CAL_MIN_TRADES,
        }
    return {
        "net_profit_positive": result["net_profit"] > 0,
        "profit_factor": result["profit_factor"] >= STAGE5_CAL_NORMAL_MIN_PROFIT_FACTOR,
        "max_drawdown": result["max_drawdown_pct"] <= STAGE5_CAL_NORMAL_MAX_DRAWDOWN_PERCENT,
        "trade_count": result["trades"] >= STAGE5_CAL_MIN_TRADES,
    }


def _calibrate_risk(cal_candidates, cal_by_symbol, normal, stress, settings, execution):
    rows = []
    print("\n" + "═" * 80)
    print("🛡 QUALITY-FIRST RISK CALIBRATION")
    print("═" * 80)
    for risk in STAGE5_RISK_GRID:
        proxy = SettingsProxy(settings, risk)
        per_balance = {}
        all_safe = True
        returns, dds = [], []
        for balance in PRIMARY_ACCEPTANCE_BALANCES:
            nr = s4._portfolio_backtest(cal_candidates, cal_by_symbol, balance, normal, proxy, execution=execution)
            sr = s4._portfolio_backtest(cal_candidates, cal_by_symbol, balance, stress, proxy, execution=execution)
            nc, sc = _risk_checks(nr, False), _risk_checks(sr, True)
            npass, spass = all(nc.values()), all(sc.values())
            all_safe = all_safe and npass and spass
            per_balance[str(float(balance))] = {
                "normal": s4._public_result(nr), "stress": s4._public_result(sr),
                "normal_checks": nc, "stress_checks": sc,
                "normal_pass": npass, "stress_pass": spass,
            }
            returns.extend([nr["return_pct"], sr["return_pct"]])
            dds.extend([nr["max_drawdown_pct"], sr["max_drawdown_pct"]])
        worst_return = min(returns) if returns else -999.0
        worst_dd = max(dds) if dds else 999.0
        mean_return = float(np.mean(returns)) if returns else -999.0
        score = worst_return + 0.10 * mean_return - 0.25 * worst_dd
        print(f"risk={risk:>4.2f}% | {'PASS' if all_safe else 'FAIL'} | worst_return={worst_return:+6.2f}% | worst_DD={worst_dd:5.2f}%")
        rows.append({
            "risk_percent": float(risk), "all_safe": bool(all_safe), "score": float(score),
            "worst_return_pct": float(worst_return), "mean_return_pct": mean_return,
            "worst_drawdown_pct": float(worst_dd), "balances": per_balance,
        })
    safe = [r for r in rows if r["all_safe"]]
    safe.sort(key=lambda r: (r["score"], -r["worst_drawdown_pct"]), reverse=True)
    return (safe[0] if safe else None), rows


def _final_gate(result, stress=False):
    if stress:
        checks = {
            "net_profit_nonnegative": result["net_profit"] >= 0,
            "profit_factor": result["profit_factor"] >= ACCEPT_STRESS_MIN_PROFIT_FACTOR,
            "max_drawdown": result["max_drawdown_pct"] <= ACCEPT_STRESS_MAX_DRAWDOWN_PERCENT,
            "trade_count": result["trades"] >= ACCEPT_STRESS_MIN_TRADES,
        }
    else:
        checks = {
            "net_profit_positive": result["net_profit"] > 0,
            "profit_factor": result["profit_factor"] >= ACCEPT_NORMAL_MIN_PROFIT_FACTOR,
            "max_drawdown": result["max_drawdown_pct"] <= ACCEPT_NORMAL_MAX_DRAWDOWN_PERCENT,
            "trade_count": result["trades"] >= ACCEPT_NORMAL_MIN_TRADES,
        }
    return checks, all(checks.values())


def _write_failure(reason: str, **extra):
    report = {
        "stage": "stage11_structural_meta_validation_v1",
        "created_utc": _utc_now(),
        "acceptance_pass": False,
        "reason": reason,
        "target_version": TARGET_VERSION,
        **extra,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return report


def run():
    settings = s4._load_prod_settings()
    execution = ValidationExecutionContext(settings, require_profile=bool(getattr(settings, "USE_BROKER_PROFILE", False)))
    execution.assert_symbols_profiled(SYMBOLS)

    print("\n" + "═" * 80)
    print("🎯 TRADEAI STAGE 11 — STRUCTURAL META-OPPORTUNITY VALIDATION")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🎯 Target         : {TARGET_VERSION}")
    print("🧠 Model          : structural setup gate + per-direction quality probability + expected-net-R meta-model")
    print("🔒 Selection      : older policy window only; exact deployment final slice cannot choose thresholds/risk")

    artifact = joblib.load(MODEL_PATH)
    validate_model_artifact(artifact, expected_symbols=SYMBOLS, expected_timeframe=TIMEFRAME_NAME, expected_target_version=TARGET_VERSION)
    artifact_window = require_training_window(artifact)
    df, features, loaded_window = _load_data(artifact_window.get("cutoff_exclusive"))
    if int(loaded_window["rows"]) != int(artifact_window["rows"]):
        raise RuntimeError("Validation training-window rows differ from trained artifact")

    normal = s4.CostScenario("NORMAL", float(settings.DEFAULT_SPREAD_PIPS), float(settings.SIMULATED_SLIPPAGE_PIPS), float(settings.COMMISSION_PER_LOT))
    stress = s4.CostScenario("STRESS", float(settings.STRESS_SPREAD_PIPS), float(settings.STRESS_SLIPPAGE_PIPS), float(settings.STRESS_COMMISSION_PER_LOT))

    policies, qualification_models, deployment_models = {}, {}, {}
    symbol_validation, deployment_ranges = {}, {}
    cal_by_symbol, test_by_symbol, cal_candidates, test_candidates = {}, {}, [], []

    for symbol in SYMBOLS:
        print("\n" + "─" * 80)
        print(f"📈 QUALITY OPPORTUNITY MODEL + POLICY: {symbol}")
        data = df[df["symbol"] == symbol].sort_values("time").reset_index(drop=True)
        n = len(data)

        q_end_raw = int(n * QUALIFICATION_TRAIN_FRACTION)
        score_cal_end_raw = int(n * QUALIFICATION_SCORE_CAL_END_FRACTION)
        policy_end_raw = int(n * POLICY_SELECTION_END_FRACTION)
        deploy_cal_end_raw = int(n * DEPLOYMENT_CALIBRATION_END_FRACTION)

        q_train_end = q_end_raw - MAX_TARGET_HORIZON_BARS
        score_cal_end = score_cal_end_raw - MAX_TARGET_HORIZON_BARS
        policy_end = policy_end_raw - MAX_TARGET_HORIZON_BARS
        deploy_train_end = policy_end_raw - MAX_TARGET_HORIZON_BARS
        deploy_cal_end = deploy_cal_end_raw - MAX_TARGET_HORIZON_BARS

        q_train = data.iloc[:q_train_end].copy()
        policy_score_cal = data.iloc[q_end_raw:score_cal_end].copy().reset_index(drop=True)
        policy_select = data.iloc[score_cal_end_raw:policy_end].copy().reset_index(drop=True)
        deploy_train = data.iloc[:deploy_train_end].copy()
        deploy_cal = data.iloc[policy_end_raw:deploy_cal_end].copy().reset_index(drop=True)
        final = data.iloc[deploy_cal_end_raw:].copy().reset_index(drop=True)

        if min(len(q_train), len(policy_score_cal), len(policy_select), len(deploy_train), len(deploy_cal), len(final)) < 3000:
            raise RuntimeError(f"Insufficient Stage 11 chronological zones for {symbol}")

        # 1) Older qualification model chooses thresholds only.
        qualification = create_opportunity_bundle()
        fit_opportunity_bundle(qualification, q_train[features], q_train)
        fit_prediction_calibration(qualification, policy_score_cal[features], policy_score_cal)
        qualification_models[symbol] = qualification

        select_pred = predict_opportunities(qualification, policy_select[features])
        policy = _calibrate_symbol_policy(
            symbol, policy_select, select_pred, settings, execution, normal, stress
        )
        policies[symbol] = policy

        # 2) Exact deployment model uses more recent training, but it does NOT
        # choose thresholds. Its probability/EV scales are calibrated on a
        # dedicated slice and then judged once on the untouched final slice.
        deployment = create_opportunity_bundle()
        fit_opportunity_bundle(deployment, deploy_train[features], deploy_train)
        fit_prediction_calibration(deployment, deploy_cal[features], deploy_cal)
        deployment_models[symbol] = deployment

        deployment_ranges[symbol] = {
            "qualification_train_start": str(q_train["time"].min()),
            "qualification_train_end": str(q_train["time"].max()),
            "policy_score_cal_start": str(policy_score_cal["time"].min()),
            "policy_score_cal_end": str(policy_score_cal["time"].max()),
            "policy_select_start": str(policy_select["time"].min()),
            "policy_select_end": str(policy_select["time"].max()),
            "deployment_train_start": str(deploy_train["time"].min()),
            "deployment_train_end": str(deploy_train["time"].max()),
            "deployment_score_cal_start": str(deploy_cal["time"].min()),
            "deployment_score_cal_end": str(deploy_cal["time"].max()),
            "final_start": str(final["time"].min()),
            "final_end": str(final["time"].max()),
        }

        if not policy.get("enabled"):
            calibration = policy.get("calibration", {})
            reason = calibration.get("reason", "no_valid_edge")
            print(f"Policy DISABLED — {reason}")
            diagnostics = calibration.get("diagnostics") or {}
            near = diagnostics.get("best_near_miss_metrics") or {}
            if near:
                print(
                    "  nearest label candidate | "
                    f"trades={int(near.get('trades', 0))} "
                    f"PF_R={float(near.get('profit_factor_r', 0.0)):.3f} "
                    f"AVG_R={float(near.get('avg_r', 0.0)):+.3f} "
                    f"coverage={float(near.get('coverage', 0.0))*100:.2f}%"
                )
            near_normal = diagnostics.get("best_near_miss_normal") or {}
            near_stress = diagnostics.get("best_near_miss_stress") or {}
            if near_normal or near_stress:
                print(
                    "  nearest money candidate | "
                    f"N_PF={float(near_normal.get('profit_factor', 0.0)):.3f} "
                    f"S_PF={float(near_stress.get('profit_factor', 0.0)):.3f} "
                    f"N_NET={float(near_normal.get('net_profit', 0.0)):+.2f} "
                    f"S_NET={float(near_stress.get('net_profit', 0.0)):+.2f} "
                    f"N_TRADES={int(near_normal.get('trades', 0))} "
                    f"S_TRADES={int(near_stress.get('trades', 0))} "
                    f"failed={','.join(diagnostics.get('failed_checks') or [])}"
                )
                nrej = near_normal.get("rejected") or {}
                srej = near_stress.get("rejected") or {}
                if nrej or srej:
                    print(f"  execution rejects      | NORMAL={nrej} STRESS={srej}")
            symbol_validation[symbol] = {"final": None, "policy_diagnostics": diagnostics}
            cal_by_symbol[symbol] = policy_select
            test_by_symbol[symbol] = final
            continue

        c = policy["calibration"]
        nm, sm = c["normal_money"], c["stress_money"]
        direction_text = (
            "BOTH" if policy.get("allow_buy", True) and policy.get("allow_sell", True)
            else ("BUY" if policy.get("allow_buy", True) else "SELL")
        )
        print(
            f"Policy {direction_text} SETUP>={policy['min_setup_score']:.2f}/gap{policy['min_setup_gap']:.2f} "
            f"QP>={policy['min_tp_probability']:.2f} EV>={policy['min_expected_r']:.2f}R "
            f"EVgap>={policy['min_ev_gap']:.2f} | SELECT PF={nm['profit_factor']:.2f}/{sm['profit_factor']:.2f} "
            f"PAY={nm.get('payoff_ratio', 0.0):.2f} trades={nm['trades']}"
        )

        select_enriched = _attach_management_state(policy_select, select_pred, policy)
        cal_by_symbol[symbol] = select_enriched
        cal_candidates.extend(_build_candidates(symbol, select_enriched, select_pred, policy))

        final_pred = predict_opportunities(deployment, final[features])
        final_metrics = evaluate_opportunities(
            final, final_pred,
            min_tp_probability=policy["min_tp_probability"],
            min_expected_r=policy["min_expected_r"],
            min_ev_gap=policy["min_ev_gap"],
            min_probability_gap=policy["min_probability_gap"],
            min_setup_score=policy["min_setup_score"],
            min_setup_gap=policy["min_setup_gap"],
            allow_buy=policy.get("allow_buy", True),
            allow_sell=policy.get("allow_sell", True),
            decluster=True,
        )
        print(
            f"Final exact-model diagnostic: AUC={final_metrics['buy_auc']:.3f}/{final_metrics['sell_auc']:.3f} "
            f"trades={final_metrics['trades']} PF_R={final_metrics['profit_factor_r']:.3f} "
            f"AVG_R={final_metrics['avg_r']:+.3f}"
        )
        symbol_validation[symbol] = {"final": final_metrics}
        final_enriched = _attach_management_state(final, final_pred, policy)
        test_by_symbol[symbol] = final_enriched
        test_candidates.extend(_build_candidates(symbol, final_enriched, final_pred, policy))

    enabled_symbols = [s for s, p in policies.items() if p.get("enabled")]
    if not enabled_symbols:
        print("\n❌ NO SYMBOL PROVED POSITIVE EXECUTABLE EDGE")
        print("Fail-safe HOLD remains active. No risk policy will be promoted.")
        _write_failure(
            "no_symbol_passed_opportunity_money_gates",
            decision_policy={"version": DECISION_POLICY_VERSION, "symbols": policies},
            symbol_validation=symbol_validation,
        )
        return

    selected, risk_grid = _calibrate_risk(cal_candidates, cal_by_symbol, normal, stress, settings, execution)
    if selected is None:
        print("\n❌ RISK CALIBRATION FAILED — no safe risk level")
        _write_failure(
            "no_calibrated_risk_passed_safety_margin",
            enabled_symbols=enabled_symbols,
            decision_policy={"version": DECISION_POLICY_VERSION, "symbols": policies},
            symbol_validation=symbol_validation,
            risk_grid=risk_grid,
        )
        return

    risk_percent = float(selected["risk_percent"])
    print(f"\n✅ CALIBRATED RISK: {risk_percent:.2f}% per trade")
    proxy = SettingsProxy(settings, risk_percent)
    money_results = {"NORMAL": {}, "STRESS": {}}
    primary_trades = None

    print("\n" + "═" * 80)
    print("💵 FINAL MONEY TEST — FROZEN POLICY + RISK")
    print("═" * 80)
    for balance in VALIDATION_BALANCES:
        nr = s4._portfolio_backtest(test_candidates, test_by_symbol, balance, normal, proxy, execution=execution)
        sr = s4._portfolio_backtest(test_candidates, test_by_symbol, balance, stress, proxy, execution=execution)
        money_results["NORMAL"][str(balance)] = s4._public_result(nr)
        money_results["STRESS"][str(balance)] = s4._public_result(sr)
        print(f"NORMAL ", end=""); s4._print_money(nr)
        print(f"STRESS ", end=""); s4._print_money(sr)
        if float(balance) == float(PRIMARY_ACCEPTANCE_BALANCES[0]):
            primary_trades = nr["trades_detail"]

    gates, all_pass = {}, True
    for balance in PRIMARY_ACCEPTANCE_BALANCES:
        key = str(float(balance))
        nc, npass = _final_gate(money_results["NORMAL"][key], False)
        sc, spass = _final_gate(money_results["STRESS"][key], True)
        gates[key] = {"normal": nc, "normal_pass": npass, "stress": sc, "stress_pass": spass}
        all_pass = all_pass and npass and spass

    deployment_validation = {
        "exact_runtime_model": True,
        "refit_after_policy_freeze": True,
        "qualification_model_distinct_from_runtime_refit": True,
        "score_calibration_separate_from_policy_selection": True,
        "architecture": "structural_meta_opportunity_v3",
        "split_method": (
            f"qtrain={QUALIFICATION_TRAIN_FRACTION:.2f};"
            f"qscore_cal_end={QUALIFICATION_SCORE_CAL_END_FRACTION:.2f};"
            f"policy_end={POLICY_SELECTION_END_FRACTION:.2f};"
            f"deploy_cal_end={DEPLOYMENT_CALIBRATION_END_FRACTION:.2f};final=rest"
        ),
        "label_purge_bars": int(MAX_TARGET_HORIZON_BARS),
        "symbol_ranges": deployment_ranges,
        "information_safe_from": artifact_window.get("backtest_safe_from"),
    }
    decision_policy = {
        "version": DECISION_POLICY_VERSION,
        "created_utc": _utc_now(),
        "method": "structural_meta_sparse_stable_broker_money_selection",
        "symbols": policies,
    }
    risk_policy = {
        "version": RISK_POLICY_VERSION,
        "created_utc": _utc_now(),
        "risk_percent": risk_percent,
        "source": "stage11_structural_meta_policy_selection_normal_and_stress",
        "calibration_safety": {
            "normal_max_drawdown_percent": STAGE5_CAL_NORMAL_MAX_DRAWDOWN_PERCENT,
            "stress_max_drawdown_percent": STAGE5_CAL_STRESS_MAX_DRAWDOWN_PERCENT,
            "selected_worst_drawdown_percent": selected["worst_drawdown_pct"],
            "selected_worst_return_percent": selected["worst_return_pct"],
            "portfolio_risk_cap_percent": float(getattr(settings, "MAX_PORTFOLIO_RISK_PERCENT", 0.0)),
        },
    }
    report = {
        "stage": "stage11_structural_meta_validation_v1",
        "created_utc": _utc_now(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "training_window": artifact_window,
        "deployment_validation": deployment_validation,
        "enabled_symbols": enabled_symbols,
        "decision_policy": decision_policy,
        "risk_policy": risk_policy,
        "risk_grid": risk_grid,
        "symbol_validation": symbol_validation,
        "money_results": money_results,
        "acceptance_gates": gates,
        "acceptance_pass": bool(all_pass),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f: json.dump(report, f, indent=2, default=str)
    with open(RISK_POLICY_JSON, "w", encoding="utf-8") as f: json.dump(risk_policy, f, indent=2, default=str)
    with open(DECISION_POLICY_JSON, "w", encoding="utf-8") as f: json.dump(decision_policy, f, indent=2, default=str)
    if primary_trades is not None:
        pd.DataFrame(primary_trades).to_csv(TRADES_CSV, index=False)

    print("\n" + "═" * 80)
    print("🚦 OPPORTUNITY PRODUCTION ACCEPTANCE")
    print("═" * 80)
    for balance in PRIMARY_ACCEPTANCE_BALANCES:
        g = gates[str(float(balance))]
        print(f"${balance:.0f} NORMAL={'PASS' if g['normal_pass'] else 'FAIL'} | STRESS={'PASS' if g['stress_pass'] else 'FAIL'}")

    if all_pass:
        # Policy/risk are frozen from the qualification models. Runtime uses the
        # trainer artifact refitted on ALL information-safe pre-June rows. The
        # untouched June backtest therefore evaluates the exact deployment
        # model while keeping June completely outside selection/refit.
        promoted = _with_exact_deployment_models(artifact, deployment_models, deployment_validation)
        promoted = attach_decision_policy(promoted, decision_policy, report)
        promoted = attach_risk_policy(promoted, risk_policy, report)
        temp = MODEL_PATH.with_suffix(".stage11.tmp.pkl")
        joblib.dump(promoted, temp)
        os.replace(temp, MODEL_PATH)
        print("\n✅ STAGE 11 ACCEPTANCE: PASS")
        print("🔐 Exact score-calibrated deployment models + frozen policy + risk promoted")
    else:
        print("\n❌ STAGE 11 ACCEPTANCE: FAIL")
        print("No model promotion. Fail-safe HOLD remains active.")


if __name__ == "__main__":
    run()
