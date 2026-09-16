from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import stage4_production_validate as s4
from config_model import (
    CALIBRATION_END_FRACTION,
    CALIBRATION_TRAIN_FRACTION,
    MODEL_PATH,
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
from training_utils import compute_weights, create_model, evaluate_probabilities
from shared.tradeai_core.decision_policy import DECISION_POLICY_VERSION
from shared.tradeai_core.model_contract import (
    attach_decision_policy,
    attach_risk_policy,
    validate_model_artifact,
)
from shared.tradeai_core.risk_policy import RISK_POLICY_VERSION
from shared.tradeai_core.target_definition import MAX_TARGET_HORIZON_BARS, TARGET_VERSION


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


def _risk_calibration_checks(result, *, stress: bool):
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


def _calibrate_risk(cal_candidates, cal_by_symbol, normal, stress, settings):
    candidates = []

    print("\n" + "═" * 80)
    print("🛡 STAGE 5 RISK CALIBRATION — CALIBRATION SLICE ONLY")
    print("═" * 80)
    print(
        f"Safety margin gates: NORMAL DD<={STAGE5_CAL_NORMAL_MAX_DRAWDOWN_PERCENT:.1f}% | "
        f"STRESS DD<={STAGE5_CAL_STRESS_MAX_DRAWDOWN_PERCENT:.1f}%"
    )

    for risk_percent in STAGE5_RISK_GRID:
        proxy = SettingsProxy(settings, risk_percent)
        per_balance = {}
        all_safe = True
        returns = []
        drawdowns = []

        for balance in PRIMARY_ACCEPTANCE_BALANCES:
            normal_result = s4._portfolio_backtest(
                cal_candidates, cal_by_symbol, balance, normal, proxy
            )
            stress_result = s4._portfolio_backtest(
                cal_candidates, cal_by_symbol, balance, stress, proxy
            )

            normal_checks = _risk_calibration_checks(normal_result, stress=False)
            stress_checks = _risk_calibration_checks(stress_result, stress=True)
            normal_pass = all(normal_checks.values())
            stress_pass = all(stress_checks.values())
            all_safe = all_safe and normal_pass and stress_pass

            per_balance[str(float(balance))] = {
                "normal": s4._public_result(normal_result),
                "stress": s4._public_result(stress_result),
                "normal_checks": normal_checks,
                "stress_checks": stress_checks,
                "normal_pass": normal_pass,
                "stress_pass": stress_pass,
            }
            returns.extend([normal_result["return_pct"], stress_result["return_pct"]])
            drawdowns.extend([normal_result["max_drawdown_pct"], stress_result["max_drawdown_pct"]])

        worst_return = float(min(returns)) if returns else -999.0
        mean_return = float(np.mean(returns)) if returns else -999.0
        worst_dd = float(max(drawdowns)) if drawdowns else 999.0
        score = worst_return + 0.15 * mean_return - 0.25 * worst_dd

        status = "PASS" if all_safe else "FAIL"
        print(
            f"risk={risk_percent:>4.2f}% | {status} | "
            f"worst_return={worst_return:+7.1f}% | worst_DD={worst_dd:5.2f}%"
        )

        candidates.append({
            "risk_percent": float(risk_percent),
            "all_safe": bool(all_safe),
            "score": float(score),
            "worst_return_pct": worst_return,
            "mean_return_pct": mean_return,
            "worst_drawdown_pct": worst_dd,
            "balances": per_balance,
        })

    safe = [c for c in candidates if c["all_safe"]]
    if not safe:
        return None, candidates

    # Maximize the worst-case calibration return subject to the strict safety
    # gates. Lower drawdown is the tie-breaker.
    safe.sort(
        key=lambda c: (c["score"], -c["worst_drawdown_pct"]),
        reverse=True,
    )
    return safe[0], candidates


def run():
    settings = s4._load_prod_settings()

    print("\n" + "═" * 80)
    print("🛡 TRADEAI STAGE 5 — RISK-CALIBRATED PRODUCTION VALIDATION")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🔐 Feature hash   : {FEATURE_HASH}")
    print(f"🎯 Target         : {TARGET_VERSION}")
    print("🔒 Risk selection : calibration slice only; final slice cannot choose risk")

    df, features = s4._load_data()
    artifact = joblib.load(MODEL_PATH)
    validate_model_artifact(
        artifact,
        expected_symbols=SYMBOLS,
        expected_timeframe=TIMEFRAME_NAME,
        expected_target_version=TARGET_VERSION,
    )

    policies = {}
    cal_by_symbol = {}
    test_by_symbol = {}
    cal_candidates = []
    test_candidates = []
    symbol_validation = {}

    for symbol in SYMBOLS:
        print("\n" + "─" * 80)
        print(f"📈 MODEL + POLICY: {symbol}")
        data = df[df["symbol"] == symbol].sort_values("time").reset_index(drop=True)
        n = len(data)
        split_train = int(n * CALIBRATION_TRAIN_FRACTION)
        split_test = int(n * CALIBRATION_END_FRACTION)
        train_end = split_train - MAX_TARGET_HORIZON_BARS
        cal_end = split_test - MAX_TARGET_HORIZON_BARS

        train = data.iloc[:train_end].copy()
        cal = data.iloc[split_train:cal_end].copy().reset_index(drop=True)
        test = data.iloc[split_test:].copy().reset_index(drop=True)

        if min(len(train), len(cal), len(test)) < 5000:
            raise RuntimeError(f"Insufficient Stage 5 split rows for {symbol}")

        model = create_model()
        y_train = train["target_class"].to_numpy(dtype=np.int32)
        model.fit(
            train[features],
            y_train,
            sample_weight=compute_weights(y_train, train["target_best_r"].to_numpy()),
            verbose=False,
        )

        cal_proba = model.predict_proba(cal[features])
        policy = s4._calibrate_policy(symbol, model, cal, cal_proba)
        policies[symbol] = policy

        if policy["enabled"]:
            c = policy["calibration"]
            print(
                f"Policy conf>={policy['min_confidence']:.2f} "
                f"edge>={policy['signal_threshold']:.2f} | "
                f"CAL PF_R={c['profit_factor_r']:.3f} AVG_R={c['avg_r']:+.4f}"
            )
        else:
            print("Policy DISABLED")

        test_proba = model.predict_proba(test[features])
        if policy["enabled"]:
            final_r = evaluate_probabilities(
                model,
                test["target_class"].to_numpy(dtype=np.int32),
                test_proba,
                test["target_buy_r"].to_numpy(dtype=np.float64),
                test["target_sell_r"].to_numpy(dtype=np.float64),
                min_confidence=policy["min_confidence"],
                signal_threshold=policy["signal_threshold"],
            )
            print(
                f"Final R diagnostic: trades={final_r['trades']:,} "
                f"PF_R={final_r['profit_factor_r']:.3f} AVG_R={final_r['avg_r']:+.4f}"
            )
            symbol_validation[symbol] = {"final_r": final_r}
        else:
            symbol_validation[symbol] = {"final_r": None}

        cal_by_symbol[symbol] = cal
        test_by_symbol[symbol] = test
        cal_candidates.extend(s4._build_candidates(symbol, model, cal, cal_proba, policy))
        test_candidates.extend(s4._build_candidates(symbol, model, test, test_proba, policy))

    enabled_symbols = [s for s, p in policies.items() if p.get("enabled")]
    if not enabled_symbols:
        raise RuntimeError("Stage 5 calibration disabled every symbol")

    normal = s4.CostScenario(
        "NORMAL",
        float(settings.DEFAULT_SPREAD_PIPS),
        float(settings.SIMULATED_SLIPPAGE_PIPS),
        float(settings.COMMISSION_PER_LOT),
    )
    stress = s4.CostScenario(
        "STRESS",
        float(settings.STRESS_SPREAD_PIPS),
        float(settings.STRESS_SLIPPAGE_PIPS),
        float(settings.STRESS_COMMISSION_PER_LOT),
    )

    selected, risk_grid_report = _calibrate_risk(
        cal_candidates, cal_by_symbol, normal, stress, settings
    )

    if selected is None:
        print("\n❌ STAGE 5 RISK CALIBRATION FAILED")
        print("No risk level passed the calibration safety-margin gates.")
        report = {
            "stage": "stage5_risk_calibrated_validation_v1",
            "created_utc": _utc_now(),
            "acceptance_pass": False,
            "reason": "no_calibrated_risk_passed_safety_margin",
            "risk_grid": risk_grid_report,
        }
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORT_JSON, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        return

    risk_percent = float(selected["risk_percent"])
    print("\n" + "═" * 80)
    print(f"✅ CALIBRATED PRODUCTION RISK: {risk_percent:.2f}% per trade")
    print(
        f"Calibration worst DD={selected['worst_drawdown_pct']:.2f}% | "
        f"worst return={selected['worst_return_pct']:+.1f}%"
    )

    proxy = SettingsProxy(settings, risk_percent)
    money_results = {"NORMAL": {}, "STRESS": {}}
    primary_trades = None

    print("\n" + "═" * 80)
    print("💵 FINAL MONEY TEST — FROZEN RISK + NORMAL COSTS")
    print("═" * 80)
    for balance in VALIDATION_BALANCES:
        result = s4._portfolio_backtest(test_candidates, test_by_symbol, balance, normal, proxy)
        money_results["NORMAL"][str(balance)] = s4._public_result(result)
        s4._print_money(result)
        if float(balance) == 20.0:
            primary_trades = result["trades_detail"]

    print("\n" + "═" * 80)
    print("🔥 FINAL STRESS TEST — FROZEN RISK")
    print("═" * 80)
    for balance in VALIDATION_BALANCES:
        result = s4._portfolio_backtest(test_candidates, test_by_symbol, balance, stress, proxy)
        money_results["STRESS"][str(balance)] = s4._public_result(result)
        s4._print_money(result)

    gates = {}
    all_pass = True
    for balance in PRIMARY_ACCEPTANCE_BALANCES:
        key = str(float(balance))
        normal_checks, normal_pass = s4._gate_result(money_results["NORMAL"][key], stress=False)
        stress_checks, stress_pass = s4._gate_result(money_results["STRESS"][key], stress=True)
        gates[key] = {
            "normal": normal_checks,
            "normal_pass": normal_pass,
            "stress": stress_checks,
            "stress_pass": stress_pass,
        }
        all_pass = all_pass and normal_pass and stress_pass

    decision_policy = {
        "version": DECISION_POLICY_VERSION,
        "created_utc": _utc_now(),
        "method": "70_15_15_chronological_calibration_with_purged_boundaries",
        "symbols": policies,
    }
    risk_policy = {
        "version": RISK_POLICY_VERSION,
        "created_utc": _utc_now(),
        "risk_percent": risk_percent,
        "source": "stage5_calibration_slice_normal_and_stress",
        "calibration_safety": {
            "normal_max_drawdown_percent": STAGE5_CAL_NORMAL_MAX_DRAWDOWN_PERCENT,
            "stress_max_drawdown_percent": STAGE5_CAL_STRESS_MAX_DRAWDOWN_PERCENT,
            "selected_worst_drawdown_percent": selected["worst_drawdown_pct"],
            "selected_worst_return_percent": selected["worst_return_pct"],
        },
    }

    validation_report = {
        "stage": "stage5_risk_calibrated_validation_v1",
        "created_utc": _utc_now(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "enabled_symbols": enabled_symbols,
        "decision_policy": decision_policy,
        "risk_policy": risk_policy,
        "risk_grid": risk_grid_report,
        "symbol_validation": symbol_validation,
        "money_results": money_results,
        "acceptance_gates": gates,
        "acceptance_pass": bool(all_pass),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, default=str)
    with open(RISK_POLICY_JSON, "w", encoding="utf-8") as f:
        json.dump(risk_policy, f, indent=2, default=str)
    with open(DECISION_POLICY_JSON, "w", encoding="utf-8") as f:
        json.dump(decision_policy, f, indent=2, default=str)
    if primary_trades is not None:
        pd.DataFrame(primary_trades).to_csv(TRADES_CSV, index=False)

    print("\n" + "═" * 80)
    print("🚦 STAGE 5 PRODUCTION ACCEPTANCE")
    print("═" * 80)
    for balance in PRIMARY_ACCEPTANCE_BALANCES:
        key = str(float(balance))
        g = gates[key]
        print(
            f"${balance:.0f} NORMAL={'PASS' if g['normal_pass'] else 'FAIL'} | "
            f"STRESS={'PASS' if g['stress_pass'] else 'FAIL'}"
        )

    if all_pass:
        promoted = attach_decision_policy(artifact, decision_policy, validation_report)
        promoted = attach_risk_policy(promoted, risk_policy, validation_report)
        temp_model = MODEL_PATH.with_suffix(".stage5.tmp.pkl")
        joblib.dump(promoted, temp_model)
        os.replace(temp_model, MODEL_PATH)
        print("\n✅ STAGE 5 ACCEPTANCE: PASS")
        print(f"🔐 Decision + risk policies embedded into: {MODEL_PATH}")
        print("The artifact is eligible for the TradeAI demo-forward gate.")
    else:
        print("\n❌ STAGE 5 ACCEPTANCE: FAIL")
        print("The model artifact was NOT promoted.")
        print("TradeAI remains fail-safe because calibrated decision and risk policies are required.")

    print(f"📄 Report      : {REPORT_JSON}")
    print(f"🛡 Risk policy : {RISK_POLICY_JSON}")
    print(f"🎚 Decision    : {DECISION_POLICY_JSON}")
    print(f"🧾 Trades      : {TRADES_CSV}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    run()
