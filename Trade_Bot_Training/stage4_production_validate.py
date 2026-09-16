# filename: Trade_Bot_Traning/stage4_production_validate.py

from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from config_model import (
    ACCEPT_NORMAL_MAX_DRAWDOWN_PERCENT,
    ACCEPT_NORMAL_MIN_PROFIT_FACTOR,
    ACCEPT_NORMAL_MIN_TRADES,
    ACCEPT_STRESS_MAX_DRAWDOWN_PERCENT,
    ACCEPT_STRESS_MIN_PROFIT_FACTOR,
    ACCEPT_STRESS_MIN_TRADES,
    CALIBRATION_CONFIDENCE_GRID,
    CALIBRATION_EDGE_GRID,
    CALIBRATION_END_FRACTION,
    CALIBRATION_MAX_COVERAGE,
    CALIBRATION_MIN_PROFIT_FACTOR,
    CALIBRATION_MIN_TRADES,
    CALIBRATION_TRAIN_FRACTION,
    DATA_PATH,
    DATASET_METADATA_PATH,
    MODEL_PATH,
    PRIMARY_ACCEPTANCE_BALANCES,
    REPORT_DIR,
    SYMBOLS,
    TIMEFRAME_NAME,
    VALIDATION_BALANCES,
)
from feature_engine import FEATURE_HASH, FEATURE_SCHEMA_VERSION, FeatureTransformer
from training_utils import (
    actions_from_probabilities,
    compute_weights,
    create_model,
    evaluate_probabilities,
    probability_columns,
)
from shared.tradeai_core.decision_policy import DECISION_POLICY_VERSION
from shared.tradeai_core.model_contract import (
    attach_decision_policy,
    validate_model_artifact,
)
from shared.tradeai_core.production_economics import (
    entry_price_with_costs,
    fallback_pip_value_per_lot,
    pip_size,
    risk_sized_lot,
)
from shared.tradeai_core.target_definition import (
    BUY_CLASS,
    HOLD_CLASS,
    MAX_TARGET_HORIZON_BARS,
    SELL_CLASS,
    TARGET_VERSION,
)


SYSTEM_ROOT = Path(__file__).resolve().parent.parent
PROD_SETTINGS_PATH = SYSTEM_ROOT / "TradeAI" / "config" / "settings.py"
REPORT_JSON = REPORT_DIR / "stage4_production_validation.json"
POLICY_JSON = REPORT_DIR / "stage4_decision_policy.json"
TRADES_CSV = REPORT_DIR / "stage4_primary_trades.csv"


@dataclass(frozen=True)
class CostScenario:
    name: str
    spread_pips: float
    slippage_pips: float
    commission_per_lot: float


def _load_prod_settings():
    spec = importlib.util.spec_from_file_location("tradeai_stage4_prod_settings", PROD_SETTINGS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load production settings: {PROD_SETTINGS_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_dataset_metadata():
    with open(DATASET_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    checks = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "timeframe": TIMEFRAME_NAME,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise RuntimeError(
                f"Dataset contract mismatch {key}: expected {expected!r}, got {metadata.get(key)!r}"
            )
    if metadata.get("closed_candles_only") is not True or metadata.get("causal_h1_alignment") is not True:
        raise RuntimeError("Dataset does not carry Stage 2 causality certification")
    return metadata


def _load_data():
    _validate_dataset_metadata()
    features = FeatureTransformer().get_feature_list()
    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    required = features + [
        "symbol", "time", "open", "high", "low", "close", "atr",
        "target_class", "target_buy_r", "target_sell_r", "target_best_r",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Stage 4 dataset missing columns: {missing}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=required, inplace=True)
    df["target_class"] = df["target_class"].astype(np.int32)
    return df, features


def _block_stability(model, proba, data, min_confidence, signal_threshold, blocks=3):
    indices = np.array_split(np.arange(len(data)), blocks)
    block_metrics = []
    for idx in indices:
        if len(idx) == 0:
            continue
        m = evaluate_probabilities(
            model,
            data["target_class"].to_numpy(dtype=np.int32)[idx],
            proba[idx],
            data["target_buy_r"].to_numpy(dtype=np.float64)[idx],
            data["target_sell_r"].to_numpy(dtype=np.float64)[idx],
            min_confidence=min_confidence,
            signal_threshold=signal_threshold,
        )
        block_metrics.append(m)
    positive = sum(1 for m in block_metrics if m["avg_r"] > 0 and m["profit_factor_r"] >= 1.0)
    avg_r_values = [m["avg_r"] for m in block_metrics]
    std = float(np.std(avg_r_values)) if avg_r_values else 999.0
    return positive, std, block_metrics


def _calibrate_policy(symbol, model, cal, proba):
    best = None
    for min_conf in CALIBRATION_CONFIDENCE_GRID:
        for edge in CALIBRATION_EDGE_GRID:
            m = evaluate_probabilities(
                model,
                cal["target_class"].to_numpy(dtype=np.int32),
                proba,
                cal["target_buy_r"].to_numpy(dtype=np.float64),
                cal["target_sell_r"].to_numpy(dtype=np.float64),
                min_confidence=min_conf,
                signal_threshold=edge,
            )
            if m["trades"] < CALIBRATION_MIN_TRADES:
                continue
            if m["profit_factor_r"] < CALIBRATION_MIN_PROFIT_FACTOR:
                continue
            if m["coverage"] > CALIBRATION_MAX_COVERAGE:
                continue
            if m["avg_r"] <= 0:
                continue

            positive_blocks, block_std, block_metrics = _block_stability(
                model, proba, cal, min_conf, edge
            )
            if positive_blocks < 2:
                continue

            score = (
                m["avg_r"] * math.log1p(m["trades"])
                + 0.05 * min(m["profit_factor_r"], 3.0)
                + 0.04 * positive_blocks
                - 0.20 * block_std
            )
            candidate = {
                "enabled": True,
                "min_confidence": float(min_conf),
                "signal_threshold": float(edge),
                "score": float(score),
                "calibration": {
                    "trades": int(m["trades"]),
                    "coverage": float(m["coverage"]),
                    "win_rate": float(m["win_rate"]),
                    "avg_r": float(m["avg_r"]),
                    "profit_factor_r": float(m["profit_factor_r"]),
                    "positive_blocks": int(positive_blocks),
                    "block_avg_r_std": float(block_std),
                },
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    if best is None:
        return {
            "enabled": False,
            "min_confidence": 0.99,
            "signal_threshold": 0.99,
            "score": -999.0,
            "calibration": {
                "reason": "no_threshold_pair_passed_robustness_gates"
            },
        }
    return best


def _build_candidates(symbol, model, test, proba, policy):
    if not policy.get("enabled", True):
        return []

    actions, edge, conf, p_hold = actions_from_probabilities(
        model,
        proba,
        min_confidence=policy["min_confidence"],
        signal_threshold=policy["signal_threshold"],
    )
    p_sell, _, p_buy = probability_columns(model, proba)

    candidates = []
    max_idx = len(test) - MAX_TARGET_HORIZON_BARS
    for idx in np.flatnonzero(actions != HOLD_CLASS):
        if idx >= max_idx:
            continue
        action = int(actions[idx])
        candidates.append({
            "time": pd.Timestamp(test.iloc[idx]["time"]),
            "symbol": symbol,
            "index": int(idx),
            "direction": "BUY" if action == BUY_CLASS else "SELL",
            "edge": float(edge[idx]),
            "confidence": float(conf[idx]),
            "p_hold": float(p_hold[idx]),
            "p_buy": float(p_buy[idx]),
            "p_sell": float(p_sell[idx]),
        })
    return candidates


def _simulate_trade_path(data, idx, candidate, balance, scenario, settings):
    row = data.iloc[idx]
    symbol = candidate["symbol"]
    direction = candidate["direction"]
    requested = float(row["close"])
    atr0 = float(row["atr"])
    sl_distance = atr0 * float(settings.ATR_SL_MULTIPLIER)
    tp_distance = atr0 * float(settings.ATR_TP_MULTIPLIER)

    if sl_distance <= 0 or tp_distance <= 0:
        return None, "invalid_atr"

    if direction == "BUY":
        stop_loss = requested - sl_distance
        take_profit = requested + tp_distance
    else:
        stop_loss = requested + sl_distance
        take_profit = requested - tp_distance

    lot, risk_info = risk_sized_lot(
        balance=balance,
        risk_percent=float(settings.RISK_PERCENT),
        stop_distance=sl_distance,
        symbol=symbol,
        price=requested,
        minimum=float(settings.MIN_LOT),
        maximum=float(settings.MAX_LOT),
        step=float(settings.LOT_STEP),
        max_actual_risk_percent=float(settings.MAX_ACTUAL_RISK_PERCENT),
    )
    if lot is None:
        return None, risk_info.get("reason", "lot_rejected")

    entry = entry_price_with_costs(
        requested_price=requested,
        direction=direction,
        symbol=symbol,
        spread_pips=scenario.spread_pips,
        slippage_pips=scenario.slippage_pips,
    )

    if direction == "BUY" and not (stop_loss < entry < take_profit):
        return None, "entry_cost_invalidates_stops"
    if direction == "SELL" and not (take_profit < entry < stop_loss):
        return None, "entry_cost_invalidates_stops"

    pip = pip_size(symbol)
    pip_value = fallback_pip_value_per_lot(symbol, entry)
    commission = float(scenario.commission_per_lot) * float(lot)
    be_done = False
    current_sl = float(stop_loss)
    exit_price = None
    exit_reason = None
    exit_time = None
    worst_equity = balance

    for step in range(1, MAX_TARGET_HORIZON_BARS + 1):
        future = data.iloc[idx + step]
        high = float(future["high"])
        low = float(future["low"])
        close = float(future["close"])
        exit_time = pd.Timestamp(future["time"])

        # Conservative intrabar equity observation.
        adverse_price = low if direction == "BUY" else high
        adverse_diff = (adverse_price - entry) if direction == "BUY" else (entry - adverse_price)
        adverse_pips = adverse_diff / pip
        adverse_net = adverse_pips * pip_value * lot - commission
        worst_equity = min(worst_equity, balance + adverse_net)

        if direction == "BUY":
            sl_hit = low <= current_sl
            tp_hit = high >= take_profit
        else:
            sl_hit = high >= current_sl
            tp_hit = low <= take_profit

        # Same-candle ambiguity: stop wins, matching PaperExecutor.
        if sl_hit:
            exit_price = current_sl
            exit_reason = "STOP_LOSS" if not be_done or current_sl != entry else "BREAK_EVEN"
            break
        if tp_hit:
            exit_price = take_profit
            exit_reason = "TAKE_PROFIT"
            break

        # Stage 4 production lifecycle aligns with the 24-bar training target.
        if step >= MAX_TARGET_HORIZON_BARS:
            exit_price = close
            exit_reason = "MAX_HOLD"
            break

        profit_pips = ((close - entry) if direction == "BUY" else (entry - close)) / pip

        if bool(settings.USE_BREAK_EVEN) and profit_pips >= float(settings.BREAK_EVEN_TRIGGER_PIPS) and not be_done:
            current_sl = entry
            be_done = True

        if bool(settings.USE_TRAILING_STOP) and profit_pips >= float(settings.TRAILING_TRIGGER_PIPS):
            # CoreEngine supplies the ATR from the historical frame before the
            # current execution candle, so use the previous row's ATR.
            atr_prev = float(data.iloc[idx + step - 1]["atr"])
            if atr_prev > 0:
                if direction == "BUY":
                    new_sl = close - atr_prev
                    if new_sl > current_sl:
                        current_sl = new_sl
                else:
                    new_sl = close + atr_prev
                    if new_sl < current_sl:
                        current_sl = new_sl

    if exit_price is None or exit_time is None:
        return None, "no_exit"

    difference = (exit_price - entry) if direction == "BUY" else (entry - exit_price)
    pips = difference / pip
    gross = pips * pip_value * lot
    net = round(gross - commission, 2)
    entry_cost_pips = float(scenario.spread_pips) / 2.0 + float(scenario.slippage_pips)
    entry_cost_money = entry_cost_pips * pip_value * lot

    return {
        "symbol": symbol,
        "direction": direction,
        "open_time": pd.Timestamp(row["time"]),
        "close_time": exit_time,
        "requested_price": requested,
        "entry_price": entry,
        "exit_price": float(exit_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "lot": float(lot),
        "pips": float(pips),
        "gross_profit": float(gross),
        "commission": float(commission),
        "entry_cost_money": float(entry_cost_money),
        "profit": float(net),
        "exit_reason": exit_reason,
        "confidence": candidate["confidence"],
        "edge": candidate["edge"],
        "actual_risk_percent": float(risk_info.get("actual_risk_percent", 0.0)),
        "worst_equity": float(worst_equity),
    }, None


def _portfolio_backtest(candidates, test_by_symbol, start_balance, scenario, settings):
    symbol_priority = {symbol: i for i, symbol in enumerate(SYMBOLS)}
    candidates = sorted(candidates, key=lambda c: (c["time"], symbol_priority[c["symbol"]]))

    balance = float(start_balance)
    peak = balance
    max_dd_pct = 0.0
    max_dd_money = 0.0
    free_after = None
    last_close = {}
    day_key = None
    day_start_balance = balance
    trades = []
    rejected = {
        "position_busy": 0,
        "cooldown": 0,
        "daily_loss": 0,
        "drawdown_protection": 0,
        "lot_or_risk": 0,
        "invalid_trade": 0,
    }

    for candidate in candidates:
        time = pd.Timestamp(candidate["time"])
        symbol = candidate["symbol"]

        # Conservative: require strictly later timestamp after the previous
        # position closes. This avoids same-timestamp symbol-order ambiguity.
        if free_after is not None and time <= free_after:
            rejected["position_busy"] += 1
            continue

        last = last_close.get(symbol)
        if last is not None and (time - last).total_seconds() < float(settings.COOLDOWN_SECONDS):
            rejected["cooldown"] += 1
            continue

        current_day = time.date().isoformat()
        if current_day != day_key:
            day_key = current_day
            day_start_balance = balance

        if day_start_balance > 0:
            daily_loss_pct = max(0.0, (day_start_balance - balance) / day_start_balance * 100.0)
            if daily_loss_pct >= float(settings.MAX_DAILY_LOSS_PERCENT):
                rejected["daily_loss"] += 1
                continue

        if start_balance > 0:
            start_dd = max(0.0, (start_balance - balance) / start_balance * 100.0)
            if start_dd >= float(settings.MAX_DRAWDOWN_PERCENT):
                rejected["drawdown_protection"] += 1
                continue

        trade, reason = _simulate_trade_path(
            test_by_symbol[symbol], candidate["index"], candidate, balance, scenario, settings
        )
        if trade is None:
            if reason in {"below_minimum_or_invalid_lot", "actual_risk_limit", "invalid_input", "invalid_pip_economics"}:
                rejected["lot_or_risk"] += 1
            else:
                rejected["invalid_trade"] += 1
            continue

        # Drawdown includes conservative intratrade adverse excursion.
        adverse_equity = float(trade["worst_equity"])
        if peak > 0:
            dd_money = peak - adverse_equity
            dd_pct = dd_money / peak * 100.0
            max_dd_money = max(max_dd_money, dd_money)
            max_dd_pct = max(max_dd_pct, dd_pct)

        balance = round(balance + float(trade["profit"]), 2)
        peak = max(peak, balance)
        if peak > 0:
            dd_money = peak - balance
            dd_pct = dd_money / peak * 100.0
            max_dd_money = max(max_dd_money, dd_money)
            max_dd_pct = max(max_dd_pct, dd_pct)

        free_after = pd.Timestamp(trade["close_time"])
        last_close[symbol] = free_after
        trade["balance_after"] = balance
        trades.append(trade)

    profits = np.array([t["profit"] for t in trades], dtype=np.float64)
    gross_wins = float(profits[profits > 0].sum()) if len(profits) else 0.0
    gross_losses = abs(float(profits[profits < 0].sum())) if len(profits) else 0.0
    pf = gross_wins / gross_losses if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)
    wins = int(np.sum(profits > 0)) if len(profits) else 0

    monthly = {}
    for t in trades:
        month = pd.Timestamp(t["close_time"]).strftime("%Y-%m")
        monthly[month] = round(monthly.get(month, 0.0) + float(t["profit"]), 2)

    per_symbol = {}
    for symbol in SYMBOLS:
        sp = np.array([t["profit"] for t in trades if t["symbol"] == symbol], dtype=np.float64)
        if len(sp):
            sw = float(sp[sp > 0].sum())
            sl = abs(float(sp[sp < 0].sum()))
            per_symbol[symbol] = {
                "trades": int(len(sp)),
                "net_profit": float(sp.sum()),
                "win_rate": float(np.mean(sp > 0)),
                "profit_factor": float(sw / sl) if sl > 0 else (999.0 if sw > 0 else 0.0),
            }
        else:
            per_symbol[symbol] = {"trades": 0, "net_profit": 0.0, "win_rate": 0.0, "profit_factor": 0.0}

    return {
        "scenario": scenario.name,
        "start_balance": float(start_balance),
        "end_balance": float(balance),
        "net_profit": float(balance - start_balance),
        "return_pct": float((balance - start_balance) / start_balance * 100.0) if start_balance > 0 else 0.0,
        "trades": int(len(trades)),
        "wins": wins,
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "profit_factor": float(pf),
        "expectancy": float(profits.mean()) if len(profits) else 0.0,
        "max_drawdown_money": float(max_dd_money),
        "max_drawdown_pct": float(max_dd_pct),
        "total_commission": float(sum(t["commission"] for t in trades)),
        "estimated_entry_friction": float(sum(t["entry_cost_money"] for t in trades)),
        "rejected": rejected,
        "monthly_pl": monthly,
        "best_month": max(monthly.items(), key=lambda x: x[1]) if monthly else None,
        "worst_month": min(monthly.items(), key=lambda x: x[1]) if monthly else None,
        "per_symbol": per_symbol,
        "trades_detail": trades,
    }


def _public_result(result):
    return {k: v for k, v in result.items() if k != "trades_detail"}


def _gate_result(result, *, stress=False):
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


def _print_money(result):
    print(
        f"${result['start_balance']:>6.2f} -> ${result['end_balance']:>8.2f} "
        f"NET={result['net_profit']:+8.2f} ({result['return_pct']:+6.1f}%) "
        f"TRADES={result['trades']:,} WIN={result['win_rate']*100:5.1f}% "
        f"PF={result['profit_factor']:.3f} DD={result['max_drawdown_pct']:.2f}%"
    )


def run():
    settings = _load_prod_settings()
    print("\n" + "═" * 80)
    print("🏭 TRADEAI STAGE 4 — PRODUCTION MONEY VALIDATION")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🔐 Feature hash   : {FEATURE_HASH}")
    print(f"🎯 Target         : {TARGET_VERSION}")
    print("🔒 Split          : 70% train / 15% calibration / 15% final test + 24-bar purges")

    df, features = _load_data()
    artifact = joblib.load(MODEL_PATH)
    validate_model_artifact(
        artifact,
        expected_symbols=SYMBOLS,
        expected_timeframe=TIMEFRAME_NAME,
        expected_target_version=TARGET_VERSION,
    )

    policies = {}
    test_by_symbol = {}
    test_candidates = []
    symbol_validation = {}

    for symbol in SYMBOLS:
        print("\n" + "─" * 80)
        print(f"📈 CALIBRATE + FINAL TEST PREDICTIONS: {symbol}")
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
            raise RuntimeError(f"Insufficient Stage 4 split rows for {symbol}")

        print(f"Train={len(train):,} | Cal={len(cal):,} | Test={len(test):,}")
        print(f"Final test range: {test['time'].min()} -> {test['time'].max()}")

        model = create_model()
        y_train = train["target_class"].to_numpy(dtype=np.int32)
        model.fit(
            train[features],
            y_train,
            sample_weight=compute_weights(y_train, train["target_best_r"].to_numpy()),
            verbose=False,
        )

        cal_proba = model.predict_proba(cal[features])
        policy = _calibrate_policy(symbol, model, cal, cal_proba)
        policies[symbol] = policy

        if policy["enabled"]:
            c = policy["calibration"]
            print(
                f"Policy: conf>={policy['min_confidence']:.2f} edge>={policy['signal_threshold']:.2f} | "
                f"CAL trades={c['trades']:,} PF_R={c['profit_factor_r']:.3f} AVG_R={c['avg_r']:+.4f} "
                f"stable={c['positive_blocks']}/3"
            )
        else:
            print("Policy: DISABLED — no robust calibration threshold passed")

        test_proba = model.predict_proba(test[features])
        if policy["enabled"]:
            test_r = evaluate_probabilities(
                model,
                test["target_class"].to_numpy(dtype=np.int32),
                test_proba,
                test["target_buy_r"].to_numpy(dtype=np.float64),
                test["target_sell_r"].to_numpy(dtype=np.float64),
                min_confidence=policy["min_confidence"],
                signal_threshold=policy["signal_threshold"],
            )
            print(
                f"Final R check: trades={test_r['trades']:,} WIN={test_r['win_rate']*100:.1f}% "
                f"AVG_R={test_r['avg_r']:+.4f} PF_R={test_r['profit_factor_r']:.3f}"
            )
            symbol_validation[symbol] = {"final_r": test_r}
        else:
            symbol_validation[symbol] = {"final_r": None}

        test_by_symbol[symbol] = test
        test_candidates.extend(_build_candidates(symbol, model, test, test_proba, policy))

    enabled_symbols = [s for s, p in policies.items() if p.get("enabled")]
    if not enabled_symbols:
        raise RuntimeError("Stage 4 calibration disabled every symbol")

    normal = CostScenario(
        "NORMAL",
        float(settings.DEFAULT_SPREAD_PIPS),
        float(settings.SIMULATED_SLIPPAGE_PIPS),
        float(settings.COMMISSION_PER_LOT),
    )
    stress = CostScenario(
        "STRESS",
        float(settings.STRESS_SPREAD_PIPS),
        float(settings.STRESS_SLIPPAGE_PIPS),
        float(settings.STRESS_COMMISSION_PER_LOT),
    )

    money_results = {"NORMAL": {}, "STRESS": {}}
    primary_trades = None

    print("\n" + "═" * 80)
    print("💵 FINAL MONEY TEST — NORMAL COSTS")
    print("═" * 80)
    for balance in VALIDATION_BALANCES:
        result = _portfolio_backtest(test_candidates, test_by_symbol, balance, normal, settings)
        money_results["NORMAL"][str(balance)] = _public_result(result)
        _print_money(result)
        if float(balance) == 20.0:
            primary_trades = result["trades_detail"]

    print("\n" + "═" * 80)
    print("🔥 STRESS TEST — WIDER SPREAD / SLIPPAGE / COMMISSION")
    print("═" * 80)
    for balance in VALIDATION_BALANCES:
        result = _portfolio_backtest(test_candidates, test_by_symbol, balance, stress, settings)
        money_results["STRESS"][str(balance)] = _public_result(result)
        _print_money(result)

    gates = {}
    all_pass = True
    for balance in PRIMARY_ACCEPTANCE_BALANCES:
        key = str(float(balance))
        normal_checks, normal_pass = _gate_result(money_results["NORMAL"][key], stress=False)
        stress_checks, stress_pass = _gate_result(money_results["STRESS"][key], stress=True)
        gates[key] = {
            "normal": normal_checks,
            "normal_pass": normal_pass,
            "stress": stress_checks,
            "stress_pass": stress_pass,
        }
        all_pass = all_pass and normal_pass and stress_pass

    decision_policy = {
        "version": DECISION_POLICY_VERSION,
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "method": "70_15_15_chronological_calibration_with_purged_boundaries",
        "symbols": policies,
    }

    validation_report = {
        "stage": "stage4_production_money_validation_v1",
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "enabled_symbols": enabled_symbols,
        "decision_policy": decision_policy,
        "symbol_validation": symbol_validation,
        "money_results": money_results,
        "acceptance_gates": gates,
        "acceptance_pass": bool(all_pass),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2, default=str)
    with open(POLICY_JSON, "w", encoding="utf-8") as f:
        json.dump(decision_policy, f, indent=2, default=str)

    if primary_trades is not None:
        pd.DataFrame(primary_trades).to_csv(TRADES_CSV, index=False)

    print("\n" + "═" * 80)
    print("🚦 PRODUCTION ACCEPTANCE")
    print("═" * 80)
    for balance in PRIMARY_ACCEPTANCE_BALANCES:
        key = str(float(balance))
        g = gates[key]
        print(
            f"${balance:.0f} NORMAL={'PASS' if g['normal_pass'] else 'FAIL'} | "
            f"STRESS={'PASS' if g['stress_pass'] else 'FAIL'}"
        )

    if all_pass:
        # Promote the already-trained all-data Stage 3 models by attaching the
        # independently calibrated decision policy and Stage 4 validation proof.
        promoted = attach_decision_policy(artifact, decision_policy, validation_report)
        temp_model = MODEL_PATH.with_suffix(".stage4.tmp.pkl")
        joblib.dump(promoted, temp_model)
        os.replace(temp_model, MODEL_PATH)
        print("\n✅ STAGE 4 ACCEPTANCE: PASS")
        print(f"🔐 Calibrated production policy embedded into: {MODEL_PATH}")
        print("The ML artifact is now eligible for the TradeAI production backtest/demo gate.")
    else:
        print("\n❌ STAGE 4 ACCEPTANCE: FAIL")
        print("The model artifact was NOT promoted with a production decision policy.")
        print("TradeAI remains fail-safe because REQUIRE_CALIBRATED_POLICY=True.")

    print(f"📄 Report : {REPORT_JSON}")
    print(f"🎚 Policy : {POLICY_JSON}")
    print(f"🧾 Trades : {TRADES_CSV}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    run()
