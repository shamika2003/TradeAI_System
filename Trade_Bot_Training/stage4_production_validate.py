# filename: Trade_Bot_Traning/stage4_production_validate.py

from __future__ import annotations

import importlib.util
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
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
    POLICY_CALIBRATION_BALANCE,
    POLICY_CALIBRATION_MIN_PAYOFF_RATIO,
    POLICY_CALIBRATION_MIN_TRADES,
    POLICY_CALIBRATION_NORMAL_MAX_DRAWDOWN_PERCENT,
    POLICY_CALIBRATION_NORMAL_MIN_PROFIT_FACTOR,
    POLICY_CALIBRATION_RISK_PERCENT,
    POLICY_CALIBRATION_STRESS_MAX_DRAWDOWN_PERCENT,
    POLICY_CALIBRATION_STRESS_MIN_PROFIT_FACTOR,
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
    require_training_window,
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
from shared.tradeai_core.training_window import apply_supervised_training_cutoff
from shared.tradeai_core.validation_execution import ValidationExecutionContext


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


def _load_data(cutoff_exclusive=None):
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

    df, window = apply_supervised_training_cutoff(
        df,
        cutoff_exclusive=cutoff_exclusive,
        label_horizon_bars=MAX_TARGET_HORIZON_BARS,
    )
    return df, features, window.as_dict()


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


def _attach_management_state(data, model, proba, policy):
    """Attach the causal model state used by Stage 3 trade management.

    Each row represents the model decision available after that row has closed.
    The path simulator consumes row i before executing candle i+1, matching the
    runtime replay ordering.
    """
    p_sell, p_hold, p_buy = probability_columns(model, proba)
    signal = p_buy - p_sell
    confidence = np.maximum(p_buy, p_sell)
    signal = np.where(p_hold >= confidence, 0.0, signal)

    enriched = data.copy().reset_index(drop=True)
    enriched["_mgmt_signal"] = signal.astype(np.float64)
    enriched["_mgmt_confidence"] = confidence.astype(np.float64)
    enriched["_mgmt_min_confidence"] = float(policy.get("min_confidence", 1.0))
    enriched["_mgmt_signal_threshold"] = float(policy.get("signal_threshold", 1.0))
    enriched["_mgmt_policy_enabled"] = bool(policy.get("enabled", True))
    return enriched


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


def _simulate_trade_path(data, idx, candidate, balance, scenario, settings, execution=None):
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

    execution = execution or ValidationExecutionContext(settings)
    costs = execution.costs(symbol, scenario.name)
    stop_loss = execution.normalize_price(symbol, stop_loss)
    take_profit = execution.normalize_price(symbol, take_profit)

    max_actual = min(
        float(settings.MAX_ACTUAL_RISK_PERCENT),
        float(settings.RISK_PERCENT) * 1.25,
    )
    lot, risk_info = execution.net_risk_sized_lot(
        symbol=symbol,
        direction=direction,
        requested_price=requested,
        stop_loss=stop_loss,
        balance=float(balance),
        risk_percent=float(settings.RISK_PERCENT),
        costs=costs,
        max_actual_risk_percent=max_actual,
    )
    if lot is None:
        return None, risk_info.get("reason", "lot_rejected")

    entry = float(risk_info["entry_price"])
    minimum_stop_distance = execution.minimum_stop_distance(symbol)

    if direction == "BUY":
        if not (stop_loss < entry < take_profit):
            return None, "entry_cost_invalidates_stops"
        if minimum_stop_distance > 0 and (
            (entry - stop_loss) < minimum_stop_distance
            or (take_profit - entry) < minimum_stop_distance
        ):
            return None, "broker_stop_distance"
    else:
        if not (take_profit < entry < stop_loss):
            return None, "entry_cost_invalidates_stops"
        if minimum_stop_distance > 0 and (
            (stop_loss - entry) < minimum_stop_distance
            or (entry - take_profit) < minimum_stop_distance
        ):
            return None, "broker_stop_distance"

    pip = pip_size(symbol)
    pip_value = execution.pip_value_per_lot(symbol, entry)
    commission = float(costs.commission_per_lot) * float(lot)
    initial_risk_price = abs(entry - stop_loss)
    if initial_risk_price <= 0:
        return None, "invalid_initial_risk"

    per_pip_money = max(pip_value * lot, 1e-12)
    commission_pips = commission / per_pip_money
    min_profit_pips = max(0.0, float(getattr(settings, "BREAK_EVEN_MIN_PROFIT_MONEY", 0.0))) / per_pip_money
    positive_buffer_pips = max(
        max(0.0, float(settings.BREAK_EVEN_BUFFER_PIPS)),
        min_profit_pips,
    )
    be_offset_price = (commission_pips + positive_buffer_pips) * pip
    cost_be = (
        entry + be_offset_price
        if direction == "BUY"
        else entry - be_offset_price
    )
    cost_be = execution.normalize_price(symbol, cost_be)

    be_done = False
    current_sl = float(stop_loss)
    current_mark = float(entry)
    exit_price = None
    exit_reason = None
    exit_time = None
    worst_equity = float(balance)
    bars_held = 0

    max_steps = min(MAX_TARGET_HORIZON_BARS, len(data) - idx - 1)
    for step in range(1, max_steps + 1):
        # -------------------------------------------------
        # Stage 3 management BEFORE the next execution bar.
        # -------------------------------------------------
        bars_held += 1
        management_row = data.iloc[idx + step - 1]
        management_time = pd.Timestamp(management_row["time"])

        r_multiple = (
            (current_mark - entry) / initial_risk_price
            if direction == "BUY"
            else (entry - current_mark) / initial_risk_price
        )

        if bool(settings.USE_MAX_HOLD) and bars_held >= int(settings.MAX_HOLD_BARS):
            exit_price = current_mark
            exit_reason = "MAX_HOLD"
            exit_time = management_time
            break

        if bool(getattr(settings, "USE_AI_DEFENSIVE_EXIT", False)):
            adverse_gate = -abs(float(settings.AI_DEFENSIVE_EXIT_ADVERSE_R))
            min_bars = int(settings.AI_DEFENSIVE_EXIT_MIN_BARS)
            if bars_held >= min_bars and r_multiple <= adverse_gate:
                enabled = bool(management_row.get("_mgmt_policy_enabled", True))
                signal = float(management_row.get("_mgmt_signal", 0.0) or 0.0)
                confidence = float(management_row.get("_mgmt_confidence", 0.0) or 0.0)
                min_confidence = float(management_row.get("_mgmt_min_confidence", 1.0) or 1.0)
                threshold = abs(float(management_row.get("_mgmt_signal_threshold", 1.0) or 1.0))
                required_signal = threshold * max(
                    0.0,
                    float(settings.AI_DEFENSIVE_EXIT_SIGNAL_MULTIPLIER),
                )
                opposite = (
                    signal <= -required_signal
                    if direction == "BUY"
                    else signal >= required_signal
                )
                if enabled and confidence >= min_confidence and opposite:
                    exit_price = current_mark
                    exit_reason = "AI_DEFENSIVE_EXIT"
                    exit_time = management_time
                    break

        if (
            bool(settings.USE_BREAK_EVEN)
            and not be_done
            and r_multiple >= float(settings.BREAK_EVEN_TRIGGER_R)
        ):
            if direction == "BUY" and cost_be > current_sl:
                current_sl = cost_be
                be_done = True
            elif direction == "SELL" and cost_be < current_sl:
                current_sl = cost_be
                be_done = True

        if (
            bool(getattr(settings, "USE_PROFIT_LOCK", False))
            and r_multiple >= float(getattr(settings, "PROFIT_LOCK_TRIGGER_R", 999.0))
        ):
            lock_r = max(0.0, float(getattr(settings, "PROFIT_LOCK_R", 0.0)))
            lock_distance = initial_risk_price * lock_r
            if direction == "BUY":
                new_sl = max(entry + lock_distance, cost_be)
                new_sl = execution.normalize_price(symbol, new_sl)
                if new_sl > current_sl:
                    current_sl = new_sl
                    be_done = True
            else:
                new_sl = min(entry - lock_distance, cost_be)
                new_sl = execution.normalize_price(symbol, new_sl)
                if new_sl < current_sl:
                    current_sl = new_sl
                    be_done = True

        if (
            bool(settings.USE_TRAILING_STOP)
            and r_multiple >= float(settings.TRAILING_TRIGGER_R)
        ):
            atr_prev = float(management_row["atr"])
            if atr_prev > 0:
                trail_distance = atr_prev * max(0.0, float(settings.TRAILING_ATR_MULTIPLIER))
                if direction == "BUY":
                    new_sl = current_mark - trail_distance
                    if be_done:
                        new_sl = max(new_sl, cost_be)
                    new_sl = execution.normalize_price(symbol, new_sl)
                    if new_sl > current_sl:
                        current_sl = new_sl
                else:
                    new_sl = current_mark + trail_distance
                    if be_done:
                        new_sl = min(new_sl, cost_be)
                    new_sl = execution.normalize_price(symbol, new_sl)
                    if new_sl < current_sl:
                        current_sl = new_sl

        # -------------------------------------------------
        # Execute the next complete historical BID candle.
        # -------------------------------------------------
        future = data.iloc[idx + step]
        high = float(future["high"])
        low = float(future["low"])
        close = float(future["close"])
        exit_time = pd.Timestamp(future["time"])
        sell_high, sell_low, sell_close = execution.sell_side_prices(
            symbol,
            high=high,
            low=low,
            close=close,
            costs=costs,
        )

        adverse_price = low if direction == "BUY" else sell_high
        adverse_diff = (
            adverse_price - entry
            if direction == "BUY"
            else entry - adverse_price
        )
        adverse_pips = adverse_diff / pip
        adverse_net = adverse_pips * pip_value * lot - commission
        worst_equity = min(worst_equity, float(balance) + adverse_net)

        if direction == "BUY":
            sl_hit = low <= current_sl
            tp_hit = high >= take_profit
        else:
            sl_hit = sell_high >= current_sl
            tp_hit = sell_low <= take_profit

        # Same-candle ambiguity is intentionally conservative: stop wins.
        if sl_hit:
            exit_price = current_sl
            exit_reason = "BREAK_EVEN" if be_done and abs(current_sl - cost_be) <= pip * 0.11 else "STOP_LOSS"
            break
        if tp_hit:
            exit_price = take_profit
            exit_reason = "TAKE_PROFIT"
            break

        current_mark = close if direction == "BUY" else sell_close
        current_mark = execution.normalize_price(symbol, current_mark)

    if exit_price is None or exit_time is None:
        return None, "no_exit"

    difference = (
        float(exit_price) - entry
        if direction == "BUY"
        else entry - float(exit_price)
    )
    pips = difference / pip
    gross = pips * pip_value * lot
    net = round(gross - commission, 2)

    # Friction relative to the requested BID close, used only for reporting.
    if direction == "BUY":
        entry_cost_pips = max(0.0, (entry - requested) / pip)
    else:
        entry_cost_pips = max(0.0, (requested - entry) / pip)
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
        "planned_net_risk": float(risk_info.get("net_risk", 0.0)),
        "allowed_risk": float(risk_info.get("allowed_risk", 0.0)),
        "worst_equity": float(worst_equity),
        "bars_held": int(bars_held),
        "spread_pips": float(costs.spread_pips),
        "slippage_pips": float(costs.slippage_pips),
        "commission_per_lot": float(costs.commission_per_lot),
    }, None

def _portfolio_backtest(candidates, test_by_symbol, start_balance, scenario, settings, execution=None):
    """Chronological portfolio simulator with bounded concurrency.

    Stage 7 removes the old single-global-position shortcut. Runtime permits one
    position per symbol and up to MAX_OPEN_POSITIONS globally, so validation now
    does the same while reserving each trade's planned loss-at-stop against a
    portfolio risk cap.
    """
    execution = execution or ValidationExecutionContext(settings)
    symbol_priority = {symbol: i for i, symbol in enumerate(SYMBOLS)}
    candidates = sorted(
        candidates,
        key=lambda c: (pd.Timestamp(c["time"]), symbol_priority.get(c["symbol"], 999)),
    )

    balance = float(start_balance)
    peak = balance
    max_dd_pct = 0.0
    max_dd_money = 0.0
    last_close = {}
    day_key = None
    day_start_balance = balance
    trades = []
    active = []

    max_positions = max(1, int(getattr(settings, "MAX_OPEN_POSITIONS", 1)))
    portfolio_risk_pct = max(
        0.0,
        float(getattr(settings, "MAX_PORTFOLIO_RISK_PERCENT", 100.0)),
    )

    rejected = {
        "position_busy": 0,
        "portfolio_risk": 0,
        "cooldown": 0,
        "daily_loss": 0,
        "drawdown_protection": 0,
        "lot_or_risk": 0,
        "invalid_trade": 0,
    }

    def update_drawdown(equity_value):
        nonlocal max_dd_pct, max_dd_money
        if peak <= 0:
            return
        dd_money = max(0.0, peak - float(equity_value))
        dd_pct = dd_money / peak * 100.0
        max_dd_money = max(max_dd_money, dd_money)
        max_dd_pct = max(max_dd_pct, dd_pct)

    def realize_trade(trade):
        nonlocal balance, peak
        balance = round(balance + float(trade["profit"]), 2)
        peak = max(peak, balance)
        update_drawdown(balance)
        close_time = pd.Timestamp(trade["close_time"])
        last_close[trade["symbol"]] = close_time
        trade["balance_after"] = balance
        trades.append(trade)

    def realize_until(timestamp):
        nonlocal active
        timestamp = pd.Timestamp(timestamp)
        closing = [t for t in active if pd.Timestamp(t["close_time"]) <= timestamp]
        if not closing:
            return
        closing.sort(
            key=lambda t: (
                pd.Timestamp(t["close_time"]),
                symbol_priority.get(t["symbol"], 999),
            )
        )
        for trade in closing:
            realize_trade(trade)
            active.remove(trade)

    for candidate in candidates:
        time = pd.Timestamp(candidate["time"])
        symbol = candidate["symbol"]

        # Realize positions that are already closed by this decision point.
        realize_until(time)

        if any(t["symbol"] == symbol for t in active):
            rejected["position_busy"] += 1
            continue
        if len(active) >= max_positions:
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
            daily_loss_pct = max(
                0.0,
                (day_start_balance - balance) / day_start_balance * 100.0,
            )
            if daily_loss_pct >= float(settings.MAX_DAILY_LOSS_PERCENT):
                rejected["daily_loss"] += 1
                continue

        if start_balance > 0:
            start_dd = max(0.0, (start_balance - balance) / start_balance * 100.0)
            if start_dd >= float(settings.MAX_DRAWDOWN_PERCENT):
                rejected["drawdown_protection"] += 1
                continue

        trade, reason = _simulate_trade_path(
            test_by_symbol[symbol],
            candidate["index"],
            candidate,
            balance,
            scenario,
            settings,
            execution,
        )
        if trade is None:
            if reason in {
                "below_minimum_or_invalid_lot",
                "actual_risk_limit",
                "invalid_input",
                "invalid_pip_economics",
            }:
                rejected["lot_or_risk"] += 1
            else:
                rejected["invalid_trade"] += 1
            continue

        # Reserve initial net risk for every open position. Validation stays
        # conservative by not releasing that reserve until the position closes,
        # even though runtime may free headroom sooner after stop improvements.
        active_risk = sum(max(0.0, float(t.get("planned_net_risk", 0.0))) for t in active)
        new_risk = max(0.0, float(trade.get("planned_net_risk", 0.0)))
        portfolio_cap_money = balance * portfolio_risk_pct / 100.0
        if active_risk + new_risk > portfolio_cap_money + 1e-12:
            rejected["portfolio_risk"] += 1
            continue

        trade["max_adverse_loss"] = max(
            0.0,
            float(balance) - float(trade.get("worst_equity", balance)),
        )
        active.append(trade)

        # Conservative portfolio excursion bound: assume each concurrently open
        # trade can experience its own recorded worst adverse excursion together.
        exposure_floor = balance - sum(
            max(0.0, float(t.get("max_adverse_loss", 0.0))) for t in active
        )
        update_drawdown(exposure_floor)

    # Flush remaining positions in true close-time order.
    for trade in sorted(
        list(active),
        key=lambda t: (
            pd.Timestamp(t["close_time"]),
            symbol_priority.get(t["symbol"], 999),
        ),
    ):
        realize_trade(trade)
    active.clear()

    profits = np.array([t["profit"] for t in trades], dtype=np.float64)
    wins_arr = profits[profits > 0]
    losses_arr = profits[profits < 0]
    gross_wins = float(wins_arr.sum()) if len(wins_arr) else 0.0
    gross_losses = abs(float(losses_arr.sum())) if len(losses_arr) else 0.0
    pf = gross_wins / gross_losses if gross_losses > 0 else (999.0 if gross_wins > 0 else 0.0)
    wins = int(len(wins_arr))
    avg_win = float(np.mean(wins_arr)) if len(wins_arr) else 0.0
    avg_loss = float(np.mean(losses_arr)) if len(losses_arr) else 0.0
    payoff_ratio = (
        avg_win / abs(avg_loss)
        if avg_win > 0 and avg_loss < 0
        else (999.0 if avg_win > 0 and avg_loss == 0 else 0.0)
    )

    monthly = {}
    for t in trades:
        month = pd.Timestamp(t["close_time"]).strftime("%Y-%m")
        monthly[month] = round(monthly.get(month, 0.0) + float(t["profit"]), 2)

    per_symbol = {}
    for symbol in SYMBOLS:
        sp = np.array([t["profit"] for t in trades if t["symbol"] == symbol], dtype=np.float64)
        if len(sp):
            sw_arr = sp[sp > 0]
            sl_arr = sp[sp < 0]
            sw = float(sw_arr.sum()) if len(sw_arr) else 0.0
            sl = abs(float(sl_arr.sum())) if len(sl_arr) else 0.0
            saw = float(sw_arr.mean()) if len(sw_arr) else 0.0
            sal = float(sl_arr.mean()) if len(sl_arr) else 0.0
            per_symbol[symbol] = {
                "trades": int(len(sp)),
                "net_profit": float(sp.sum()),
                "win_rate": float(np.mean(sp > 0)),
                "profit_factor": float(sw / sl) if sl > 0 else (999.0 if sw > 0 else 0.0),
                "average_win": saw,
                "average_loss": sal,
                "payoff_ratio": float(saw / abs(sal)) if saw > 0 and sal < 0 else 0.0,
            }
        else:
            per_symbol[symbol] = {
                "trades": 0,
                "net_profit": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "average_win": 0.0,
                "average_loss": 0.0,
                "payoff_ratio": 0.0,
            }

    return {
        "scenario": scenario.name,
        "start_balance": float(start_balance),
        "end_balance": float(balance),
        "net_profit": float(balance - start_balance),
        "return_pct": float((balance - start_balance) / start_balance * 100.0) if start_balance > 0 else 0.0,
        "trades": int(len(trades)),
        "wins": wins,
        "losses": int(len(losses_arr)),
        "breakeven_trades": int(np.sum(profits == 0)) if len(profits) else 0,
        "win_rate": float(wins / len(trades)) if trades else 0.0,
        "profit_factor": float(pf),
        "expectancy": float(profits.mean()) if len(profits) else 0.0,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "payoff_ratio": float(payoff_ratio),
        "max_drawdown_money": float(max_dd_money),
        "max_drawdown_pct": float(max_dd_pct),
        "max_concurrent_positions": int(max_positions),
        "portfolio_risk_cap_percent": float(portfolio_risk_pct),
        "total_commission": float(sum(t["commission"] for t in trades)),
        "estimated_entry_friction": float(sum(t["entry_cost_money"] for t in trades)),
        "rejected": rejected,
        "monthly_pl": monthly,
        "best_month": max(monthly.items(), key=lambda x: x[1]) if monthly else None,
        "worst_month": min(monthly.items(), key=lambda x: x[1]) if monthly else None,
        "per_symbol": per_symbol,
        "trades_detail": trades,
    }


class _RiskOverrideSettings:
    def __init__(self, base, risk_percent: float):
        self._base = base
        self.RISK_PERCENT = float(risk_percent)

    def __getattr__(self, name):
        return getattr(self._base, name)


def _calibrate_policy_money(
    symbol,
    model,
    cal,
    proba,
    settings,
    execution,
    normal,
    stress,
):
    """Choose thresholds from executable money P/L, not label R alone.

    Label-R remains a cheap robustness pre-filter. A threshold pair is eligible
    for production only if the same signals survive broker lot rules, spread,
    commission, trade management and stress costs on the calibration slice.
    """
    best = None
    proxy = _RiskOverrideSettings(settings, POLICY_CALIBRATION_RISK_PERCENT)

    y = cal["target_class"].to_numpy(dtype=np.int32)
    buy_r = cal["target_buy_r"].to_numpy(dtype=np.float64)
    sell_r = cal["target_sell_r"].to_numpy(dtype=np.float64)

    for min_conf in CALIBRATION_CONFIDENCE_GRID:
        for edge in CALIBRATION_EDGE_GRID:
            label_metrics = evaluate_probabilities(
                model,
                y,
                proba,
                buy_r,
                sell_r,
                min_confidence=min_conf,
                signal_threshold=edge,
            )
            if label_metrics["trades"] < CALIBRATION_MIN_TRADES:
                continue
            if label_metrics["coverage"] > CALIBRATION_MAX_COVERAGE:
                continue
            if label_metrics["avg_r"] <= 0:
                continue
            if label_metrics["profit_factor_r"] < CALIBRATION_MIN_PROFIT_FACTOR:
                continue

            positive_blocks, block_std, _ = _block_stability(
                model, proba, cal, min_conf, edge
            )
            if positive_blocks < 2:
                continue

            policy = {
                "enabled": True,
                "min_confidence": float(min_conf),
                "signal_threshold": float(edge),
            }
            enriched = _attach_management_state(cal, model, proba, policy)
            policy_candidates = _build_candidates(symbol, model, enriched, proba, policy)
            if len(policy_candidates) < POLICY_CALIBRATION_MIN_TRADES:
                continue

            by_symbol = {symbol: enriched}
            normal_result = _portfolio_backtest(
                policy_candidates,
                by_symbol,
                POLICY_CALIBRATION_BALANCE,
                normal,
                proxy,
                execution=execution,
            )
            stress_result = _portfolio_backtest(
                policy_candidates,
                by_symbol,
                POLICY_CALIBRATION_BALANCE,
                stress,
                proxy,
                execution=execution,
            )

            checks = {
                "normal_profit": normal_result["net_profit"] > 0,
                "stress_profit": stress_result["net_profit"] >= 0,
                "normal_pf": normal_result["profit_factor"] >= POLICY_CALIBRATION_NORMAL_MIN_PROFIT_FACTOR,
                "stress_pf": stress_result["profit_factor"] >= POLICY_CALIBRATION_STRESS_MIN_PROFIT_FACTOR,
                "normal_payoff": normal_result["payoff_ratio"] >= POLICY_CALIBRATION_MIN_PAYOFF_RATIO,
                "normal_trades": normal_result["trades"] >= POLICY_CALIBRATION_MIN_TRADES,
                "stress_trades": stress_result["trades"] >= POLICY_CALIBRATION_MIN_TRADES,
                "normal_dd": normal_result["max_drawdown_pct"] <= POLICY_CALIBRATION_NORMAL_MAX_DRAWDOWN_PERCENT,
                "stress_dd": stress_result["max_drawdown_pct"] <= POLICY_CALIBRATION_STRESS_MAX_DRAWDOWN_PERCENT,
            }
            if not all(checks.values()):
                continue

            worst_return = min(normal_result["return_pct"], stress_result["return_pct"])
            mean_return = 0.5 * (normal_result["return_pct"] + stress_result["return_pct"])
            worst_pf = min(normal_result["profit_factor"], stress_result["profit_factor"])
            worst_dd = max(normal_result["max_drawdown_pct"], stress_result["max_drawdown_pct"])
            payoff = normal_result["payoff_ratio"]

            score = (
                worst_return
                + 0.25 * mean_return
                + 2.0 * max(0.0, worst_pf - 1.0)
                + 0.75 * min(payoff, 3.5)
                + 0.05 * positive_blocks
                - 0.10 * worst_dd
                - 0.05 * block_std
            )

            candidate = {
                "enabled": True,
                "min_confidence": float(min_conf),
                "signal_threshold": float(edge),
                "score": float(score),
                "calibration": {
                    "method": "broker_money_normal_and_stress",
                    "reference_balance": float(POLICY_CALIBRATION_BALANCE),
                    "reference_risk_percent": float(POLICY_CALIBRATION_RISK_PERCENT),
                    "trades": int(label_metrics["trades"]),
                    "coverage": float(label_metrics["coverage"]),
                    "win_rate": float(label_metrics["win_rate"]),
                    "avg_r": float(label_metrics["avg_r"]),
                    "profit_factor_r": float(label_metrics["profit_factor_r"]),
                    "positive_blocks": int(positive_blocks),
                    "block_avg_r_std": float(block_std),
                    "normal_money": _public_result(normal_result),
                    "stress_money": _public_result(stress_result),
                    "checks": checks,
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
                "reason": "no_threshold_pair_passed_execution_money_gates",
                "reference_balance": float(POLICY_CALIBRATION_BALANCE),
                "reference_risk_percent": float(POLICY_CALIBRATION_RISK_PERCENT),
            },
        }

    return best

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
        f"PF={result['profit_factor']:.3f} PAY={result.get('payoff_ratio', 0.0):.2f} "
        f"DD={result['max_drawdown_pct']:.2f}%"
    )


def run():
    settings = _load_prod_settings()
    execution = ValidationExecutionContext(
        settings,
        require_profile=bool(getattr(settings, "USE_BROKER_PROFILE", False)),
    )
    execution.assert_symbols_profiled(SYMBOLS)
    print("\n" + "═" * 80)
    print("🏭 TRADEAI STAGE 4 — PRODUCTION MONEY VALIDATION")
    print("═" * 80)
    print(f"🧬 Feature schema : {FEATURE_SCHEMA_VERSION}")
    print(f"🔐 Feature hash   : {FEATURE_HASH}")
    print(f"🎯 Target         : {TARGET_VERSION}")
    print(
        f"🔒 Split          : {CALIBRATION_TRAIN_FRACTION*100:.0f}% train / "
        f"{(CALIBRATION_END_FRACTION-CALIBRATION_TRAIN_FRACTION)*100:.0f}% calibration / "
        f"{(1.0-CALIBRATION_END_FRACTION)*100:.0f}% final test "
        f"+ {MAX_TARGET_HORIZON_BARS}-bar purges"
    )
    print("🔒 Stage 4 role   : diagnostic only; Stage 5 owns exact-model promotion")

    artifact = joblib.load(MODEL_PATH)
    validate_model_artifact(
        artifact,
        expected_symbols=SYMBOLS,
        expected_timeframe=TIMEFRAME_NAME,
        expected_target_version=TARGET_VERSION,
    )
    artifact_window = require_training_window(artifact)
    df, features, loaded_window = _load_data(artifact_window.get("cutoff_exclusive"))

    if int(loaded_window["rows"]) != int(artifact_window["rows"]):
        raise RuntimeError(
            "Stage 4 training-window row count differs from the trained artifact; "
            "dataset/model provenance no longer matches"
        )
    if str(loaded_window["fit_end"]) != str(artifact_window["fit_end"]):
        raise RuntimeError(
            "Stage 4 training-window end differs from the trained artifact; "
            "dataset/model provenance no longer matches"
        )

    print(
        f"🔒 Artifact window : {artifact_window['mode']} | "
        f"fit_end={artifact_window['fit_end']} | "
        f"backtest_safe_from={artifact_window['backtest_safe_from']}"
    )

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
        policy = _calibrate_policy_money(
            symbol, model, cal, cal_proba, settings, execution, normal, stress
        )
        policies[symbol] = policy

        if policy["enabled"]:
            c = policy["calibration"]
            nm = c.get("normal_money", {})
            sm = c.get("stress_money", {})
            print(
                f"Policy: conf>={policy['min_confidence']:.2f} edge>={policy['signal_threshold']:.2f} | "
                f"CAL PF_R={c['profit_factor_r']:.3f} AVG_R={c['avg_r']:+.4f} | "
                f"MONEY PF={nm.get('profit_factor', 0.0):.3f}/{sm.get('profit_factor', 0.0):.3f} "
                f"PAY={nm.get('payoff_ratio', 0.0):.2f}"
            )
        else:
            reason = policy.get("calibration", {}).get("reason", "no_money_safe_threshold")
            print(f"Policy: DISABLED — {reason}")

        test_proba = model.predict_proba(test[features])
        test = _attach_management_state(test, model, test_proba, policy)
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

    money_results = {"NORMAL": {}, "STRESS": {}}
    primary_trades = None

    print("\n" + "═" * 80)
    print("💵 FINAL MONEY TEST — NORMAL COSTS")
    print("═" * 80)
    for balance in VALIDATION_BALANCES:
        result = _portfolio_backtest(test_candidates, test_by_symbol, balance, normal, settings)
        money_results["NORMAL"][str(balance)] = _public_result(result)
        _print_money(result)
        if float(balance) == float(PRIMARY_ACCEPTANCE_BALANCES[0]):
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
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "recent_chronological_calibration_with_purged_boundaries",
        "symbols": policies,
    }

    validation_report = {
        "stage": "stage4_production_money_validation_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_hash": FEATURE_HASH,
        "target_version": TARGET_VERSION,
        "training_window": artifact_window,
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
        print("\n✅ STAGE 4 DIAGNOSTIC: PASS")
        print(
            "No model is promoted here. Stage 5 must calibrate risk and "
            "promote the exact model objects that produced the validation result."
        )
    else:
        print("\n❌ STAGE 4 DIAGNOSTIC: FAIL")
        print(
            "Stage 4 remains diagnostic. Stage 5 is the final authority and "
            "will fail closed if no exact-model configuration passes."
        )

    print(f"📄 Report : {REPORT_JSON}")
    print(f"🎚 Policy : {POLICY_JSON}")
    print(f"🧾 Trades : {TRADES_CSV}")
    print("═" * 80 + "\n")


if __name__ == "__main__":
    run()