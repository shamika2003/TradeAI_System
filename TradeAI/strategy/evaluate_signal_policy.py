from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from Trade_Bot_Training.direction_config import (
    SYMBOLS,
    processed_path,
    symbol_model_dir,
)
from TradeAI.strategy.cost_model import attach_historical_costs, load_symbol_cost_spec
from TradeAI.strategy.signal_policy import SignalPolicy
from TradeAI.strategy.strategy_config import (
    POLICY_VERSION,
    SIGNAL_REPORT_ROOT,
    ensure_strategy_directories,
    symbol_policy_dir,
    symbol_signal_report_dir,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def _load_json(path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_test_rows(symbol: str, metadata: dict) -> pd.DataFrame:
    df = pd.read_csv(processed_path(symbol), compression="gzip")
    for column in ("decision_time", "entry_time", "exit_time"):
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    start = pd.Timestamp(metadata["split"]["test_start"])

    return (
        df.loc[df["decision_time"] >= start]
        .sort_values("decision_time")
        .reset_index(drop=True)
    )


def _non_overlapping(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()

    selected = []
    next_available = None

    for index, row in rows.sort_values("entry_time").iterrows():
        entry = pd.Timestamp(row["entry_time"])
        exit_ = pd.Timestamp(row["exit_time"])

        if next_available is not None and entry < next_available:
            continue

        selected.append(index)
        next_available = exit_

    return rows.loc[selected].sort_values("entry_time").reset_index(drop=True)


def _profit_factor(values: pd.Series) -> float | None:
    positive = float(values.loc[values > 0.0].sum())
    negative = float(-values.loc[values < 0.0].sum())

    if negative <= 0.0:
        return float("inf") if positive > 0.0 else None

    return positive / negative


def _max_drawdown(values: pd.Series) -> float:
    if values.empty:
        return 0.0

    cumulative = values.astype(float).cumsum().to_numpy(dtype=float)
    equity = np.concatenate(([0.0], cumulative))
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def _summary(rows: pd.DataFrame) -> dict:
    if rows.empty:
        return {
            "trades": 0,
            "buy_trades": 0,
            "sell_trades": 0,
            "win_rate": None,
            "net_pips": 0.0,
            "mean_net_pips": None,
            "profit_factor": None,
            "max_drawdown_pips": 0.0,
        }

    values = rows["net_pips"].astype(float)

    return {
        "trades": int(len(rows)),
        "buy_trades": int((rows["action"] == "BUY_CANDIDATE").sum()),
        "sell_trades": int((rows["action"] == "SELL_CANDIDATE").sum()),
        "win_rate": float((values > 0.0).mean()),
        "net_pips": float(values.sum()),
        "mean_net_pips": float(values.mean()),
        "profit_factor": _profit_factor(values),
        "max_drawdown_pips": _max_drawdown(values),
    }


def _attach_economics(
    test: pd.DataFrame,
    scores: pd.DataFrame,
    symbol: str,
) -> pd.DataFrame:
    spec = load_symbol_cost_spec(symbol)

    out = test.reset_index(drop=True).copy()
    scored = scores.reset_index(drop=True)

    for column in scored.columns:
        if column != "symbol":
            out[column] = scored[column].to_numpy()

    move_pips = (
        out["exit_close"].astype(float)
        - out["entry_open"].astype(float)
    ) / spec.pip_size

    out["long_net_pips"] = (
        move_pips - out["buy_all_in_cost_pips"].astype(float)
    )
    out["short_net_pips"] = (
        -move_pips - out["sell_all_in_cost_pips"].astype(float)
    )

    out["net_pips"] = np.where(
        out["action"] == "BUY_CANDIDATE",
        out["long_net_pips"],
        np.where(
            out["action"] == "SELL_CANDIDATE",
            out["short_net_pips"],
            np.nan,
        ),
    )

    return out


def evaluate_symbol_policy(symbol: str) -> dict:
    symbol = str(symbol).upper()

    print()
    print("=" * 78)
    print(f"FINAL ECONOMIC TEST: {symbol}")
    print("=" * 78)

    model_metadata = _load_json(symbol_model_dir(symbol) / "metadata.json")
    policy_payload = _load_json(symbol_policy_dir(symbol) / "policy.json")

    if policy_payload.get("policy_version") != POLICY_VERSION:
        raise RuntimeError(f"{symbol}: rebuild signal policy before evaluation.")

    test = _load_test_rows(symbol, model_metadata)

    fallback = float(
        policy_payload["cost_contract"]["spread_fallback_pips"]
    )

    test, cost_meta = attach_historical_costs(
        test,
        symbol,
        spread_fallback_pips=fallback,
    )

    policy = SignalPolicy(symbol)

    scores = policy.score_frame(
        test,
        test["entry_spread_pips"].to_numpy(dtype=float),
    )

    economic = _attach_economics(
        test,
        scores,
        symbol,
    )

    candidates = economic.loc[
        economic["action"].isin(
            ["BUY_CANDIDATE", "SELL_CANDIDATE"]
        )
    ].copy()

    executed = _non_overlapping(candidates)
    metrics = _summary(executed)

    # Same primary direction model with no precision gate.
    baseline = economic.copy()
    baseline["action"] = np.where(
        baseline["p_up"] >= 0.5,
        "BUY_CANDIDATE",
        "SELL_CANDIDATE",
    )
    baseline["net_pips"] = np.where(
        baseline["action"] == "BUY_CANDIDATE",
        baseline["long_net_pips"],
        baseline["short_net_pips"],
    )
    baseline = _non_overlapping(baseline)
    baseline_metrics = _summary(baseline)

    if not executed.empty:
        executed["month"] = executed["entry_time"].dt.strftime("%Y-%m")
        monthly_rows = [
            {
                "month": month,
                **_summary(group),
            }
            for month, group in executed.groupby("month", sort=True)
        ]
    else:
        monthly_rows = []

    monthly = pd.DataFrame(monthly_rows)

    positive_months = (
        int((monthly["net_pips"] > 0.0).sum())
        if not monthly.empty
        else 0
    )
    negative_months = (
        int((monthly["net_pips"] < 0.0).sum())
        if not monthly.empty
        else 0
    )

    report_dir = symbol_signal_report_dir(symbol)
    report_dir.mkdir(parents=True, exist_ok=True)

    economic.to_csv(
        report_dir / "test_scored_rows.csv.gz",
        index=False,
        compression="gzip",
    )
    executed.to_csv(
        report_dir / "test_executed_trades.csv.gz",
        index=False,
        compression="gzip",
    )
    monthly.to_csv(
        report_dir / "test_monthly.csv",
        index=False,
    )

    payload = {
        "policy_version": POLICY_VERSION,
        "symbol": symbol,
        "evaluated_utc": datetime.now(timezone.utc).isoformat(),
        "test_start": str(test["decision_time"].min()),
        "test_end": str(test["decision_time"].max()),
        "market_rows": int(len(test)),
        "policy_trade_enabled": bool(policy_payload.get("trade_enabled", False)),
        "policy_live_enabled": bool(policy_payload.get("live_enabled", False)),
        "minimum_confidence": policy_payload.get("minimum_confidence"),
        "minimum_edge_atr": policy_payload.get("minimum_edge_atr"),
        "candidate_rows": int(len(candidates)),
        "executed_non_overlapping_rows": int(len(executed)),
        "precision_policy": metrics,
        "baseline_direction_model": baseline_metrics,
        "monthly": monthly_rows,
        "positive_months": positive_months,
        "negative_months": negative_months,
        "cost_contract": cost_meta,
        "notes": [
            "FINAL TEST is not used to select the gate.",
            "One position per symbol at a time in this evaluator.",
            "BUY uses entry spread; SELL realized result uses exit spread because historical OHLC is BID.",
        ],
    }

    (report_dir / "test_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    pf = metrics["profit_factor"]

    print(f"Policy trade enabled   : {payload['policy_trade_enabled']}")
    print(f"Policy live enabled    : {payload['policy_live_enabled']}")

    if payload["minimum_confidence"] is not None:
        print(f"Confidence gate        : {payload['minimum_confidence']:.4f}")
        print(f"Net-edge gate          : {payload['minimum_edge_atr']:+.4f} ATR")

    print(f"Candidates             : {len(candidates):,}")
    print(f"Executed trades        : {metrics['trades']:,}")
    print(
        f"BUY / SELL             : "
        f"{metrics['buy_trades']:,} / {metrics['sell_trades']:,}"
    )
    print(
        "Win rate               : "
        + (
            f"{metrics['win_rate']:.2%}"
            if metrics["win_rate"] is not None
            else "n/a"
        )
    )
    print(f"Net pips               : {metrics['net_pips']:+.2f}")
    print(
        "Profit factor          : "
        + (
            "inf"
            if pf == float("inf")
            else (
                f"{pf:.3f}"
                if pf is not None
                else "n/a"
            )
        )
    )
    print(
        "Mean net/trade         : "
        + (
            f"{metrics['mean_net_pips']:+.3f}"
            if metrics["mean_net_pips"] is not None
            else "n/a"
        )
    )
    print(f"Max drawdown           : {metrics['max_drawdown_pips']:.2f} pips")
    print(
        f"Positive / negative mo.: "
        f"{positive_months} / {negative_months}"
    )
    print(
        f"Ungated baseline net   : "
        f"{baseline_metrics['net_pips']:+.2f} pips"
    )
    print(f"Summary                : {report_dir / 'test_summary.json'}")

    return payload


def evaluate_all_policies() -> None:
    ensure_strategy_directories()

    summaries = {
        symbol: evaluate_symbol_policy(symbol)
        for symbol in SYMBOLS
    }

    combined_path = SIGNAL_REPORT_ROOT / "test_summary_all.json"
    combined_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "policy_version": POLICY_VERSION,
                    "evaluated_utc": datetime.now(timezone.utc).isoformat(),
                    "symbols": summaries,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("FINAL ECONOMIC TEST COMPLETE")
    print("=" * 78)
    print(f"Combined: {combined_path}")


if __name__ == "__main__":
    evaluate_all_policies()
