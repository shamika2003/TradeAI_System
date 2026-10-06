from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from Trade_Bot_Training.direction_config import (
    MODEL_VERSION,
    PURGE_MINUTES,
    SYMBOLS,
    processed_path,
    symbol_model_dir,
)
from Trade_Bot_Training.direction_model import DirectionModel
from TradeAI.strategy.cost_model import (
    attach_historical_costs,
    derive_spread_fallback,
    load_symbol_cost_spec,
)
from TradeAI.strategy.strategy_config import (
    CONFIDENCE_QUANTILES,
    EDGE_QUANTILES,
    META_SEARCH_FRACTION,
    MIN_CONFIRM_TRADES,
    MIN_PROFIT_FACTOR,
    MIN_SEARCH_TRADES,
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


def _load_meta_rows(symbol: str, metadata: dict) -> pd.DataFrame:
    df = pd.read_csv(processed_path(symbol), compression="gzip")
    for column in ("decision_time", "entry_time", "exit_time"):
        df[column] = pd.to_datetime(df[column], utc=True, errors="coerce")

    split = metadata["split"]
    calibration_end = pd.Timestamp(split["calibration_end"])
    meta_end = pd.Timestamp(split["meta_end"])
    purge = pd.Timedelta(minutes=int(split["purge_minutes"]))

    rows = df.loc[
        (df["decision_time"] > calibration_end + purge)
        & (df["decision_time"] <= meta_end)
    ].copy()

    return rows.sort_values("decision_time").reset_index(drop=True)


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


def _stats(rows: pd.DataFrame) -> dict:
    executed = _non_overlapping(rows)
    values = executed["realized_net_pips"].astype(float)

    if executed.empty:
        return {
            "trades": 0,
            "win_rate": None,
            "net_pips": 0.0,
            "mean_net_pips": None,
            "profit_factor": None,
            "max_drawdown_pips": 0.0,
        }

    return {
        "trades": int(len(executed)),
        "win_rate": float((values > 0.0).mean()),
        "net_pips": float(values.sum()),
        "mean_net_pips": float(values.mean()),
        "profit_factor": _profit_factor(values),
        "max_drawdown_pips": _max_drawdown(values),
    }


def _build_frame(
    rows: pd.DataFrame,
    symbol: str,
    model: DirectionModel,
) -> pd.DataFrame:
    spec = load_symbol_cost_spec(symbol)
    pred = model.predict_proba(rows).reset_index(drop=True)
    frame = rows.reset_index(drop=True).copy()

    frame["p_up"] = pred["p_up"].to_numpy(dtype=float)
    frame["p_down"] = pred["p_down"].to_numpy(dtype=float)
    frame["confidence"] = pred["confidence"].to_numpy(dtype=float)
    frame["ensemble_disagreement"] = pred["ensemble_disagreement"].to_numpy(dtype=float)
    frame["expected_move_atr"] = pred["expected_move_atr"].to_numpy(dtype=float)

    frame["selected_side"] = np.where(frame["p_up"] >= 0.5, "LONG", "SHORT")
    side_sign = np.where(frame["selected_side"] == "LONG", 1.0, -1.0)
    frame["expected_side_atr"] = side_sign * frame["expected_move_atr"].astype(float)

    close = pd.to_numeric(frame["close"], errors="coerce")
    atr_pct = pd.to_numeric(frame["m5_atr_pct_14"], errors="coerce")
    frame["atr_pips"] = close * atr_pct / spec.pip_size

    # Only information known at entry is used for the live gate.
    frame["live_cost_pips"] = (
        frame["entry_spread_pips"].astype(float)
        + frame["commission_pips"].astype(float)
        + frame["slippage_pips"].astype(float)
    )
    frame["cost_atr"] = (
        frame["live_cost_pips"]
        / frame["atr_pips"].replace(0.0, np.nan)
    )

    # Expected move in the chosen direction after currently-known transaction cost.
    frame["edge_proxy_atr"] = frame["expected_side_atr"] - frame["cost_atr"]

    raw_move_pips = (
        frame["exit_close"].astype(float)
        - frame["entry_open"].astype(float)
    ) / spec.pip_size

    frame["long_net_pips"] = (
        raw_move_pips - frame["buy_all_in_cost_pips"].astype(float)
    )
    frame["short_net_pips"] = (
        -raw_move_pips - frame["sell_all_in_cost_pips"].astype(float)
    )
    frame["realized_net_pips"] = np.where(
        frame["selected_side"] == "LONG",
        frame["long_net_pips"],
        frame["short_net_pips"],
    )

    required = [
        "decision_time", "entry_time", "exit_time",
        "confidence", "expected_side_atr", "edge_proxy_atr",
        "ensemble_disagreement", "realized_net_pips",
    ]
    return (
        frame.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .sort_values("decision_time")
        .reset_index(drop=True)
    )


def _split_search_confirm(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    times = pd.to_datetime(frame["decision_time"], utc=True)
    unique_times = times.drop_duplicates().sort_values().reset_index(drop=True)

    if len(unique_times) < 4000:
        raise RuntimeError("Reserved META partition is too small.")

    cut_idx = max(
        1,
        min(
            len(unique_times) - 2,
            int(len(unique_times) * META_SEARCH_FRACTION) - 1,
        ),
    )
    cut = pd.Timestamp(unique_times.iloc[cut_idx])
    purge = pd.Timedelta(minutes=PURGE_MINUTES)

    search = frame.loc[times <= cut].copy()
    confirm = frame.loc[times > cut + purge].copy()

    if len(search) < 1000 or len(confirm) < 1000:
        raise RuntimeError("SEARCH/CONFIRM split is too small.")

    return (
        search.reset_index(drop=True),
        confirm.reset_index(drop=True),
        {
            "search_end": str(cut),
            "confirm_start": str(confirm["decision_time"].min()),
            "confirm_end": str(confirm["decision_time"].max()),
            "search_rows": int(len(search)),
            "confirm_rows": int(len(confirm)),
            "purge_minutes": int(PURGE_MINUTES),
            "rule": (
                "Chronological reserved-META split. Gate candidates are learned "
                "on SEARCH and must remain profitable on independent CONFIRM. "
                "FINAL TEST is not used."
            ),
        },
    )


def _unique_quantiles(values: pd.Series, qs: list[float]) -> list[float]:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return []
    out = np.quantile(arr, np.asarray(qs, dtype=float))
    return sorted({float(x) for x in out if np.isfinite(x)})


def _gate(
    frame: pd.DataFrame,
    min_confidence: float,
    min_edge_atr: float,
) -> pd.DataFrame:
    return frame.loc[
        (frame["confidence"] >= float(min_confidence))
        & (frame["edge_proxy_atr"] >= float(min_edge_atr))
    ].copy()


def _valid_block(metrics: dict, minimum_trades: int) -> bool:
    pf = metrics["profit_factor"]
    return bool(
        metrics["trades"] >= minimum_trades
        and metrics["net_pips"] > 0.0
        and metrics["mean_net_pips"] is not None
        and metrics["mean_net_pips"] > 0.0
        and pf is not None
        and (pf == float("inf") or pf > MIN_PROFIT_FACTOR)
    )


def _select_gate(
    search: pd.DataFrame,
    confirm: pd.DataFrame,
) -> tuple[dict, list[dict]]:
    confidence_candidates = _unique_quantiles(
        search["confidence"],
        CONFIDENCE_QUANTILES,
    )

    positive_edge = search.loc[
        search["edge_proxy_atr"] > 0.0,
        "edge_proxy_atr",
    ]
    edge_candidates = _unique_quantiles(
        positive_edge,
        EDGE_QUANTILES,
    )

    rows = []

    for confidence in confidence_candidates:
        for edge in edge_candidates:
            search_metrics = _stats(_gate(search, confidence, edge))
            confirm_metrics = _stats(_gate(confirm, confidence, edge))

            valid = (
                _valid_block(search_metrics, MIN_SEARCH_TRADES)
                and _valid_block(confirm_metrics, MIN_CONFIRM_TRADES)
            )

            worst_mean = min(
                float(search_metrics["mean_net_pips"] or -1e9),
                float(confirm_metrics["mean_net_pips"] or -1e9),
            )
            worst_pf = min(
                float(search_metrics["profit_factor"] or 0.0),
                float(confirm_metrics["profit_factor"] or 0.0),
            )
            total_net = float(search_metrics["net_pips"]) + float(confirm_metrics["net_pips"])
            total_dd = (
                float(search_metrics["max_drawdown_pips"])
                + float(confirm_metrics["max_drawdown_pips"])
            )

            # Reward consistency first, not raw in-sample total profit.
            score = (
                100.0 * worst_mean
                + 20.0 * max(0.0, worst_pf - 1.0)
                + 0.03 * total_net
                - 0.02 * total_dd
            )

            rows.append(
                {
                    "minimum_confidence": float(confidence),
                    "minimum_edge_atr": float(edge),
                    "search": search_metrics,
                    "confirm": confirm_metrics,
                    "valid": bool(valid),
                    "score": float(score),
                }
            )

    valid_rows = [row for row in rows if row["valid"]]

    if not valid_rows:
        diagnostic = max(rows, key=lambda row: row["score"]) if rows else {}
        return (
            {
                "trade_enabled": False,
                "minimum_confidence": None,
                "minimum_edge_atr": None,
                "selected": diagnostic,
            },
            rows,
        )

    selected = max(
        valid_rows,
        key=lambda row: (
            row["score"],
            min(row["search"]["mean_net_pips"], row["confirm"]["mean_net_pips"]),
            row["confirm"]["net_pips"],
        ),
    )

    return (
        {
            "trade_enabled": True,
            "minimum_confidence": float(selected["minimum_confidence"]),
            "minimum_edge_atr": float(selected["minimum_edge_atr"]),
            "selected": selected,
        },
        rows,
    )


def build_symbol_policy(symbol: str) -> dict:
    symbol = str(symbol).upper()

    print()
    print("=" * 78)
    print(f"BUILD PRECISION ENTRY POLICY: {symbol}")
    print("=" * 78)

    metadata = _load_json(symbol_model_dir(symbol) / "metadata.json")

    if metadata.get("model_version") != MODEL_VERSION:
        raise RuntimeError(f"{symbol}: direction model version mismatch.")

    meta_rows = _load_meta_rows(symbol, metadata)
    if len(meta_rows) < 5000:
        raise RuntimeError(f"{symbol}: reserved META partition is too small.")

    fallback_cutoff = meta_rows["entry_time"].min() - pd.Timedelta(minutes=5)

    spread_fallback, spread_diag = derive_spread_fallback(
        symbol,
        cutoff_time=fallback_cutoff,
    )

    meta_rows, cost_meta = attach_historical_costs(
        meta_rows,
        symbol,
        spread_fallback_pips=spread_fallback,
    )

    direction_model = DirectionModel(symbol)
    frame = _build_frame(meta_rows, symbol, direction_model)

    search, confirm, split = _split_search_confirm(frame)
    gate, search_rows = _select_gate(search, confirm)

    spread_warning = bool(spread_diag.get("spread_regime_warning", False))
    trade_enabled = bool(gate["trade_enabled"])
    live_enabled = bool(trade_enabled and not spread_warning)

    if not trade_enabled:
        disabled_reason = (
            "No precision gate stayed profitable on both chronological "
            "SEARCH and CONFIRM blocks."
        )
    elif spread_warning:
        disabled_reason = (
            "Historical spread regime is inconsistent with broker snapshot."
        )
    else:
        disabled_reason = None

    policy_dir = symbol_policy_dir(symbol)
    report_dir = symbol_signal_report_dir(symbol)
    policy_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "policy_version": POLICY_VERSION,
        "symbol": symbol,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "architecture": (
            "Precision-first entry gate on top of the causal direction ensemble. "
            "The gate uses only model confidence and expected selected-side move "
            "after currently-known transaction cost. FINAL TEST is not used."
        ),
        "source_model_version": MODEL_VERSION,
        "trade_enabled": trade_enabled,
        "live_enabled": live_enabled,
        "live_disabled_reason": disabled_reason,
        "minimum_confidence": gate["minimum_confidence"],
        "minimum_edge_atr": gate["minimum_edge_atr"],
        "selection": gate["selected"],
        "meta_split": split,
        "spread_fallback": spread_diag,
        "cost_contract": cost_meta,
        "final_test_used": False,
    }

    (policy_dir / "policy.json").write_text(
        json.dumps(_json_safe(payload), indent=2),
        encoding="utf-8",
    )

    pd.DataFrame(
        [
            {
                "minimum_confidence": row["minimum_confidence"],
                "minimum_edge_atr": row["minimum_edge_atr"],
                "valid": row["valid"],
                "score": row["score"],
                "search_trades": row["search"]["trades"],
                "search_net_pips": row["search"]["net_pips"],
                "search_mean_net_pips": row["search"]["mean_net_pips"],
                "search_profit_factor": row["search"]["profit_factor"],
                "confirm_trades": row["confirm"]["trades"],
                "confirm_net_pips": row["confirm"]["net_pips"],
                "confirm_mean_net_pips": row["confirm"]["mean_net_pips"],
                "confirm_profit_factor": row["confirm"]["profit_factor"],
            }
            for row in search_rows
        ]
    ).to_csv(report_dir / "policy_gate_search.csv", index=False)

    frame[
        [
            "decision_time", "entry_time", "exit_time",
            "p_up", "p_down", "confidence",
            "expected_move_atr", "expected_side_atr",
            "ensemble_disagreement",
            "entry_spread_pips", "live_cost_pips",
            "atr_pips", "cost_atr", "edge_proxy_atr",
            "selected_side", "long_net_pips",
            "short_net_pips", "realized_net_pips",
        ]
    ].to_csv(
        report_dir / "meta_policy_rows.csv.gz",
        index=False,
        compression="gzip",
    )

    selected = gate.get("selected") or {}

    print(f"META rows              : {len(frame):,}")
    print(f"Trade enabled          : {trade_enabled}")
    print(f"Live enabled           : {live_enabled}")

    if trade_enabled:
        print(f"Learned confidence gate: {gate['minimum_confidence']:.4f}")
        print(f"Learned net-edge gate  : {gate['minimum_edge_atr']:+.4f} ATR")
        print(
            "SEARCH                 : "
            f"{selected['search']['trades']} trades | "
            f"{selected['search']['net_pips']:+.2f} pips | "
            f"PF {selected['search']['profit_factor']:.3f}"
        )
        print(
            "CONFIRM                : "
            f"{selected['confirm']['trades']} trades | "
            f"{selected['confirm']['net_pips']:+.2f} pips | "
            f"PF {selected['confirm']['profit_factor']:.3f}"
        )
    else:
        print("Gate                   : DISABLED - no robust META edge")

    print(
        f"Spread fallback        : {spread_fallback:.3f} pips "
        f"({spread_diag['fallback_source']})"
    )

    if spread_warning:
        print("WARNING                : spread regime mismatch; live disabled.")

    print(f"Policy                 : {policy_dir / 'policy.json'}")
    return payload


def build_all_policies() -> None:
    ensure_strategy_directories()
    summaries = {symbol: build_symbol_policy(symbol) for symbol in SYMBOLS}

    path = SIGNAL_REPORT_ROOT / "policy_build_summary.json"
    path.write_text(
        json.dumps(
            _json_safe(
                {
                    "policy_version": POLICY_VERSION,
                    "created_utc": datetime.now(timezone.utc).isoformat(),
                    "symbols": summaries,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("PRECISION ENTRY POLICY BUILD COMPLETE")
    print("=" * 78)
    print(f"Summary: {path}")


if __name__ == "__main__":
    build_all_policies()
