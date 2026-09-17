from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TrainingWindow:
    mode: str
    cutoff_exclusive: str | None
    backtest_safe_from: str | None
    label_horizon_bars: int
    purged_rows_per_symbol: int
    fit_start: str | None
    fit_end: str | None
    rows: int
    symbol_ranges: dict[str, dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "cutoff_exclusive": self.cutoff_exclusive,
            "backtest_safe_from": self.backtest_safe_from,
            "label_horizon_bars": self.label_horizon_bars,
            "purged_rows_per_symbol": self.purged_rows_per_symbol,
            "fit_start": self.fit_start,
            "fit_end": self.fit_end,
            "rows": self.rows,
            "symbol_ranges": self.symbol_ranges,
        }


def normalize_cutoff(value: str | None) -> pd.Timestamp | None:
    text = str(value or "").strip()
    if not text:
        return None
    ts = pd.Timestamp(text)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts


def _timestamp_text(value: Any) -> str | None:
    if value is None:
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return ts.isoformat(sep=" ")


def apply_supervised_training_cutoff(
    df: pd.DataFrame,
    *,
    cutoff_exclusive: str | None,
    label_horizon_bars: int,
    symbol_column: str = "symbol",
    time_column: str = "time",
) -> tuple[pd.DataFrame, TrainingWindow]:
    """Return the training frame with a leakage-safe historical cutoff.

    Targets in TradeAI use future bars.  A simple timestamp filter such as
    ``time < cutoff`` is therefore not sufficient: the last rows before the
    cutoff can still have labels that were calculated from bars after the
    cutoff.  This function removes exactly ``label_horizon_bars`` rows from
    the tail of every symbol before an explicit cutoff.

    With no cutoff the full available supervised dataset is retained.  Such
    an artifact is intended for forward/demo/live use; historical runtime
    backtests are blocked by the model provenance guard until a dedicated
    cutoff artifact is trained.
    """

    if df is None or df.empty:
        raise RuntimeError("Training dataset is empty")
    if symbol_column not in df.columns or time_column not in df.columns:
        raise RuntimeError(
            f"Training dataset must contain {symbol_column!r} and {time_column!r}"
        )

    horizon = int(label_horizon_bars)
    if horizon < 1:
        raise ValueError("label_horizon_bars must be positive")

    data = df.copy()
    data[time_column] = pd.to_datetime(data[time_column], errors="coerce")
    data = data.dropna(subset=[symbol_column, time_column])
    data = data.sort_values([symbol_column, time_column]).reset_index(drop=True)

    cutoff = normalize_cutoff(cutoff_exclusive)
    selected_groups: list[pd.DataFrame] = []
    symbol_ranges: dict[str, dict[str, Any]] = {}

    for symbol, group in data.groupby(symbol_column, sort=True):
        group = group.sort_values(time_column).reset_index(drop=True)
        original_rows = len(group)

        if cutoff is not None:
            before_cutoff = group[group[time_column] < cutoff].reset_index(drop=True)
            if len(before_cutoff) <= horizon:
                raise RuntimeError(
                    f"Not enough pre-cutoff rows for {symbol}: "
                    f"rows={len(before_cutoff)}, purge={horizon}"
                )
            group = before_cutoff.iloc[:-horizon].copy()

        if group.empty:
            raise RuntimeError(f"Training window is empty for {symbol}")

        selected_groups.append(group)
        symbol_ranges[str(symbol)] = {
            "source_rows": int(original_rows),
            "fit_rows": int(len(group)),
            "start": _timestamp_text(group[time_column].min()),
            "end": _timestamp_text(group[time_column].max()),
        }

    filtered = (
        pd.concat(selected_groups, ignore_index=True)
        .sort_values([time_column, symbol_column])
        .reset_index(drop=True)
    )

    fit_start = _timestamp_text(filtered[time_column].min())
    fit_end = _timestamp_text(filtered[time_column].max())

    window = TrainingWindow(
        mode="historical_cutoff" if cutoff is not None else "all_available_forward",
        cutoff_exclusive=_timestamp_text(cutoff) if cutoff is not None else None,
        backtest_safe_from=_timestamp_text(cutoff) if cutoff is not None else None,
        label_horizon_bars=horizon,
        purged_rows_per_symbol=horizon if cutoff is not None else 0,
        fit_start=fit_start,
        fit_end=fit_end,
        rows=int(len(filtered)),
        symbol_ranges=symbol_ranges,
    )
    return filtered, window
