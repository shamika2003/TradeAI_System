from __future__ import annotations

import pandas as pd
import pytest

from shared.tradeai_core.model_contract import assert_backtest_is_out_of_sample
from shared.tradeai_core.training_window import apply_supervised_training_cutoff


def test_historical_cutoff_purges_target_horizon_rows_per_symbol():
    # Deliberately include a weekend-sized clock gap.  The guard must purge by
    # bar count, not by "120 minutes", because 24 M5 bars can cross a weekend.
    pre = list(pd.date_range("2026-05-29 20:00", periods=40, freq="5min"))
    post = list(pd.date_range("2026-06-01 00:00", periods=10, freq="5min"))

    rows = []
    for symbol in ("EURUSD", "GBPUSD"):
        for ts in pre + post:
            rows.append({"symbol": symbol, "time": ts, "x": 1.0})

    df = pd.DataFrame(rows)
    filtered, window = apply_supervised_training_cutoff(
        df,
        cutoff_exclusive="2026-06-01",
        label_horizon_bars=24,
    )

    assert window.mode == "historical_cutoff"
    assert window.backtest_safe_from.startswith("2026-06-01")
    assert window.purged_rows_per_symbol == 24

    for symbol in ("EURUSD", "GBPUSD"):
        symbol_df = filtered[filtered["symbol"] == symbol]
        assert len(symbol_df) == 16
        assert symbol_df["time"].max() == pre[15]


def test_cutoff_model_allows_backtest_at_cutoff():
    artifact = {
        "created_utc": "2026-09-17T10:00:00+00:00",
        "training_window": {
            "mode": "historical_cutoff",
            "cutoff_exclusive": "2026-06-01 00:00:00",
            "backtest_safe_from": "2026-06-01 00:00:00",
            "label_horizon_bars": 24,
            "purged_rows_per_symbol": 24,
            "fit_start": "2020-01-01 00:00:00",
            "fit_end": "2026-05-29 21:15:00",
            "rows": 10000,
            "symbol_ranges": {"EURUSD": {"fit_rows": 10000}},
        },
    }

    window = assert_backtest_is_out_of_sample(
        artifact,
        backtest_start="2026-06-01",
        backtest_end="2026-06-30",
    )
    assert window["mode"] == "historical_cutoff"


def test_full_history_model_blocks_old_historical_backtest():
    artifact = {
        "created_utc": "2026-09-17T10:00:00+00:00",
        "training_window": {
            "mode": "all_available_forward",
            "cutoff_exclusive": None,
            "backtest_safe_from": "2026-09-17T10:00:00+00:00",
            "label_horizon_bars": 24,
            "purged_rows_per_symbol": 0,
            "fit_start": "2020-01-01 00:00:00",
            "fit_end": "2026-09-07 07:25:00",
            "rows": 10000,
            "symbol_ranges": {"EURUSD": {"fit_rows": 10000}},
        },
    }

    with pytest.raises(RuntimeError, match="BACKTEST LEAKAGE BLOCKED"):
        assert_backtest_is_out_of_sample(
            artifact,
            backtest_start="2026-06-01",
            backtest_end="2026-06-30",
        )
