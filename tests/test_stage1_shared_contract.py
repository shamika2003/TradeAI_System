# filename: tests/test_stage1_shared_contract.py

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from shared.tradeai_core.feature_engine import FeatureTransformer
from shared.tradeai_core.feature_schema import FEATURE_HASH, FEATURE_NAMES
from shared.tradeai_core.model_contract import (
    create_model_artifact,
    validate_model_artifact,
)


def _bars(start, periods, minutes, base=1.10):
    time = pd.date_range(start=start, periods=periods, freq=f"{minutes}min")
    x = np.arange(periods, dtype=float)
    close = base + x * 0.00001 + np.sin(x / 17.0) * 0.0002
    open_ = close - 0.00001
    high = np.maximum(open_, close) + 0.00008
    low = np.minimum(open_, close) - 0.00008

    return pd.DataFrame(
        {
            "time": time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": 100 + (x % 50),
            "spread": 12.0,
            "real_volume": 0.0,
        }
    )


def test_causal_h1_alignment_and_semantics():
    start = datetime(2026, 1, 1)
    m5 = _bars(start, 1400, 5)
    h1 = _bars(start - timedelta(hours=50), 200, 60)

    engine = FeatureTransformer()
    out = engine.build_multi_timeframe_features(m5, h1)

    assert not out.empty
    assert engine.get_feature_list() == FEATURE_NAMES
    assert len(FEATURE_HASH) == 64

    h1_available = pd.to_datetime(out["h1_source_time"]) + pd.Timedelta(hours=1)
    m5_decision = pd.to_datetime(out["time"]) + pd.Timedelta(minutes=5)
    assert (h1_available <= m5_decision).all()

    m5_sign = pd.Series([-1.0, 1.0, 1.0, -1.0])
    h1_sign = pd.Series([-2.0, 2.0, -2.0, 2.0])
    aligned = engine._alignment(m5_sign, h1_sign).tolist()
    assert aligned == [1, 1, 0, 0]


def test_model_contract_rejects_legacy_artifact():
    class DummyModel:
        pass

    artifact = create_model_artifact(
        models={"EURUSD": DummyModel()},
        symbols=["EURUSD"],
        timeframe="M5",
    )

    validate_model_artifact(
        artifact,
        expected_symbols=["EURUSD"],
        expected_timeframe="M5",
    )

    try:
        validate_model_artifact(
            {"EURUSD": DummyModel()},
            expected_symbols=["EURUSD"],
            expected_timeframe="M5",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("Legacy symbol->model dictionary should be rejected")
