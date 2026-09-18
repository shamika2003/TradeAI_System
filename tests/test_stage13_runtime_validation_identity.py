from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "Trade_Bot_Training"
TRADEAI = ROOT / "TradeAI"

for path in (ROOT, TRAINING, TRADEAI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_stage5():
    import stage5_risk_validate as module
    return module


def _load_final_validation():
    spec = importlib.util.spec_from_file_location(
        "tradeai_stage6_final_validation",
        TRAINING / "final_clean_validation.py",
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage11_separates_policy_selection_score_calibration_and_final_test():
    import config_model as cfg

    assert cfg.QUALIFICATION_TRAIN_FRACTION == pytest.approx(0.64)
    assert cfg.QUALIFICATION_SCORE_CAL_END_FRACTION == pytest.approx(0.70)
    assert cfg.POLICY_SELECTION_END_FRACTION == pytest.approx(0.90)
    assert cfg.DEPLOYMENT_CALIBRATION_END_FRACTION == pytest.approx(0.95)
    assert cfg.QUALIFICATION_TRAIN_FRACTION < cfg.QUALIFICATION_SCORE_CAL_END_FRACTION
    assert cfg.QUALIFICATION_SCORE_CAL_END_FRACTION < cfg.POLICY_SELECTION_END_FRACTION
    assert cfg.POLICY_SELECTION_END_FRACTION < cfg.DEPLOYMENT_CALIBRATION_END_FRACTION < 1.0


def test_stage5_promotes_exact_validated_model_objects():
    s5 = _load_stage5()

    eur_model = object()
    gbp_model = object()
    artifact = {
        "contract": {"symbols": ["EURUSD", "GBPUSD"]},
        "models": {"EURUSD": object(), "GBPUSD": object()},
        "training_metadata": {"source": "unit-test"},
    }

    promoted = s5._with_exact_deployment_models(
        artifact,
        {"EURUSD": eur_model, "GBPUSD": gbp_model},
        {"split_method": "unit-test"},
    )

    assert promoted["models"]["EURUSD"] is eur_model
    assert promoted["models"]["GBPUSD"] is gbp_model
    assert promoted["training_metadata"]["deployment_validation"]["exact_runtime_model"] is True
    assert artifact["models"]["EURUSD"] is not eur_model


def test_final_pipeline_rejects_non_exact_model_promotion():
    module = _load_final_validation()

    bad = {
        "contract": {"symbols": ["EURUSD"]},
        "models": {"EURUSD": object()},
        "training_metadata": {},
    }
    with pytest.raises(RuntimeError, match="exact validated models"):
        module._require_exact_deployment_model(bad)

    good = {
        "contract": {"symbols": ["EURUSD"]},
        "models": {"EURUSD": object()},
        "training_metadata": {
            "deployment_validation": {
                "exact_runtime_model": True,
                "split_method": "test",
            }
        },
    }
    result = module._require_exact_deployment_model(good)
    assert result["split_method"] == "test"


def _write_replay_csv(path: Path):
    rows = []

    # Two warmup bars before June 1, then June 1 active bars.
    for ts in (
        "2026-05-31 23:50:00",
        "2026-05-31 23:55:00",
        "2026-06-01 00:00:00",
        "2026-06-01 00:05:00",
        "2026-06-30 23:50:00",
        "2026-06-30 23:55:00",
        "2026-07-01 00:00:00",
    ):
        rows.append(
            {
                "time": ts,
                "symbol": "EURUSD",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
            }
        )

    # A second symbol intentionally has a missing 00:00 bar. Chronological
    # replay must not advance it together with EURUSD.
    for ts in (
        "2026-05-31 23:50:00",
        "2026-05-31 23:55:00",
        "2026-06-01 00:05:00",
        "2026-06-30 23:55:00",
        "2026-07-01 00:00:00",
    ):
        rows.append(
            {
                "time": ts,
                "symbol": "GBPUSD",
                "open": 1.0,
                "high": 1.0,
                "low": 1.0,
                "close": 1.0,
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def test_replay_uses_prestart_warmup_and_includes_full_end_date(tmp_path, monkeypatch):
    from market import replay_engine as replay_module

    csv_path = tmp_path / "replay.csv"
    _write_replay_csv(csv_path)

    monkeypatch.setattr(replay_module, "BACKTEST_START_DATE", "2026-06-01")
    monkeypatch.setattr(replay_module, "BACKTEST_END_DATE", "2026-06-30")

    replay = replay_module.ReplayEngine(
        csv_path,
        ["EURUSD", "GBPUSD"],
        history_size=2,
    )

    first = replay.next_market_snapshot()
    assert list(first.keys()) == ["EURUSD"]
    assert replay.get_current_time() == pd.Timestamp("2026-06-01 00:00:00")
    history = first["EURUSD"]
    assert list(history["time"]) == [
        pd.Timestamp("2026-05-31 23:50:00"),
        pd.Timestamp("2026-05-31 23:55:00"),
    ]

    second = replay.next_market_snapshot()
    assert set(second.keys()) == {"EURUSD", "GBPUSD"}
    assert replay.get_current_time() == pd.Timestamp("2026-06-01 00:05:00")

    seen = [replay.get_current_time()]
    while True:
        market = replay.next_market_snapshot()
        if market is None:
            break
        seen.append(replay.get_current_time())

    assert pd.Timestamp("2026-06-30 23:55:00") in seen
    assert pd.Timestamp("2026-07-01 00:00:00") not in seen
