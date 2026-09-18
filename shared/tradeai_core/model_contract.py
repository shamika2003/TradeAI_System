from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from .feature_schema import FEATURE_HASH, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .target_definition import CLASS_MAP, PREDICTION_TYPE, TARGET_VERSION, target_contract
from .decision_policy import validate_decision_policy
from .risk_policy import validate_risk_policy


MODEL_ARTIFACT_VERSION = "tradeai_model_artifact_v5_structural_meta"


def _parse_datetime(value: Any, field_name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise RuntimeError(f"Model training window missing {field_name}")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise RuntimeError(f"Invalid {field_name}: {value!r}") from exc
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _normalize_training_window(training_window: dict | None, created_utc: str) -> dict:
    if not isinstance(training_window, dict):
        raise RuntimeError(
            "Training provenance is required for the opportunity model artifact. "
            "Retrain with the Stage 1 backtest-integrity code."
        )

    normalized = dict(training_window)
    mode = str(normalized.get("mode", "")).strip()
    if mode not in {"historical_cutoff", "all_available_forward"}:
        raise RuntimeError(f"Invalid training window mode: {mode!r}")

    try:
        horizon = int(normalized["label_horizon_bars"])
        rows = int(normalized["rows"])
    except Exception as exc:
        raise RuntimeError("Training window is missing numeric provenance") from exc

    if horizon <= 0:
        raise RuntimeError("Training window label_horizon_bars must be positive")
    if rows <= 0:
        raise RuntimeError("Training window rows must be positive")

    _parse_datetime(normalized.get("fit_start"), "fit_start")
    _parse_datetime(normalized.get("fit_end"), "fit_end")

    cutoff = normalized.get("cutoff_exclusive")
    if mode == "historical_cutoff":
        _parse_datetime(cutoff, "cutoff_exclusive")
        safe_from = normalized.get("backtest_safe_from") or cutoff
    else:
        # A model fitted on all available history is a forward model.  Historical
        # dates before artifact creation are in-sample and must not be presented
        # as an out-of-sample runtime backtest.
        normalized["cutoff_exclusive"] = None
        safe_from = normalized.get("backtest_safe_from") or created_utc

    _parse_datetime(safe_from, "backtest_safe_from")
    normalized["backtest_safe_from"] = str(safe_from)
    normalized["label_horizon_bars"] = horizon
    normalized["rows"] = rows
    normalized["purged_rows_per_symbol"] = int(
        normalized.get("purged_rows_per_symbol", 0) or 0
    )

    symbol_ranges = normalized.get("symbol_ranges")
    if not isinstance(symbol_ranges, dict) or not symbol_ranges:
        raise RuntimeError("Training window symbol_ranges are missing")

    return normalized


def create_model_artifact(
    *,
    models: dict,
    symbols: Iterable[str],
    timeframe: str,
    training_metadata: dict | None = None,
    metrics: dict | None = None,
    model_params: dict | None = None,
    training_window: dict | None = None,
) -> dict:
    symbols = list(symbols)
    created_utc = datetime.now(timezone.utc).isoformat()
    normalized_window = _normalize_training_window(training_window, created_utc)

    return {
        "artifact_type": "TradeAIModelArtifact",
        "artifact_version": MODEL_ARTIFACT_VERSION,
        "created_utc": created_utc,
        "contract": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_hash": FEATURE_HASH,
            "feature_names": list(FEATURE_NAMES),
            "target_version": TARGET_VERSION,
            "prediction_type": PREDICTION_TYPE,
            "class_map": dict(CLASS_MAP),
            "timeframe": str(timeframe),
            "symbols": symbols,
            "target_contract": target_contract(),
        },
        "training_window": normalized_window,
        "training_metadata": training_metadata or {},
        "metrics": metrics or {},
        "model_params": model_params or {},
        "models": models,
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_training_window(artifact: dict) -> dict:
    _require(isinstance(artifact, dict), "Model artifact is not a dictionary")
    created_utc = str(artifact.get("created_utc", "") or "")
    _parse_datetime(created_utc, "created_utc")
    window = _normalize_training_window(artifact.get("training_window"), created_utc)
    return dict(window)


def validate_model_artifact(
    artifact: Any,
    *,
    expected_symbols: Iterable[str] | None = None,
    expected_timeframe: str | None = None,
    expected_target_version: str = TARGET_VERSION,
) -> dict:
    _require(isinstance(artifact, dict), "Model artifact is not a dictionary")
    _require(
        artifact.get("artifact_type") == "TradeAIModelArtifact",
        "Legacy/unknown model artifact rejected; retraining is required",
    )
    _require(
        artifact.get("artifact_version") == MODEL_ARTIFACT_VERSION,
        "Unsupported model artifact version; retraining is required",
    )

    contract = artifact.get("contract")
    _require(isinstance(contract, dict), "Model artifact contract is missing")
    _require(contract.get("feature_schema_version") == FEATURE_SCHEMA_VERSION, "Feature schema version mismatch")
    _require(contract.get("feature_hash") == FEATURE_HASH, "Feature hash mismatch: training/live semantics differ")
    _require(contract.get("feature_names") == list(FEATURE_NAMES), "Feature list/order mismatch")
    _require(contract.get("target_version") == expected_target_version, "Target definition mismatch")
    _require(contract.get("prediction_type") == PREDICTION_TYPE, "Prediction type mismatch")
    _require(contract.get("class_map") == dict(CLASS_MAP), "Class mapping mismatch")

    if expected_timeframe is not None:
        _require(contract.get("timeframe") == str(expected_timeframe), f"Timeframe mismatch: expected {expected_timeframe}")

    contract_symbols = list(contract.get("symbols") or [])
    models = artifact.get("models")
    _require(isinstance(models, dict), "Model dictionary is missing")
    _require(bool(models), "Model dictionary is empty")
    _require(set(models.keys()) == set(contract_symbols), "Model keys do not match contract symbols")

    if expected_symbols is not None:
        expected = list(expected_symbols)
        _require(set(contract_symbols) == set(expected), f"Symbol contract mismatch: expected {expected}, got {contract_symbols}")

    window = require_training_window(artifact)
    _require(
        set((window.get("symbol_ranges") or {}).keys()) == set(contract_symbols),
        "Training provenance symbols do not match model contract symbols",
    )

    return artifact


def assert_backtest_is_out_of_sample(
    artifact: dict,
    *,
    backtest_start: str,
    backtest_end: str | None = None,
) -> dict:
    """Fail closed when a runtime backtest overlaps model training information."""

    window = require_training_window(artifact)
    start = _parse_datetime(backtest_start, "backtest_start")
    safe_from = _parse_datetime(window["backtest_safe_from"], "backtest_safe_from")

    if start < safe_from:
        mode = window.get("mode")
        cutoff = window.get("cutoff_exclusive")
        if mode == "all_available_forward":
            raise RuntimeError(
                "BACKTEST LEAKAGE BLOCKED: this model was trained on all available "
                f"history and is only safe for forward data from {window['backtest_safe_from']}. "
                "Train a dedicated historical artifact with "
                "TRADEAI_TRAINING_CUTOFF_DATE set to the backtest start date."
            )
        raise RuntimeError(
            "BACKTEST LEAKAGE BLOCKED: requested backtest starts at "
            f"{backtest_start}, but this model is safe only from {window['backtest_safe_from']} "
            f"(training cutoff {cutoff})."
        )

    if backtest_end:
        end = _parse_datetime(backtest_end, "backtest_end")
        if end < start:
            raise RuntimeError("Backtest end date is before start date")

    return window


def attach_decision_policy(artifact: dict, policy: dict, validation: dict | None = None) -> dict:
    """Attach a calibrated production decision policy without replacing models."""
    if not isinstance(artifact, dict):
        raise RuntimeError("Model artifact is not a dictionary")
    symbols = list((artifact.get("contract") or {}).get("symbols") or [])
    normalized = validate_decision_policy(policy, symbols)
    artifact["decision_policy"] = normalized
    if validation is not None:
        artifact["production_validation"] = validation
    return artifact


def require_decision_policy(artifact: dict) -> dict:
    symbols = list((artifact.get("contract") or {}).get("symbols") or [])
    return validate_decision_policy(artifact.get("decision_policy"), symbols)


def attach_risk_policy(artifact: dict, policy: dict, validation: dict | None = None) -> dict:
    """Attach a calibrated production risk policy without replacing models."""
    if not isinstance(artifact, dict):
        raise RuntimeError("Model artifact is not a dictionary")
    normalized = validate_risk_policy(policy)
    artifact["risk_policy"] = normalized
    if validation is not None:
        artifact["risk_validation"] = validation
    return artifact


def require_risk_policy(artifact: dict) -> dict:
    return validate_risk_policy(artifact.get("risk_policy"))
