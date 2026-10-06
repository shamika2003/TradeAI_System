from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

from Trade_Bot_Training.direction_config import (
    CALIBRATION_FRACTION,
    CLASSIFIER_ENSEMBLE,
    CONFIDENCE_LEVELS,
    EARLY_STOPPING_ROUNDS,
    HORIZON_BARS,
    HORIZON_MINUTES,
    META_FRACTION,
    MODEL_VERSION,
    N_JOBS,
    PURGE_MINUTES,
    RANDOM_STATE,
    REGRESSION_TARGET_CLIP_ATR,
    REGRESSOR_PARAMS,
    SYMBOLS,
    TEST_FRACTION,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    XGB_DEVICE,
    ensure_directories,
    processed_path,
    symbol_model_dir,
    symbol_report_dir,
)
from Trade_Bot_Training.feature_engine import (
    model_feature_columns,
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(k): _json_safe(v)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(v)
            for v in value
        ]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    return value


def _classification_metrics(
    y_true: np.ndarray,
    p_up: np.ndarray,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    p_up = np.clip(
        np.asarray(
            p_up,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )
    y_pred = (
        p_up >= 0.5
    ).astype(int)

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=[0, 1],
            zero_division=0,
        )
    )

    result = {
        "rows": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "log_loss": float(
            log_loss(
                y_true,
                p_up,
                labels=[0, 1],
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                p_up,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                y_pred,
                labels=[0, 1],
            ).tolist()
        ),
        "DOWN": {
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": float(f1[0]),
            "support": int(support[0]),
        },
        "UP": {
            "precision": float(precision[1]),
            "recall": float(recall[1]),
            "f1": float(f1[1]),
            "support": int(support[1]),
        },
    }

    try:
        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                p_up,
            )
        )
    except ValueError:
        result["roc_auc"] = None

    return result


def _regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    y_true = np.asarray(
        y_true,
        dtype=float,
    )
    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )
    error = (
        y_pred - y_true
    )

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):
        corr = float(
            np.corrcoef(
                y_true,
                y_pred,
            )[0, 1]
        )
    else:
        corr = None

    return {
        "rows": int(len(y_true)),
        "mae_atr": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "rmse_atr": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "mean_error_atr": float(
            np.mean(error)
        ),
        "correlation": corr,
        "sign_accuracy": float(
            np.mean(
                (y_pred >= 0.0)
                == (y_true >= 0.0)
            )
        ),
        "predicted_move_atr_mean": float(
            np.mean(y_pred)
        ),
        "actual_move_atr_mean": float(
            np.mean(y_true)
        ),
    }


def _confidence_table(
    y_true: np.ndarray,
    p_up: np.ndarray,
    realized_move_atr: np.ndarray,
) -> list[dict]:
    y_true = np.asarray(
        y_true,
        dtype=int,
    )
    p_up = np.asarray(
        p_up,
        dtype=float,
    )
    realized_move_atr = np.asarray(
        realized_move_atr,
        dtype=float,
    )

    direction = (
        p_up >= 0.5
    ).astype(int)

    confidence = np.maximum(
        p_up,
        1.0 - p_up,
    )

    signed_realized = np.where(
        direction == 1,
        realized_move_atr,
        -realized_move_atr,
    )

    rows = []

    for threshold in CONFIDENCE_LEVELS:
        mask = (
            confidence
            >= threshold
        )
        count = int(
            mask.sum()
        )

        rows.append(
            {
                "confidence_at_least": float(
                    threshold
                ),
                "rows": count,
                "coverage": float(
                    count / len(y_true)
                    if len(y_true)
                    else 0.0
                ),
                "accuracy": (
                    float(
                        accuracy_score(
                            y_true[mask],
                            direction[mask],
                        )
                    )
                    if count
                    else None
                ),
                "mean_signed_realized_move_atr": (
                    float(
                        np.mean(
                            signed_realized[
                                mask
                            ]
                        )
                    )
                    if count
                    else None
                ),
            }
        )

    return rows


def _time_split(
    df: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    fractions = (
        TRAIN_FRACTION
        + VALIDATION_FRACTION
        + CALIBRATION_FRACTION
        + META_FRACTION
        + TEST_FRACTION
    )

    if abs(
        fractions - 1.0
    ) > 1e-9:
        raise ValueError(
            "Temporal split fractions "
            "must sum to 1."
        )

    times = pd.to_datetime(
        df["decision_time"],
        utc=True,
        errors="coerce",
    )

    unique_times = (
        times
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    if len(unique_times) < 5000:
        raise RuntimeError(
            "Not enough timestamps "
            "for production temporal split."
        )

    n = len(unique_times)

    boundaries = []
    cumulative = 0.0

    for fraction in (
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        CALIBRATION_FRACTION,
        META_FRACTION,
    ):
        cumulative += fraction
        index = min(
            n - 2,
            max(
                0,
                int(
                    n
                    * cumulative
                )
                - 1,
            ),
        )
        boundaries.append(
            pd.Timestamp(
                unique_times.iloc[
                    index
                ]
            )
        )

    (
        train_end,
        validation_end,
        calibration_end,
        meta_end,
    ) = boundaries

    purge = pd.Timedelta(
        minutes=PURGE_MINUTES
    )

    train = df.loc[
        times <= train_end
    ].copy()

    validation = df.loc[
        (
            times
            > train_end
            + purge
        )
        & (
            times
            <= validation_end
        )
    ].copy()

    calibration = df.loc[
        (
            times
            > validation_end
            + purge
        )
        & (
            times
            <= calibration_end
        )
    ].copy()

    meta = df.loc[
        (
            times
            > calibration_end
            + purge
        )
        & (
            times
            <= meta_end
        )
    ].copy()

    test = df.loc[
        times
        > meta_end
        + purge
    ].copy()

    parts = [
        train,
        validation,
        calibration,
        meta,
        test,
    ]

    parts = [
        part
        .sort_values(
            "decision_time"
        )
        .reset_index(drop=True)
        for part in parts
    ]

    if min(
        len(part)
        for part in parts
    ) == 0:
        raise RuntimeError(
            "Temporal split produced "
            "an empty partition."
        )

    metadata = {
        "train_end": str(
            train_end
        ),
        "validation_end": str(
            validation_end
        ),
        "calibration_end": str(
            calibration_end
        ),
        "meta_end": str(
            meta_end
        ),
        "test_start": str(
            test[
                "decision_time"
            ].min()
        ),
        "test_end": str(
            test[
                "decision_time"
            ].max()
        ),
        "purge_minutes": (
            PURGE_MINUTES
        ),
        "train_rows": int(
            len(train)
        ),
        "validation_rows": int(
            len(validation)
        ),
        "calibration_rows": int(
            len(calibration)
        ),
        "meta_rows": int(
            len(meta)
        ),
        "test_rows": int(
            len(test)
        ),
        "rule": (
            "Chronological 60/15/10/5/10 split: "
            "TRAIN / VALIDATION / CALIBRATION / "
            "META / FINAL TEST, with a 30-minute "
            "target-horizon purge between partitions."
        ),
    }

    return (
        train,
        validation,
        calibration,
        meta,
        test,
        metadata,
    )


def _xgb_common(
    random_state: int,
) -> dict:
    params = {
        "tree_method": "hist",
        "random_state": int(
            random_state
        ),
        "n_jobs": N_JOBS,
    }

    if XGB_DEVICE not in (
        "",
        "cpu",
    ):
        params[
            "device"
        ] = XGB_DEVICE

    return params


def _member_params(
    spec: dict,
    *,
    early_stopping: bool,
    n_estimators: int | None = None,
) -> dict:
    spec = dict(spec)
    spec.pop(
        "name",
        None,
    )
    offset = int(
        spec.pop(
            "random_state_offset",
            0,
        )
    )

    params = {
        **_xgb_common(
            RANDOM_STATE
            + offset
        ),
        "objective": (
            "binary:logistic"
        ),
        "eval_metric": (
            "logloss"
        ),
        **spec,
    }

    if n_estimators is not None:
        params[
            "n_estimators"
        ] = int(
            n_estimators
        )

    if early_stopping:
        params[
            "early_stopping_rounds"
        ] = EARLY_STOPPING_ROUNDS

    return params


def _regressor_params(
    *,
    early_stopping: bool,
    n_estimators: int | None = None,
) -> dict:
    params = {
        **_xgb_common(
            RANDOM_STATE
            + 101
        ),
        "objective": (
            "reg:pseudohubererror"
        ),
        "eval_metric": "mae",
        **REGRESSOR_PARAMS,
    }

    if n_estimators is not None:
        params[
            "n_estimators"
        ] = int(
            n_estimators
        )

    if early_stopping:
        params[
            "early_stopping_rounds"
        ] = EARLY_STOPPING_ROUNDS

    return params


def _best_tree_count(
    model,
) -> int:
    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    if best_iteration is None:
        return int(
            model.get_params()[
                "n_estimators"
            ]
        )

    return max(
        1,
        int(
            best_iteration
        )
        + 1,
    )


def _probability_logit(
    probability: np.ndarray,
) -> np.ndarray:
    p = np.clip(
        np.asarray(
            probability,
            dtype=float,
        ),
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        p
        / (
            1.0 - p
        )
    ).reshape(
        -1,
        1,
    )


def _fit_platt_calibrator(
    raw_p_up: np.ndarray,
    y_true: np.ndarray,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        _probability_logit(
            raw_p_up
        ),
        np.asarray(
            y_true,
            dtype=int,
        ),
    )

    return calibrator


def _apply_calibrator(
    calibrator: LogisticRegression,
    raw_p_up: np.ndarray,
) -> np.ndarray:
    return calibrator.predict_proba(
        _probability_logit(
            raw_p_up
        )
    )[:, 1]


def _ensemble_probability(
    models: list[XGBClassifier],
    X: pd.DataFrame,
) -> np.ndarray:
    matrix = np.column_stack(
        [
            model.predict_proba(
                X
            )[:, 1]
            for model in models
        ]
    )

    return matrix.mean(
        axis=1
    )


def train_symbol(
    symbol: str,
) -> dict:
    symbol = str(
        symbol
    ).upper()

    path = processed_path(
        symbol
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed dataset: {path}\n"
            "Run dataset build first."
        )

    print()
    print("=" * 78)
    print(
        "TRAINING FINAL DIRECTION ENSEMBLE: "
        f"{symbol}"
    )
    print("=" * 78)

    df = pd.read_csv(
        path,
        compression="gzip",
    )

    df[
        "decision_time"
    ] = pd.to_datetime(
        df[
            "decision_time"
        ],
        utc=True,
        errors="coerce",
    )

    feature_columns = (
        model_feature_columns(
            df
        )
    )

    required = {
        "target_up",
        "future_move_atr",
        "m5_atr_pct_14",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise RuntimeError(
            "Processed dataset is missing: "
            + ", ".join(
                sorted(missing)
            )
        )

    (
        train,
        validation,
        calibration,
        meta,
        test,
        split_metadata,
    ) = _time_split(
        df
    )

    X_train = train[
        feature_columns
    ].astype(
        "float32"
    )

    y_train = train[
        "target_up"
    ].astype(
        "int8"
    )

    X_validation = validation[
        feature_columns
    ].astype(
        "float32"
    )

    y_validation = validation[
        "target_up"
    ].astype(
        "int8"
    )

    fit_rows = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    X_fit = fit_rows[
        feature_columns
    ].astype(
        "float32"
    )

    y_fit = fit_rows[
        "target_up"
    ].astype(
        "int8"
    )

    tuning_models = []
    member_metadata = []

    for index, spec in enumerate(
        CLASSIFIER_ENSEMBLE
    ):
        name = str(
            spec[
                "name"
            ]
        )

        tuning = XGBClassifier(
            **_member_params(
                spec,
                early_stopping=True,
            )
        )

        tuning.fit(
            X_train,
            y_train,
            eval_set=[
                (
                    X_validation,
                    y_validation,
                )
            ],
            verbose=False,
        )

        trees = _best_tree_count(
            tuning
        )

        tuning_models.append(
            tuning
        )

        member_metadata.append(
            {
                "index": index,
                "name": name,
                "selected_trees": (
                    trees
                ),
            }
        )

        print(
            f"  {name:<12} trees={trees:,}"
        )

    validation_raw = (
        _ensemble_probability(
            tuning_models,
            X_validation,
        )
    )

    validation_metrics = (
        _classification_metrics(
            y_validation.to_numpy(),
            validation_raw,
        )
    )

    final_models = []

    for spec, member_info in zip(
        CLASSIFIER_ENSEMBLE,
        member_metadata,
    ):
        model = XGBClassifier(
            **_member_params(
                spec,
                early_stopping=False,
                n_estimators=(
                    member_info[
                        "selected_trees"
                    ]
                ),
            )
        )

        model.fit(
            X_fit,
            y_fit,
            verbose=False,
        )

        final_models.append(
            model
        )

    X_calibration = calibration[
        feature_columns
    ].astype(
        "float32"
    )

    y_calibration = calibration[
        "target_up"
    ].astype(
        "int8"
    )

    raw_p_calibration = (
        _ensemble_probability(
            final_models,
            X_calibration,
        )
    )

    calibrator = (
        _fit_platt_calibrator(
            raw_p_calibration,
            y_calibration.to_numpy(),
        )
    )

    # Companion signed-move model remains auxiliary.
    y_move_train = (
        train[
            "future_move_atr"
        ]
        .astype(
            "float32"
        )
        .clip(
            -REGRESSION_TARGET_CLIP_ATR,
            REGRESSION_TARGET_CLIP_ATR,
        )
    )

    y_move_validation = (
        validation[
            "future_move_atr"
        ]
        .astype(
            "float32"
        )
        .clip(
            -REGRESSION_TARGET_CLIP_ATR,
            REGRESSION_TARGET_CLIP_ATR,
        )
    )

    tuning_regressor = (
        XGBRegressor(
            **_regressor_params(
                early_stopping=True
            )
        )
    )

    tuning_regressor.fit(
        X_train,
        y_move_train,
        eval_set=[
            (
                X_validation,
                y_move_validation,
            )
        ],
        verbose=False,
    )

    regressor_trees = (
        _best_tree_count(
            tuning_regressor
        )
    )

    y_move_fit = (
        fit_rows[
            "future_move_atr"
        ]
        .astype(
            "float32"
        )
        .clip(
            -REGRESSION_TARGET_CLIP_ATR,
            REGRESSION_TARGET_CLIP_ATR,
        )
    )

    regressor = XGBRegressor(
        **_regressor_params(
            early_stopping=False,
            n_estimators=(
                regressor_trees
            ),
        )
    )

    regressor.fit(
        X_fit,
        y_move_fit,
        verbose=False,
    )

    X_test = test[
        feature_columns
    ].astype(
        "float32"
    )

    y_test = test[
        "target_up"
    ].astype(
        "int8"
    )

    raw_p_test = (
        _ensemble_probability(
            final_models,
            X_test,
        )
    )

    p_test = (
        _apply_calibrator(
            calibrator,
            raw_p_test,
        )
    )

    move_test_pred = (
        regressor.predict(
            X_test
        ).astype(float)
    )

    test_metrics_raw = (
        _classification_metrics(
            y_test.to_numpy(),
            raw_p_test,
        )
    )

    test_metrics = (
        _classification_metrics(
            y_test.to_numpy(),
            p_test,
        )
    )

    move_test_metrics = (
        _regression_metrics(
            test[
                "future_move_atr"
            ].to_numpy(
                dtype=float
            ),
            move_test_pred,
        )
    )

    confidence_table = (
        _confidence_table(
            y_test.to_numpy(),
            p_test,
            test[
                "future_move_atr"
            ].to_numpy(
                dtype=float
            ),
        )
    )

    majority_class = int(
        y_train.mean()
        >= 0.5
    )

    majority_accuracy = float(
        np.mean(
            y_test.to_numpy()
            == majority_class
        )
    )

    persistence = (
        test[
            "m5_body_pct"
        ].to_numpy(
            dtype=float
        )
        >= 0.0
    ).astype(int)

    persistence_accuracy = float(
        np.mean(
            persistence
            == y_test.to_numpy()
        )
    )

    model_dir = (
        symbol_model_dir(
            symbol
        )
    )

    report_dir = (
        symbol_report_dir(
            symbol
        )
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Remove stale ensemble members from an older run.
    for stale in model_dir.glob(
        "classifier_*.json"
    ):
        stale.unlink(
            missing_ok=True
        )

    member_files = []

    for index, (
        model,
        member_info,
    ) in enumerate(
        zip(
            final_models,
            member_metadata,
        )
    ):
        filename = (
            f"classifier_{index}.json"
        )
        path_out = (
            model_dir
            / filename
        )

        model.save_model(
            path_out
        )

        member_info[
            "file"
        ] = filename

        member_info[
            "weight"
        ] = (
            1.0
            / len(
                final_models
            )
        )

        member_files.append(
            filename
        )

    # Compatibility alias for components that only check model.json.
    shutil.copyfile(
        model_dir
        / member_files[0],
        model_dir
        / "model.json",
    )

    regressor.save_model(
        model_dir
        / "move_model.json"
    )

    joblib.dump(
        calibrator,
        model_dir
        / "calibrator.joblib",
    )

    ensemble_payload = {
        "aggregation": (
            "equal_weight_probability_mean"
        ),
        "members": (
            member_metadata
        ),
    }

    (
        model_dir
        / "ensemble.json"
    ).write_text(
        json.dumps(
            ensemble_payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    average_importance = np.mean(
        np.vstack(
            [
                model.feature_importances_
                for model in final_models
            ]
        ),
        axis=0,
    )

    importance_rows = [
        {
            "feature": feature,
            "importance": float(
                importance
            ),
        }
        for feature, importance
        in zip(
            feature_columns,
            average_importance,
        )
    ]

    importance_rows.sort(
        key=lambda row: (
            row[
                "importance"
            ]
        ),
        reverse=True,
    )

    metadata = {
        "model_version": (
            MODEL_VERSION
        ),
        "symbol": symbol,
        "trained_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "architecture": (
            "three-member regularized XGBoost "
            "probability ensemble + Platt calibration"
        ),
        "prediction_horizon_bars": (
            HORIZON_BARS
        ),
        "prediction_horizon_minutes": (
            HORIZON_MINUTES
        ),
        "target_contract": (
            "Direction of executable move from next "
            "contiguous M5 open to close 30 minutes later."
        ),
        "feature_columns": (
            feature_columns
        ),
        "feature_count": (
            len(
                feature_columns
            )
        ),
        "split": (
            split_metadata
        ),
        "ensemble": (
            ensemble_payload
        ),
        "validation_metrics_before_refit": (
            validation_metrics
        ),
        "calibration": {
            "method": (
                "Platt scaling on ensemble log-odds"
            ),
            "rows": int(
                len(
                    calibration
                )
            ),
            "strictly_after_model_fit": True,
        },
        "meta_partition": {
            "rows": int(
                len(meta)
            ),
            "used_by_direction_model": False,
            "purpose": (
                "Reserved for cost-aware meta-label "
                "trade selection."
            ),
        },
        "test_metrics_raw": (
            test_metrics_raw
        ),
        "test_metrics_calibrated": (
            test_metrics
        ),
        "move_model": {
            "selected_trees": (
                regressor_trees
            ),
            "test_metrics": (
                move_test_metrics
            ),
        },
        "baselines": {
            "majority_test_accuracy": (
                majority_accuracy
            ),
            "persistence_test_accuracy": (
                persistence_accuracy
            ),
        },
        "confidence_table": (
            confidence_table
        ),
        "top_feature_importance": (
            importance_rows[:30]
        ),
        "final_test_was_not_used_for": [
            "model fitting",
            "tree-count selection",
            "probability calibration",
            "meta-label training",
            "trade threshold selection",
        ],
    }

    (
        model_dir
        / "metadata.json"
    ).write_text(
        json.dumps(
            _json_safe(
                metadata
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    metrics_path = (
        report_dir
        / "metrics.json"
    )

    metrics_path.write_text(
        json.dumps(
            _json_safe(
                metadata
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions = test[
        [
            "time",
            "decision_time",
            "entry_time",
            "entry_open",
            "exit_time",
            "exit_close",
            "target_up",
            "future_move_atr",
            "future_move_bps",
        ]
    ].copy()

    predictions[
        "p_up_raw"
    ] = raw_p_test

    predictions[
        "p_up"
    ] = p_test

    predictions[
        "p_down"
    ] = (
        1.0 - p_test
    )

    predictions[
        "confidence"
    ] = np.maximum(
        p_test,
        1.0 - p_test,
    )

    predictions[
        "direction"
    ] = np.where(
        p_test >= 0.5,
        "UP",
        "DOWN",
    )

    predictions[
        "expected_move_atr"
    ] = (
        move_test_pred
    )

    predictions.to_csv(
        report_dir
        / "test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    print(
        f"Ensemble members         : "
        f"{len(final_models)}"
    )
    print(
        f"Move-model trees         : "
        f"{regressor_trees}"
    )
    print(
        f"Validation accuracy      : "
        f"{validation_metrics['accuracy']:.4f}"
    )
    print(
        f"Validation ROC-AUC       : "
        f"{validation_metrics['roc_auc']}"
    )
    print(
        f"FINAL TEST accuracy      : "
        f"{test_metrics['accuracy']:.4f}"
    )
    print(
        f"FINAL TEST balanced acc  : "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )
    print(
        f"FINAL TEST ROC-AUC       : "
        f"{test_metrics['roc_auc']}"
    )
    print(
        f"FINAL TEST log loss      : "
        f"{test_metrics['log_loss']:.4f}"
    )
    print(
        f"Majority baseline        : "
        f"{majority_accuracy:.4f}"
    )
    print(
        f"Persistence baseline     : "
        f"{persistence_accuracy:.4f}"
    )
    print(
        f"Reserved META rows       : "
        f"{len(meta):,}"
    )

    print()
    print(
        "Confidence diagnostics "
        "(FINAL TEST; diagnostics only):"
    )

    for row in confidence_table:
        accuracy = row[
            "accuracy"
        ]
        move = row[
            "mean_signed_realized_move_atr"
        ]

        print(
            f"  >= {row['confidence_at_least']:.2f}"
            f" | coverage={row['coverage']:.3f}"
            f" | accuracy="
            + (
                f"{accuracy:.4f}"
                if accuracy is not None
                else "n/a"
            )
            + " | signed_move="
            + (
                f"{move:+.4f} ATR"
                if move is not None
                else "n/a"
            )
            + f" | rows={row['rows']:,}"
        )

    print()
    print(
        f"Model   : {model_dir}"
    )
    print(
        f"Metrics : {metrics_path}"
    )

    return metadata


def train_all() -> None:
    ensure_directories()

    summaries = {}

    for symbol in SYMBOLS:
        summaries[
            symbol
        ] = train_symbol(
            symbol
        )

    summary_path = (
        symbol_report_dir(
            SYMBOLS[0]
        ).parent
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "trained_utc": (
                        datetime.now(
                            timezone.utc
                        ).isoformat()
                    ),
                    "model_version": (
                        MODEL_VERSION
                    ),
                    "symbols": summaries,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "FINAL DIRECTION ENSEMBLE "
        "TRAINING COMPLETE"
    )
    print("=" * 78)
    print(
        f"Summary: {summary_path}"
    )


if __name__ == "__main__":
    train_all()
