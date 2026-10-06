from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from Trade_Bot_Training.precision_config import (
    ACTION_CONFIDENCE,
    HORIZON_BARS,
    HORIZON_MINUTES,
    MIN_MEANINGFUL_MOVE_ATR,
    SYMBOLS as MODEL_SYMBOLS,
)
from Trade_Bot_Training.precision_feature_engine import (
    build_precision_features,
)
from Trade_Bot_Training.precision_model import (
    PrecisionDirectionModel,
)
from TradeAI.analytics.logger import log


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = SYSTEM_ROOT / "artifacts" / "reports"

STATE_PATH = REPORT_DIR / "precision_live_state.json"
PENDING_PATH = REPORT_DIR / "precision_live_pending.json"

DECISIONS_CSV = REPORT_DIR / "precision_live_decisions.csv"
SIGNALS_CSV = REPORT_DIR / "precision_live_signals.csv"
RESOLVED_CSV = REPORT_DIR / "precision_live_resolved.csv"

RECENT_M5_BARS = 1200
M5_DELTA = pd.Timedelta(minutes=5)

DECISION_FIELDS = [
    "recorded_utc",
    "symbol",
    "bar_time",
    "decision_time",
    "p_down",
    "p_hold",
    "p_up",
    "raw_class",
    "signal",
    "action",
    "confidence",
    "tradeable",
    "historical_gate_pass",
    "current_atr",
    "bid",
    "ask",
    "spread_pips",
]

RESOLVED_FIELDS = [
    "resolved_utc",
    "symbol",
    "bar_time",
    "decision_time",
    "signal",
    "action",
    "confidence",
    "p_down",
    "p_hold",
    "p_up",
    "current_atr",
    "entry_time",
    "entry_open",
    "exit_time",
    "exit_close",
    "future_move_price",
    "future_move_atr",
    "realized_class",
    "correct",
    "signed_move_atr",
    "resolution_status",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _read_json(path: Path, default):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data
    except Exception:
        return default


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temp.write_text(
            json.dumps(_json_safe(payload), indent=2),
            encoding="utf-8",
        )
        os.replace(temp, path)
    finally:
        try:
            temp.unlink(missing_ok=True)
        except Exception:
            pass


def _append_csv(
    path: Path,
    row: dict[str, Any],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0

    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )

        if not exists:
            writer.writeheader()

        safe = {}
        for field in fields:
            value = row.get(field)
            if isinstance(value, (pd.Timestamp, datetime)):
                value = str(value)
            elif isinstance(value, (np.integer, np.floating)):
                value = value.item()
            safe[field] = value

        writer.writerow(safe)


def _normalize_rates(
    symbol: str,
    rates,
) -> pd.DataFrame:
    if rates is None or len(rates) == 0:
        return pd.DataFrame()

    frame = pd.DataFrame(rates)

    if "time" not in frame.columns:
        return pd.DataFrame()

    frame["time"] = pd.to_datetime(
        frame["time"],
        unit="s",
        utc=True,
        errors="coerce",
    )

    for column in (
        "open",
        "high",
        "low",
        "close",
        "tick_volume",
        "spread",
        "real_volume",
    ):
        if column not in frame.columns:
            frame[column] = 0.0

        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    frame["symbol"] = symbol

    return (
        frame
        .dropna(
            subset=[
                "time",
                "open",
                "high",
                "low",
                "close",
            ]
        )
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )


@dataclass
class LiveModelState:
    symbol: str
    model: PrecisionDirectionModel
    last_processed_bar: pd.Timestamp | None = None


class PrecisionLiveEngine:
    """Continuous precision-model signal engine.

    This engine does not place broker orders.

    Responsibilities:
      * watch every configured symbol;
      * run exactly once for each new fully closed M5 candle;
      * expose HOLD / BUY / SELL from the trained precision model;
      * persist every decision;
      * automatically resolve tradeable signals after the model's fixed
        30-minute / 6-bar horizon;
      * write MODEL_STATE log records understood by the existing dashboard.
    """

    def __init__(
        self,
        *,
        symbols: list[str] | None = None,
        poll_seconds: float = 1.0,
    ) -> None:
        self.symbols = [
            str(s).strip().upper()
            for s in (
                symbols
                if symbols
                else MODEL_SYMBOLS
            )
            if str(s).strip()
        ]

        self.poll_seconds = max(
            0.25,
            float(poll_seconds),
        )

        self.mt5 = None
        self.models: dict[str, LiveModelState] = {}
        self.pending: list[dict[str, Any]] = []

        self.started_utc = _utc_now()

        previous = _read_json(
            STATE_PATH,
            {},
        )

        previous_symbols = (
            previous.get("symbols", {})
            if isinstance(previous, dict)
            else {}
        )

        for symbol in self.symbols:
            model = PrecisionDirectionModel(
                symbol
            )

            last_processed = None
            raw_last = (
                previous_symbols
                .get(symbol, {})
                .get("bar_time")
                if isinstance(
                    previous_symbols,
                    dict,
                )
                else None
            )

            if raw_last:
                try:
                    last_processed = pd.Timestamp(
                        raw_last
                    )
                except Exception:
                    last_processed = None

            self.models[symbol] = (
                LiveModelState(
                    symbol=symbol,
                    model=model,
                    last_processed_bar=(
                        last_processed
                    ),
                )
            )

        raw_pending = _read_json(
            PENDING_PATH,
            [],
        )

        if isinstance(raw_pending, list):
            self.pending = [
                item
                for item in raw_pending
                if isinstance(item, dict)
            ]

        self.state: dict[str, Any] = {
            "version": "tradeai_precision_live_v1",
            "started_utc": self.started_utc,
            "updated_utc": self.started_utc,
            "model": "TradeAI Selective Precision Model",
            "timeframe": "M5",
            "horizon_bars": int(
                HORIZON_BARS
            ),
            "horizon_minutes": int(
                HORIZON_MINUTES
            ),
            "action_confidence": float(
                ACTION_CONFIDENCE
            ),
            "meaningful_move_atr": float(
                MIN_MEANINGFUL_MOVE_ATR
            ),
            "execution_enabled": False,
            "note": (
                "Signal engine only. It does not place MT5 orders. "
                "MT5 bar timestamps are preserved exactly as supplied by the "
                "broker because the trained feature clock uses the same convention."
            ),
            "symbols": (
                previous_symbols
                if isinstance(
                    previous_symbols,
                    dict,
                )
                else {}
            ),
            "forward_audit": {},
        }

    def connect(self) -> None:
        try:
            import MetaTrader5 as mt5
        except Exception as exc:
            raise RuntimeError(
                "MetaTrader5 Python package is not available."
            ) from exc

        self.mt5 = mt5

        if not mt5.initialize():
            raise RuntimeError(
                f"MT5 initialization failed: "
                f"{mt5.last_error()}"
            )

        for symbol in self.symbols:
            info = mt5.symbol_info(
                symbol
            )

            if info is None:
                raise RuntimeError(
                    f"MT5 symbol not available: "
                    f"{symbol}"
                )

            if (
                not info.visible
                and not mt5.symbol_select(
                    symbol,
                    True,
                )
            ):
                raise RuntimeError(
                    f"Unable to enable MT5 symbol: "
                    f"{symbol}"
                )

        print(
            "MT5 connected."
        )

    def disconnect(self) -> None:
        if self.mt5 is None:
            return

        try:
            self.mt5.shutdown()
        except Exception:
            pass

        self.mt5 = None

        print(
            "MT5 disconnected."
        )

    def _latest_closed_bar_time(
        self,
        symbol: str,
    ) -> pd.Timestamp | None:
        rates = self.mt5.copy_rates_from_pos(
            symbol,
            self.mt5.TIMEFRAME_M5,
            1,
            1,
        )

        frame = _normalize_rates(
            symbol,
            rates,
        )

        if frame.empty:
            return None

        return pd.Timestamp(
            frame[
                "time"
            ].iloc[
                -1
            ]
        )

    def _closed_history(
        self,
        symbol: str,
        count: int = RECENT_M5_BARS,
    ) -> pd.DataFrame:
        rates = self.mt5.copy_rates_from_pos(
            symbol,
            self.mt5.TIMEFRAME_M5,
            1,
            int(count),
        )

        frame = _normalize_rates(
            symbol,
            rates,
        )

        if frame.empty:
            raise RuntimeError(
                f"No closed M5 data for "
                f"{symbol}."
            )

        return frame

    def _quote(
        self,
        symbol: str,
    ) -> tuple[
        float | None,
        float | None,
        float | None,
    ]:
        try:
            tick = self.mt5.symbol_info_tick(
                symbol
            )

            info = self.mt5.symbol_info(
                symbol
            )

            if (
                tick is None
                or info is None
            ):
                return (
                    None,
                    None,
                    None,
                )

            bid = float(
                tick.bid
            )

            ask = float(
                tick.ask
            )

            digits = int(
                info.digits
            )

            point = float(
                info.point
            )

            pip = (
                point * 10.0
                if digits in (
                    3,
                    5,
                )
                else point
            )

            spread = (
                (ask - bid) / pip
                if pip > 0.0
                else None
            )

            return (
                bid,
                ask,
                spread,
            )
        except Exception:
            return (
                None,
                None,
                None,
            )

    def _predict_symbol(
        self,
        symbol: str,
        history: pd.DataFrame,
    ) -> dict[str, Any]:
        model_state = self.models[
            symbol
        ]

        feature_frame = (
            build_precision_features(
                history
            )
        )

        if feature_frame.empty:
            raise RuntimeError(
                f"{symbol}: no precision "
                "features."
            )

        latest = feature_frame.tail(
            1
        ).copy()

        latest_bar_time = pd.Timestamp(
            latest[
                "time"
            ].iloc[
                0
            ]
        )

        newest_closed_time = pd.Timestamp(
            history[
                "time"
            ].iloc[
                -1
            ]
        )

        if (
            latest_bar_time
            != newest_closed_time
        ):
            raise RuntimeError(
                f"{symbol}: precision features "
                "did not finish on the newest "
                "closed M5 candle."
            )

        missing = [
            column
            for column
            in model_state.model.feature_columns
            if column
            not in latest.columns
        ]

        if missing:
            raise RuntimeError(
                f"{symbol}: feature/model mismatch: "
                + ", ".join(
                    missing[:20]
                )
            )

        matrix = latest[
            model_state.model.feature_columns
        ].astype(
            "float64"
        )

        bad = [
            column
            for column
            in model_state.model.feature_columns
            if not np.isfinite(
                matrix[
                    column
                ].iloc[
                    0
                ]
            )
        ]

        if bad:
            raise RuntimeError(
                f"{symbol}: latest closed M5 has "
                "invalid precision features: "
                + ", ".join(
                    bad[:20]
                )
            )

        predicted = (
            model_state.model.predict(
                latest
            ).iloc[
                0
            ]
        )

        signal = str(
            predicted[
                "signal"
            ]
        ).upper()

        if signal == "UP":
            action = "BUY"
            numeric_signal = 1.0
        elif signal == "DOWN":
            action = "SELL"
            numeric_signal = -1.0
        else:
            action = "HOLD"
            numeric_signal = 0.0

        bid, ask, spread = (
            self._quote(
                symbol
            )
        )

        decision_time = pd.Timestamp(
            latest[
                "decision_time"
            ].iloc[
                0
            ]
        )

        current_atr = float(
            latest[
                "current_atr"
            ].iloc[
                0
            ]
        )

        row = {
            "recorded_utc": (
                _utc_now()
            ),
            "symbol": symbol,
            "bar_time": str(
                latest_bar_time
            ),
            "decision_time": str(
                decision_time
            ),
            "p_down": float(
                predicted[
                    "p_down"
                ]
            ),
            "p_hold": float(
                predicted[
                    "p_hold"
                ]
            ),
            "p_up": float(
                predicted[
                    "p_up"
                ]
            ),
            "raw_class": str(
                predicted[
                    "raw_class"
                ]
            ),
            "signal": signal,
            "action": action,
            "numeric_signal": (
                numeric_signal
            ),
            "confidence": float(
                predicted[
                    "confidence"
                ]
            ),
            "tradeable": bool(
                predicted[
                    "tradeable"
                ]
            ),
            "historical_gate_pass": bool(
                predicted[
                    "historical_gate_pass"
                ]
            ),
            "action_confidence": float(
                model_state.model.action_confidence
            ),
            "current_atr": (
                current_atr
            ),
            "bid": bid,
            "ask": ask,
            "spread_pips": spread,
        }

        return row

    def _decision_id(
        self,
        row: dict[str, Any],
    ) -> str:
        return (
            f"{row['symbol']}|"
            f"{row['decision_time']}"
        )

    def _already_pending(
        self,
        decision_id: str,
    ) -> bool:
        return any(
            str(
                item.get(
                    "decision_id",
                    "",
                )
            )
            == decision_id
            for item in self.pending
        )

    def _record_decision(
        self,
        row: dict[str, Any],
    ) -> None:
        _append_csv(
            DECISIONS_CSV,
            row,
            DECISION_FIELDS,
        )

        if row[
            "tradeable"
        ]:
            _append_csv(
                SIGNALS_CSV,
                row,
                DECISION_FIELDS,
            )

            decision_id = (
                self._decision_id(
                    row
                )
            )

            if not self._already_pending(
                decision_id
            ):
                pending = {
                    "decision_id": (
                        decision_id
                    ),
                    "symbol": (
                        row[
                            "symbol"
                        ]
                    ),
                    "bar_time": (
                        row[
                            "bar_time"
                        ]
                    ),
                    "decision_time": (
                        row[
                            "decision_time"
                        ]
                    ),
                    "signal": (
                        row[
                            "signal"
                        ]
                    ),
                    "action": (
                        row[
                            "action"
                        ]
                    ),
                    "confidence": (
                        row[
                            "confidence"
                        ]
                    ),
                    "p_down": (
                        row[
                            "p_down"
                        ]
                    ),
                    "p_hold": (
                        row[
                            "p_hold"
                        ]
                    ),
                    "p_up": (
                        row[
                            "p_up"
                        ]
                    ),
                    "current_atr": (
                        row[
                            "current_atr"
                        ]
                    ),
                    "created_utc": (
                        row[
                            "recorded_utc"
                        ]
                    ),
                }

                self.pending.append(
                    pending
                )

                _atomic_json(
                    PENDING_PATH,
                    self.pending,
                )

    def _write_dashboard_log(
        self,
        row: dict[str, Any],
    ) -> None:
        # Existing TradeAIDataService understands this exact MODEL_STATE shape.
        log(
            "MODEL_STATE | "
            f"symbol={row['symbol']} "
            f"signal={row['numeric_signal']:+.1f} "
            f"confidence={row['confidence']:.6f} "
            f"min_confidence={row['action_confidence']:.6f} "
            f"signal_threshold={MIN_MEANINGFUL_MOVE_ATR:.6f} "
            f"enabled={1 if row['historical_gate_pass'] else 0} "
            f"p_buy={row['p_up']:.6f} "
            f"p_hold={row['p_hold']:.6f} "
            f"p_sell={row['p_down']:.6f} "
            f"candle={row['bar_time']}"
        )

    def _print_decision(
        self,
        row: dict[str, Any],
    ) -> None:
        prefix = (
            ">>> SIGNAL"
            if row[
                "tradeable"
            ]
            else "    STATE"
        )

        print(
            f"{prefix} | "
            f"{row['symbol']:<6} | "
            f"{row['action']:<4} | "
            f"conf={row['confidence']:.4f} | "
            f"D={row['p_down']:.4f} "
            f"H={row['p_hold']:.4f} "
            f"U={row['p_up']:.4f} | "
            f"bar={row['bar_time']}"
        )

    def _resolve_one(
        self,
        pending: dict[str, Any],
        history: pd.DataFrame,
    ) -> dict[str, Any] | None:
        decision_time = pd.Timestamp(
            pending[
                "decision_time"
            ]
        )

        wanted_times = [
            decision_time
            + (
                i
                * M5_DELTA
            )
            for i in range(
                HORIZON_BARS
            )
        ]

        lookup = (
            history
            .set_index(
                "time",
                drop=False,
            )
        )

        if (
            wanted_times[
                -1
            ]
            not in lookup.index
        ):
            return None

        if any(
            wanted
            not in lookup.index
            for wanted in wanted_times
        ):
            result = {
                "resolved_utc": (
                    _utc_now()
                ),
                **{
                    key: pending.get(
                        key
                    )
                    for key in (
                        "symbol",
                        "bar_time",
                        "decision_time",
                        "signal",
                        "action",
                        "confidence",
                        "p_down",
                        "p_hold",
                        "p_up",
                        "current_atr",
                    )
                },
                "entry_time": str(
                    wanted_times[
                        0
                    ]
                ),
                "entry_open": None,
                "exit_time": str(
                    wanted_times[
                        -1
                    ]
                ),
                "exit_close": None,
                "future_move_price": None,
                "future_move_atr": None,
                "realized_class": (
                    "INVALID_GAP"
                ),
                "correct": None,
                "signed_move_atr": None,
                "resolution_status": (
                    "INVALID_NON_CONTIGUOUS"
                ),
            }
            return result

        target_rows = [
            lookup.loc[
                wanted
            ]
            for wanted in wanted_times
        ]

        entry_open = float(
            target_rows[
                0
            ][
                "open"
            ]
        )

        exit_close = float(
            target_rows[
                -1
            ][
                "close"
            ]
        )

        atr = float(
            pending.get(
                "current_atr",
                0.0,
            )
            or 0.0
        )

        move_price = (
            exit_close
            - entry_open
        )

        if (
            not np.isfinite(
                atr
            )
            or atr <= 0.0
        ):
            future_move_atr = None
            realized = "INVALID_ATR"
            correct = None
            signed = None
            status = "INVALID_ATR"
        else:
            future_move_atr = (
                move_price
                / atr
            )

            if (
                future_move_atr
                >= MIN_MEANINGFUL_MOVE_ATR
            ):
                realized = "UP"
            elif (
                future_move_atr
                <= -MIN_MEANINGFUL_MOVE_ATR
            ):
                realized = "DOWN"
            else:
                realized = "HOLD"

            signal = str(
                pending.get(
                    "signal",
                    "HOLD",
                )
            ).upper()

            correct = bool(
                signal
                == realized
            )

            if signal == "UP":
                signed = (
                    future_move_atr
                )
            elif signal == "DOWN":
                signed = (
                    -future_move_atr
                )
            else:
                signed = 0.0

            status = "RESOLVED"

        return {
            "resolved_utc": (
                _utc_now()
            ),
            **{
                key: pending.get(
                    key
                )
                for key in (
                    "symbol",
                    "bar_time",
                    "decision_time",
                    "signal",
                    "action",
                    "confidence",
                    "p_down",
                    "p_hold",
                    "p_up",
                    "current_atr",
                )
            },
            "entry_time": str(
                wanted_times[
                    0
                ]
            ),
            "entry_open": (
                entry_open
            ),
            "exit_time": str(
                wanted_times[
                    -1
                ]
            ),
            "exit_close": (
                exit_close
            ),
            "future_move_price": (
                move_price
            ),
            "future_move_atr": (
                future_move_atr
            ),
            "realized_class": (
                realized
            ),
            "correct": (
                correct
            ),
            "signed_move_atr": (
                signed
            ),
            "resolution_status": (
                status
            ),
        }

    def _resolve_pending(
        self,
        symbol: str,
        history: pd.DataFrame,
    ) -> None:
        remaining = []

        for pending in self.pending:
            if (
                str(
                    pending.get(
                        "symbol",
                        "",
                    )
                ).upper()
                != symbol
            ):
                remaining.append(
                    pending
                )
                continue

            result = self._resolve_one(
                pending,
                history,
            )

            if result is None:
                remaining.append(
                    pending
                )
                continue

            _append_csv(
                RESOLVED_CSV,
                result,
                RESOLVED_FIELDS,
            )

            if (
                result[
                    "resolution_status"
                ]
                == "RESOLVED"
            ):
                mark = (
                    "CORRECT"
                    if result[
                        "correct"
                    ]
                    else "WRONG"
                )

                print(
                    "<<< AUDIT  | "
                    f"{symbol:<6} | "
                    f"{mark:<7} | "
                    f"pred={result['signal']} "
                    f"actual={result['realized_class']} | "
                    f"signed={result['signed_move_atr']:+.3f} ATR"
                )

                log(
                    "PRECISION_AUDIT | "
                    f"symbol={symbol} "
                    f"signal={result['signal']} "
                    f"actual={result['realized_class']} "
                    f"correct={1 if result['correct'] else 0} "
                    f"signed_move_atr={result['signed_move_atr']:+.6f} "
                    f"decision={result['decision_time']}"
                )
            else:
                print(
                    "<<< AUDIT  | "
                    f"{symbol:<6} | "
                    f"{result['resolution_status']} | "
                    f"decision={result['decision_time']}"
                )

        self.pending = remaining

        _atomic_json(
            PENDING_PATH,
            self.pending,
        )

    def _audit_summary(
        self,
    ) -> dict[str, Any]:
        if not RESOLVED_CSV.exists():
            return {
                "resolved_signals": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": None,
                "mean_signed_move_atr": None,
                "by_symbol": {},
            }

        try:
            frame = pd.read_csv(
                RESOLVED_CSV
            )
        except Exception:
            return {
                "resolved_signals": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": None,
                "mean_signed_move_atr": None,
                "by_symbol": {},
            }

        if frame.empty:
            return {
                "resolved_signals": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": None,
                "mean_signed_move_atr": None,
                "by_symbol": {},
            }

        valid = frame.loc[
            frame[
                "resolution_status"
            ]
            == "RESOLVED"
        ].copy()

        if valid.empty:
            return {
                "resolved_signals": 0,
                "correct": 0,
                "wrong": 0,
                "accuracy": None,
                "mean_signed_move_atr": None,
                "by_symbol": {},
            }

        valid[
            "correct_numeric"
        ] = (
            valid[
                "correct"
            ]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "true",
                    "1",
                ]
            )
            .astype(int)
        )

        valid[
            "signed_move_atr"
        ] = pd.to_numeric(
            valid[
                "signed_move_atr"
            ],
            errors="coerce",
        )

        by_symbol = {}

        for symbol, group in valid.groupby(
            "symbol"
        ):
            rows = len(
                group
            )

            by_symbol[
                str(symbol)
            ] = {
                "resolved": int(
                    rows
                ),
                "correct": int(
                    group[
                        "correct_numeric"
                    ].sum()
                ),
                "accuracy": (
                    float(
                        group[
                            "correct_numeric"
                        ].mean()
                    )
                    if rows
                    else None
                ),
                "mean_signed_move_atr": (
                    float(
                        group[
                            "signed_move_atr"
                        ].mean()
                    )
                    if group[
                        "signed_move_atr"
                    ].notna().any()
                    else None
                ),
            }

        resolved = int(
            len(
                valid
            )
        )

        correct = int(
            valid[
                "correct_numeric"
            ].sum()
        )

        return {
            "resolved_signals": (
                resolved
            ),
            "correct": correct,
            "wrong": (
                resolved
                - correct
            ),
            "accuracy": (
                float(
                    correct
                    / resolved
                )
                if resolved
                else None
            ),
            "mean_signed_move_atr": (
                float(
                    valid[
                        "signed_move_atr"
                    ].mean()
                )
                if valid[
                    "signed_move_atr"
                ].notna().any()
                else None
            ),
            "by_symbol": (
                by_symbol
            ),
        }

    def _publish_state(
        self,
        row: dict[str, Any] | None = None,
    ) -> None:
        if row is not None:
            symbol = row[
                "symbol"
            ]

            self.state[
                "symbols"
            ][symbol] = {
                "bar_time": (
                    row[
                        "bar_time"
                    ]
                ),
                "decision_time": (
                    row[
                        "decision_time"
                    ]
                ),
                "p_down": (
                    row[
                        "p_down"
                    ]
                ),
                "p_hold": (
                    row[
                        "p_hold"
                    ]
                ),
                "p_up": (
                    row[
                        "p_up"
                    ]
                ),
                "raw_class": (
                    row[
                        "raw_class"
                    ]
                ),
                "signal": (
                    row[
                        "signal"
                    ]
                ),
                "action": (
                    row[
                        "action"
                    ]
                ),
                "confidence": (
                    row[
                        "confidence"
                    ]
                ),
                "tradeable": (
                    row[
                        "tradeable"
                    ]
                ),
                "historical_gate_pass": (
                    row[
                        "historical_gate_pass"
                    ]
                ),
                "action_confidence": (
                    row[
                        "action_confidence"
                    ]
                ),
                "bid": (
                    row[
                        "bid"
                    ]
                ),
                "ask": (
                    row[
                        "ask"
                    ]
                ),
                "spread_pips": (
                    row[
                        "spread_pips"
                    ]
                ),
                "updated_utc": (
                    row[
                        "recorded_utc"
                    ]
                ),
            }

        self.state[
            "updated_utc"
        ] = _utc_now()

        self.state[
            "pending_forward_signals"
        ] = int(
            len(
                self.pending
            )
        )

        self.state[
            "forward_audit"
        ] = self._audit_summary()

        _atomic_json(
            STATE_PATH,
            self.state,
        )

    def process_symbol(
        self,
        symbol: str,
        *,
        force: bool = False,
    ) -> bool:
        model_state = self.models[
            symbol
        ]

        latest_time = (
            self._latest_closed_bar_time(
                symbol
            )
        )

        if latest_time is None:
            return False

        if (
            not force
            and model_state.last_processed_bar
            is not None
            and latest_time
            <= model_state.last_processed_bar
        ):
            return False

        history = self._closed_history(
            symbol
        )

        self._resolve_pending(
            symbol,
            history,
        )

        row = self._predict_symbol(
            symbol,
            history,
        )

        self._record_decision(
            row
        )

        self._write_dashboard_log(
            row
        )

        self._print_decision(
            row
        )

        model_state.last_processed_bar = pd.Timestamp(
            row[
                "bar_time"
            ]
        )

        self._publish_state(
            row
        )

        return True

    def print_startup(
        self,
    ) -> None:
        print()
        print(
            "=" * 78
        )

        print(
            "TRADEAI PRECISION LIVE SIGNAL ENGINE"
        )

        print(
            "=" * 78
        )

        print(
            "Execution       : DISABLED "
            "(signals only)"
        )

        print(
            f"Action gate     : "
            f"{ACTION_CONFIDENCE:.2f}"
        )

        print(
            f"Move definition : "
            f"±{MIN_MEANINGFUL_MOVE_ATR:.2f} ATR "
            f"over {HORIZON_MINUTES} minutes"
        )

        print(
            f"Poll interval   : "
            f"{self.poll_seconds:.2f}s"
        )

        print(
            "Symbols:"
        )

        for symbol in self.symbols:
            model = self.models[
                symbol
            ].model

            gate = (
                "PASS"
                if model.historical_gate_pass
                else "DISABLED"
            )

            print(
                f"  {symbol:<6} "
                f"historical_gate={gate:<8} "
                f"confidence={model.action_confidence:.2f}"
            )

        print()
        print(
            "BUY/SELL is emitted only when the trained model itself "
            "returns Tradeable=True."
        )

        print(
            "Every tradeable signal is automatically audited after "
            f"{HORIZON_MINUTES} minutes."
        )

        print(
            "Press Ctrl+C to stop."
        )

        print(
            "=" * 78
        )
        print()

    def run_once(
        self,
        *,
        force: bool = True,
    ) -> None:
        self.connect()

        try:
            for symbol in self.symbols:
                try:
                    self.process_symbol(
                        symbol,
                        force=force,
                    )
                except Exception as exc:
                    print(
                        f"ERROR | {symbol} | {exc}"
                    )
                    log(
                        f"ERROR | PRECISION_LIVE | "
                        f"symbol={symbol} | {exc}"
                    )

            self._publish_state()
        finally:
            self.disconnect()

    def run_forever(
        self,
    ) -> None:
        self.print_startup()
        self.connect()

        try:
            # Evaluate the latest closed bar at startup if it has not already
            # been persisted by a previous engine session.
            for symbol in self.symbols:
                try:
                    self.process_symbol(
                        symbol,
                        force=False,
                    )
                except Exception as exc:
                    print(
                        f"ERROR | {symbol} | {exc}"
                    )
                    log(
                        f"ERROR | PRECISION_LIVE | "
                        f"symbol={symbol} | {exc}"
                    )

            self._publish_state()

            while True:
                any_processed = False

                for symbol in self.symbols:
                    try:
                        processed = (
                            self.process_symbol(
                                symbol,
                                force=False,
                            )
                        )

                        any_processed = (
                            any_processed
                            or processed
                        )
                    except Exception as exc:
                        print(
                            f"ERROR | {symbol} | "
                            f"{exc}"
                        )

                        log(
                            f"ERROR | PRECISION_LIVE | "
                            f"symbol={symbol} | {exc}"
                        )

                if any_processed:
                    self._publish_state()

                time.sleep(
                    self.poll_seconds
                )

        except KeyboardInterrupt:
            print()
            print(
                "Stop requested."
            )
        finally:
            self._publish_state()
            self.disconnect()


def _parse_symbols(
    raw: str,
) -> list[str]:
    if not raw.strip():
        return list(
            MODEL_SYMBOLS
        )

    return [
        part.strip().upper()
        for part in raw.split(",")
        if part.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "TradeAI precision live M5 signal engine"
        )
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help=(
            "Evaluate the newest closed M5 candle "
            "once for each symbol, then exit."
        ),
    )

    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help=(
            "Seconds between latest-candle checks."
        ),
    )

    parser.add_argument(
        "--symbols",
        default="",
        help=(
            "Comma-separated symbol subset. "
            "Default: all trained precision symbols."
        ),
    )

    args = parser.parse_args()

    engine = PrecisionLiveEngine(
        symbols=_parse_symbols(
            args.symbols
        ),
        poll_seconds=args.poll,
    )

    if args.once:
        engine.print_startup()
        engine.run_once(
            force=True
        )
    else:
        engine.run_forever()


if __name__ == "__main__":
    main()
