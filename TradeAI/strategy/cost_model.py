from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from shared.tradeai_core.broker_profile import (
    get_symbol_spec,
    load_broker_profile,
    pip_size_from_point_digits,
)
from Trade_Bot_Training.direction_config import (
    raw_path,
)
from TradeAI.strategy.strategy_config import (
    BROKER_PROFILE_PATH,
    SPREAD_FALLBACK_LOOKBACK_DAYS,
    SPREAD_FALLBACK_MIN_ROWS,
    SPREAD_REGIME_MAX_RATIO,
    SPREAD_REGIME_MIN_RATIO,
)


@dataclass(frozen=True)
class SymbolCostSpec:
    symbol: str
    digits: int
    point: float
    pip_size: float
    tick_size: float
    tick_value: float
    commission_per_lot: float
    commission_pips: float
    slippage_pips: float
    snapshot_spread_pips: float | None


def load_symbol_cost_spec(
    symbol: str,
    profile_path: str | Path = BROKER_PROFILE_PATH,
) -> SymbolCostSpec:
    symbol = str(symbol).upper()

    profile = load_broker_profile(
        profile_path
    )

    if profile is None:
        raise RuntimeError(
            "Broker profile is missing or invalid: "
            f"{profile_path}"
        )

    spec = get_symbol_spec(
        profile,
        symbol,
    )

    if spec is None:
        raise RuntimeError(
            f"Broker profile has no symbol: "
            f"{symbol}"
        )

    digits = int(
        spec.get(
            "digits",
            0,
        )
        or 0
    )

    point = float(
        spec.get(
            "point",
            0.0,
        )
        or 0.0
    )

    tick_size = float(
        spec.get(
            "trade_tick_size",
            0.0,
        )
        or 0.0
    )

    tick_value = float(
        spec.get(
            "trade_tick_value",
            0.0,
        )
        or 0.0
    )

    commission = float(
        spec.get(
            "commission_per_lot",
            0.0,
        )
        or 0.0
    )

    slippage = float(
        spec.get(
            "slippage_pips",
            0.0,
        )
        or 0.0
    )

    snapshot_spread = spec.get(
        "spread_pips"
    )

    try:
        snapshot_spread = float(
            snapshot_spread
        )
    except (
        TypeError,
        ValueError,
    ):
        snapshot_spread = None

    if (
        snapshot_spread is not None
        and (
            not np.isfinite(
                snapshot_spread
            )
            or snapshot_spread < 0.0
        )
    ):
        snapshot_spread = None

    if (
        digits <= 0
        or point <= 0.0
        or tick_size <= 0.0
        or tick_value <= 0.0
    ):
        raise RuntimeError(
            f"{symbol}: invalid broker "
            "pricing/tick metadata."
        )

    pip_size = (
        pip_size_from_point_digits(
            point,
            digits,
        )
    )

    pip_value_per_lot = (
        tick_value
        * (
            pip_size
            / tick_size
        )
    )

    if pip_value_per_lot <= 0.0:
        raise RuntimeError(
            f"{symbol}: invalid pip value."
        )

    # Existing TradeAI broker profile semantics are preserved:
    # commission_per_lot is treated as the full round-trip commission.
    commission_pips = (
        commission
        / pip_value_per_lot
    )

    return SymbolCostSpec(
        symbol=symbol,
        digits=digits,
        point=point,
        pip_size=pip_size,
        tick_size=tick_size,
        tick_value=tick_value,
        commission_per_lot=commission,
        commission_pips=float(
            commission_pips
        ),
        slippage_pips=slippage,
        snapshot_spread_pips=(
            snapshot_spread
        ),
    )


def _raw_spread_history(
    symbol: str,
) -> pd.DataFrame:
    path = raw_path(
        symbol,
        "M5",
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing raw M5 history: {path}"
        )

    raw = pd.read_csv(
        path,
        compression="gzip",
        usecols=[
            "time",
            "spread",
        ],
    )

    raw[
        "time"
    ] = pd.to_datetime(
        raw[
            "time"
        ],
        utc=True,
        errors="coerce",
    )

    raw[
        "spread"
    ] = pd.to_numeric(
        raw[
            "spread"
        ],
        errors="coerce",
    )

    return (
        raw
        .dropna(
            subset=[
                "time",
                "spread",
            ]
        )
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values(
            "time"
        )
        .reset_index(drop=True)
    )


def _as_utc_timestamp(
    value,
) -> pd.Timestamp:
    out = pd.Timestamp(
        value
    )

    if out.tzinfo is None:
        return out.tz_localize(
            "UTC"
        )

    return out.tz_convert(
        "UTC"
    )


def derive_spread_fallback(
    symbol: str,
    *,
    cutoff_time,
) -> tuple[float, dict]:
    """Freeze a zero/missing-spread fallback using PRE-META history only.

    A recent positive-spread median is preferred because broker spread regimes
    can change. If the recent window is too sparse, the full pre-cutoff
    positive-spread history is used.
    """

    spec = load_symbol_cost_spec(
        symbol
    )

    raw = _raw_spread_history(
        symbol
    )

    cutoff = _as_utc_timestamp(
        cutoff_time
    )

    raw = raw.loc[
        raw[
            "time"
        ]
        <= cutoff
    ].copy()

    spread_pips = (
        raw[
            "spread"
        ].astype(float)
        * spec.point
        / spec.pip_size
    )

    positive_mask = (
        np.isfinite(
            spread_pips
        )
        & (
            spread_pips
            > 0.0
        )
    )

    positive = spread_pips.loc[
        positive_mask
    ]

    if positive.empty:
        raise RuntimeError(
            f"{symbol}: no positive historical "
            "spread exists before policy cutoff."
        )

    full_median = float(
        positive.median()
    )

    recent_start = (
        cutoff
        - pd.Timedelta(
            days=(
                SPREAD_FALLBACK_LOOKBACK_DAYS
            )
        )
    )

    recent_mask = (
        positive_mask
        & (
            raw[
                "time"
            ]
            >= recent_start
        )
    )

    recent = spread_pips.loc[
        recent_mask
    ]

    if (
        len(recent)
        >= SPREAD_FALLBACK_MIN_ROWS
    ):
        fallback = float(
            recent.median()
        )
        source = (
            f"trailing_{SPREAD_FALLBACK_LOOKBACK_DAYS}"
            "_day_positive_median"
        )
    else:
        fallback = full_median
        source = (
            "full_pre_meta_positive_median"
        )

    snapshot = (
        spec.snapshot_spread_pips
    )

    ratio = None
    regime_warning = False

    if (
        snapshot is not None
        and snapshot > 0.0
    ):
        ratio = (
            fallback
            / snapshot
        )

        regime_warning = bool(
            ratio
            > SPREAD_REGIME_MAX_RATIO
            or ratio
            < SPREAD_REGIME_MIN_RATIO
        )

    metadata = {
        "fallback_pips": fallback,
        "fallback_source": source,
        "cutoff_utc": str(
            cutoff
        ),
        "lookback_days": (
            SPREAD_FALLBACK_LOOKBACK_DAYS
        ),
        "recent_positive_rows": int(
            len(recent)
        ),
        "all_positive_rows": int(
            len(positive)
        ),
        "recent_positive_median_pips": (
            float(
                recent.median()
            )
            if not recent.empty
            else None
        ),
        "full_positive_median_pips": (
            full_median
        ),
        "broker_snapshot_spread_pips": (
            snapshot
        ),
        "fallback_to_snapshot_ratio": (
            float(ratio)
            if ratio is not None
            else None
        ),
        "spread_regime_warning": (
            regime_warning
        ),
        "spread_regime_rule": (
            "warning only when broker snapshot spread is "
            "positive and fallback/snapshot is outside "
            f"[{SPREAD_REGIME_MIN_RATIO:.3f}, "
            f"{SPREAD_REGIME_MAX_RATIO:.3f}]"
        ),
    }

    return fallback, metadata


def attach_historical_costs(
    rows: pd.DataFrame,
    symbol: str,
    *,
    spread_fallback_pips: float,
) -> tuple[pd.DataFrame, dict]:
    """Attach side-correct historical trading costs.

    MT5 OHLC is treated as BID data.

    LONG:
        entry at ASK  -> entry BID + entry spread
        exit  at BID  -> exit_close
        spread cost   -> ENTRY spread

    SHORT:
        entry at BID  -> entry_open
        exit  at ASK  -> exit BID + exit spread
        spread cost   -> EXIT spread

    The future exit spread is used ONLY in realized SHORT labels/evaluation.
    It is never placed in the live feature matrix.
    """

    symbol = str(
        symbol
    ).upper()

    spec = load_symbol_cost_spec(
        symbol
    )

    raw = _raw_spread_history(
        symbol
    )

    entry_lookup = raw.rename(
        columns={
            "time": "entry_time",
            "spread": (
                "entry_spread_points"
            ),
        }
    )

    exit_lookup = raw.rename(
        columns={
            "time": "exit_time",
            "spread": (
                "exit_spread_points"
            ),
        }
    )

    out = rows.copy()

    for column in (
        "entry_time",
        "exit_time",
    ):
        out[
            column
        ] = pd.to_datetime(
            out[
                column
            ],
            utc=True,
            errors="coerce",
        )

    out = out.merge(
        entry_lookup,
        how="left",
        on="entry_time",
        validate="many_to_one",
    )

    out = out.merge(
        exit_lookup,
        how="left",
        on="exit_time",
        validate="many_to_one",
    )

    out[
        "entry_spread_pips"
    ] = (
        pd.to_numeric(
            out[
                "entry_spread_points"
            ],
            errors="coerce",
        )
        * spec.point
        / spec.pip_size
    )

    out[
        "exit_spread_pips"
    ] = (
        pd.to_numeric(
            out[
                "exit_spread_points"
            ],
            errors="coerce",
        )
        * spec.point
        / spec.pip_size
    )

    bad_entry = (
        ~np.isfinite(
            out[
                "entry_spread_pips"
            ]
        )
        | (
            out[
                "entry_spread_pips"
            ]
            <= 0.0
        )
    )

    bad_exit = (
        ~np.isfinite(
            out[
                "exit_spread_pips"
            ]
        )
        | (
            out[
                "exit_spread_pips"
            ]
            <= 0.0
        )
    )

    out.loc[
        bad_entry,
        "entry_spread_pips",
    ] = float(
        spread_fallback_pips
    )

    out.loc[
        bad_exit,
        "exit_spread_pips",
    ] = float(
        spread_fallback_pips
    )

    out[
        "commission_pips"
    ] = (
        spec.commission_pips
    )

    out[
        "slippage_pips"
    ] = (
        spec.slippage_pips
    )

    out[
        "fixed_round_trip_cost_pips"
    ] = (
        out[
            "commission_pips"
        ]
        + out[
            "slippage_pips"
        ]
    )

    out[
        "buy_all_in_cost_pips"
    ] = (
        out[
            "entry_spread_pips"
        ]
        + out[
            "fixed_round_trip_cost_pips"
        ]
    )

    out[
        "sell_all_in_cost_pips"
    ] = (
        out[
            "exit_spread_pips"
        ]
        + out[
            "fixed_round_trip_cost_pips"
        ]
    )

    # Compatibility / live-feature aliases. These intentionally use only
    # information known at entry time.
    out[
        "historical_spread_pips"
    ] = out[
        "entry_spread_pips"
    ]

    out[
        "all_in_cost_pips"
    ] = (
        out[
            "entry_spread_pips"
        ]
        + out[
            "fixed_round_trip_cost_pips"
        ]
    )

    close = pd.to_numeric(
        out[
            "close"
        ],
        errors="coerce",
    )

    atr_pct = pd.to_numeric(
        out[
            "m5_atr_pct_14"
        ],
        errors="coerce",
    )

    out[
        "atr_pips"
    ] = (
        close
        * atr_pct
        / spec.pip_size
    )

    metadata = {
        "symbol": symbol,
        "pip_size": spec.pip_size,
        "commission_per_lot": (
            spec.commission_per_lot
        ),
        "commission_pips": (
            spec.commission_pips
        ),
        "slippage_pips": (
            spec.slippage_pips
        ),
        "spread_fallback_pips": float(
            spread_fallback_pips
        ),
        "entry_spread_rows_imputed": int(
            bad_entry.sum()
        ),
        "exit_spread_rows_imputed": int(
            bad_exit.sum()
        ),
        "commission_semantics": (
            "commission_per_lot is treated "
            "as full round-trip commission"
        ),
        "slippage_semantics": (
            "broker-profile slippage_pips is treated "
            "as the full round-trip slippage allowance"
        ),
        "spread_semantics": (
            "MT5 OHLC is BID. BUY realized spread cost "
            "uses entry spread; SELL realized spread cost "
            "uses exit spread. Zero/missing spread uses a "
            "fallback frozen strictly before META training."
        ),
        "live_feature_cost_semantics": (
            "At decision time only current/entry spread + "
            "commission + slippage is supplied as a feature. "
            "Future exit spread is target/evaluation only."
        ),
    }

    return out, metadata


def current_all_in_cost_pips(
    symbol: str,
    spread_pips: np.ndarray | float,
) -> np.ndarray:
    """Entry-time cost estimate available to a live policy."""

    spec = load_symbol_cost_spec(
        symbol
    )

    spread = np.asarray(
        spread_pips,
        dtype=float,
    )

    return (
        spread
        + spec.commission_pips
        + spec.slippage_pips
    )
