from __future__ import annotations

import numpy as np
import pandas as pd


M5_MINUTES = 5
H1_MINUTES = 60


def _clean_bars(
    df: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "time",
        "open",
        "high",
        "low",
        "close",
    }

    missing = required.difference(
        df.columns
    )

    if missing:
        raise ValueError(
            "Missing market columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    out = df.copy()

    out["time"] = pd.to_datetime(
        out["time"],
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
        if column not in out.columns:
            out[column] = 0.0

        out[column] = pd.to_numeric(
            out[column],
            errors="coerce",
        )

    if "symbol" not in out.columns:
        out["symbol"] = "UNKNOWN"

    out["symbol"] = (
        out["symbol"]
        .astype(str)
        .str.upper()
    )

    out = out.dropna(
        subset=[
            "time",
            "open",
            "high",
            "low",
            "close",
        ]
    )

    valid = (
        (out["open"] > 0)
        & (out["high"] > 0)
        & (out["low"] > 0)
        & (out["close"] > 0)
        & (out["high"] >= out["low"])
        & (out["high"] >= out["open"])
        & (out["high"] >= out["close"])
        & (out["low"] <= out["open"])
        & (out["low"] <= out["close"])
    )

    out = out.loc[
        valid
    ].copy()

    out = (
        out
        .drop_duplicates(
            subset=["time"],
            keep="last",
        )
        .sort_values("time")
        .reset_index(drop=True)
    )

    return out


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    zero_value: float,
) -> pd.Series:
    denominator = denominator.astype(float)

    result = (
        numerator.astype(float)
        / denominator.replace(
            0.0,
            np.nan,
        )
    )

    result = result.where(
        denominator != 0.0,
        zero_value,
    )

    return result


def _rolling_zscore(
    series: pd.Series,
    window: int,
) -> pd.Series:
    mean = series.rolling(
        window,
        min_periods=window,
    ).mean()

    std = series.rolling(
        window,
        min_periods=window,
    ).std(ddof=0)

    z = (
        (series - mean)
        / std.replace(
            0.0,
            np.nan,
        )
    )

    # A constant rolling series is valid market information,
    # not missing data. Its standardized value is neutral (0).
    z = z.where(
        std != 0.0,
        0.0,
    )

    return z


def _rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(
        lower=0.0
    )

    loss = (
        -delta.clip(
            upper=0.0
        )
    )

    avg_gain = gain.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1.0 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0.0,
            np.nan,
        )
    )

    rsi = (
        100.0
        - 100.0
        / (1.0 + rs)
    )

    rsi = rsi.where(
        avg_loss > 0,
        100.0,
    )

    rsi = rsi.where(
        ~(
            (avg_loss == 0)
            & (avg_gain == 0)
        ),
        50.0,
    )

    return rsi


def _true_range(
    df: pd.DataFrame,
) -> pd.Series:
    previous_close = (
        df["close"]
        .shift(1)
    )

    parts = pd.concat(
        [
            df["high"] - df["low"],
            (
                df["high"]
                - previous_close
            ).abs(),
            (
                df["low"]
                - previous_close
            ).abs(),
        ],
        axis=1,
    )

    return parts.max(axis=1)


def _price_features(
    df: pd.DataFrame,
    prefix: str,
) -> pd.DataFrame:
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]

    result = pd.DataFrame(
        index=df.index
    )

    safe_close = close.replace(
        0.0,
        np.nan,
    )

    candle_range = (
        high - low
    ).clip(lower=0.0)

    body = (
        close - open_
    )

    upper_wick = (
        high
        - pd.concat(
            [open_, close],
            axis=1,
        ).max(axis=1)
    ).clip(lower=0.0)

    lower_wick = (
        pd.concat(
            [open_, close],
            axis=1,
        ).min(axis=1)
        - low
    ).clip(lower=0.0)

    result[
        f"{prefix}body_pct"
    ] = body / safe_close

    result[
        f"{prefix}range_pct"
    ] = candle_range / safe_close

    result[
        f"{prefix}body_to_range"
    ] = _safe_ratio(
        body.abs(),
        candle_range,
        zero_value=0.0,
    )

    result[
        f"{prefix}upper_wick_pct"
    ] = (
        upper_wick
        / safe_close
    )

    result[
        f"{prefix}lower_wick_pct"
    ] = (
        lower_wick
        / safe_close
    )

    result[
        f"{prefix}close_location"
    ] = _safe_ratio(
        close - low,
        candle_range,
        zero_value=0.5,
    )

    result[
        f"{prefix}gap_return"
    ] = (
        open_
        / close.shift(1)
        - 1.0
    )

    for periods in (
        1,
        2,
        3,
        6,
        12,
        24,
        48,
    ):
        result[
            f"{prefix}return_{periods}"
        ] = close.pct_change(
            periods
        )

    returns_1 = close.pct_change()

    for window in (
        6,
        12,
        24,
        48,
        96,
    ):
        result[
            f"{prefix}volatility_{window}"
        ] = returns_1.rolling(
            window,
            min_periods=window,
        ).std(ddof=0)

    tr = _true_range(df)

    atr14 = tr.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    result[
        f"{prefix}atr_pct_14"
    ] = (
        atr14
        / safe_close
    )

    result[
        f"{prefix}range_atr_ratio"
    ] = _safe_ratio(
        candle_range,
        atr14,
        zero_value=0.0,
    )

    result[
        f"{prefix}rsi_14"
    ] = (
        _rsi(
            close,
            14,
        )
        / 100.0
    )

    for span in (
        5,
        9,
        20,
        50,
    ):
        ema = close.ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()

        result[
            f"{prefix}ema_distance_{span}"
        ] = (
            close
            / ema.replace(
                0.0,
                np.nan,
            )
            - 1.0
        )

        result[
            f"{prefix}ema_slope_{span}_3"
        ] = ema.pct_change(3)

    ema_fast = close.ewm(
        span=12,
        adjust=False,
        min_periods=12,
    ).mean()

    ema_slow = close.ewm(
        span=26,
        adjust=False,
        min_periods=26,
    ).mean()

    macd = (
        ema_fast
        - ema_slow
    )

    macd_signal = macd.ewm(
        span=9,
        adjust=False,
        min_periods=9,
    ).mean()

    result[
        f"{prefix}macd_pct"
    ] = (
        macd
        / safe_close
    )

    result[
        f"{prefix}macd_hist_pct"
    ] = (
        (macd - macd_signal)
        / safe_close
    )

    for window in (
        12,
        24,
        48,
        96,
    ):
        rolling_high = high.rolling(
            window,
            min_periods=window,
        ).max()

        rolling_low = low.rolling(
            window,
            min_periods=window,
        ).min()

        width = (
            rolling_high
            - rolling_low
        )

        result[
            f"{prefix}range_position_{window}"
        ] = _safe_ratio(
            close - rolling_low,
            width,
            zero_value=0.5,
        )

        result[
            f"{prefix}distance_high_{window}"
        ] = (
            close
            / rolling_high.replace(
                0.0,
                np.nan,
            )
            - 1.0
        )

        result[
            f"{prefix}distance_low_{window}"
        ] = (
            close
            / rolling_low.replace(
                0.0,
                np.nan,
            )
            - 1.0
        )

    tick_volume = df[
        "tick_volume"
    ].astype(float)

    spread = df[
        "spread"
    ].astype(float)

    result[
        f"{prefix}tick_volume_log"
    ] = np.log1p(
        tick_volume.clip(
            lower=0.0
        )
    )

    for window in (
        24,
        96,
    ):
        result[
            f"{prefix}tick_volume_z_{window}"
        ] = _rolling_zscore(
            tick_volume,
            window,
        )

        result[
            f"{prefix}spread_z_{window}"
        ] = _rolling_zscore(
            spread,
            window,
        )

    result[
        f"{prefix}spread_points"
    ] = spread

    return result


def build_m5_features(
    m5: pd.DataFrame,
) -> pd.DataFrame:
    m5 = _clean_bars(m5)

    features = _price_features(
        m5,
        "m5_",
    )

    decision_time = (
        m5["time"]
        + pd.Timedelta(
            minutes=M5_MINUTES
        )
    )

    minute_of_day = (
        decision_time.dt.hour * 60
        + decision_time.dt.minute
    )

    angle = (
        2.0
        * np.pi
        * minute_of_day
        / 1440.0
    )

    weekday_angle = (
        2.0
        * np.pi
        * decision_time.dt.dayofweek
        / 7.0
    )

    time_features = pd.DataFrame(
        {
            "time_day_sin": np.sin(angle),
            "time_day_cos": np.cos(angle),
            "time_weekday_sin": np.sin(
                weekday_angle
            ),
            "time_weekday_cos": np.cos(
                weekday_angle
            ),
        },
        index=m5.index,
    )

    out = pd.concat(
        [
            m5[
                [
                    "time",
                    "symbol",
                    "close",
                ]
            ].copy(),
            features,
            time_features,
        ],
        axis=1,
    )

    out[
        "decision_time"
    ] = decision_time

    return out


def build_h1_features(
    h1: pd.DataFrame,
) -> pd.DataFrame:
    h1 = _clean_bars(h1)

    features = _price_features(
        h1,
        "h1_",
    )

    out = pd.concat(
        [
            h1[
                [
                    "time",
                    "symbol",
                ]
            ].copy(),
            features,
        ],
        axis=1,
    )

    out[
        "h1_available_time"
    ] = (
        out["time"]
        + pd.Timedelta(
            minutes=H1_MINUTES
        )
    )

    return out


def model_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    """Return the causal model feature whitelist.

    Historical MT5 spread fields are intentionally excluded. In this
    dataset many historical spread observations can be zero/missing, so
    allowing spread-derived columns into the predictive model can teach
    feed artefacts instead of market structure. Runtime transaction cost
    handling belongs to the strategy/cost layer, not this direction model.
    """

    prefixes = (
        "m5_",
        "h1_",
        "time_",
    )

    return [
        column
        for column in frame.columns
        if column.startswith(prefixes)
        and "spread" not in column.lower()
    ]


def build_feature_frame(
    m5: pd.DataFrame,
    h1: pd.DataFrame,
) -> pd.DataFrame:
    m5_features = build_m5_features(
        m5
    )

    h1_features = build_h1_features(
        h1
    )

    m5_symbols = set(
        m5_features[
            "symbol"
        ].dropna().unique()
    )

    h1_symbols = set(
        h1_features[
            "symbol"
        ].dropna().unique()
    )

    if len(m5_symbols) != 1:
        raise ValueError(
            "M5 feature build expects exactly "
            "one symbol at a time."
        )

    if len(h1_symbols) != 1:
        raise ValueError(
            "H1 feature build expects exactly "
            "one symbol at a time."
        )

    if m5_symbols != h1_symbols:
        raise ValueError(
            "M5/H1 symbol mismatch."
        )

    right = h1_features.drop(
        columns=[
            "time",
            "symbol",
        ]
    ).sort_values(
        "h1_available_time"
    )

    merged = pd.merge_asof(
        m5_features.sort_values(
            "decision_time"
        ),
        right,
        left_on="decision_time",
        right_on="h1_available_time",
        direction="backward",
        allow_exact_matches=True,
    )

    merged = merged.drop(
        columns=[
            "h1_available_time",
        ],
        errors="ignore",
    )

    feature_columns = (
        model_feature_columns(
            merged
        )
    )

    merged[
        feature_columns
    ] = (
        merged[
            feature_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
    )

    return merged