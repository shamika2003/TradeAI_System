from __future__ import annotations

import numpy as np
import pandas as pd


_EPS = 1e-12


def _clean_m5(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Missing M5 columns: " + ", ".join(sorted(missing)))

    out = df.copy()
    out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")

    for c in ("open", "high", "low", "close", "tick_volume"):
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["time", "open", "high", "low", "close"])

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

    return (
        out.loc[valid]
        .drop_duplicates(subset=["time"], keep="last")
        .sort_values("time")
        .reset_index(drop=True)
    )


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

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

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)

    out = out.where(avg_loss > 0.0, 100.0)
    out = out.where(
        ~((avg_loss == 0.0) & (avg_gain == 0.0)),
        50.0,
    )

    return out


def _safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
    *,
    neutral: float = 0.0,
) -> pd.Series:
    """Divide causally and use a neutral value only for a true zero denominator.

    Warm-up NaNs remain NaN so model lookback requirements are still enforced.
    """
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")

    result = num / den.where(den.abs() > _EPS)

    zero_den = (
        den.notna()
        & (den.abs() <= _EPS)
        & num.notna()
    )

    return result.where(~zero_den, neutral)


def _rolling_z(
    values: pd.Series,
    window: int,
) -> pd.Series:
    mean = values.rolling(
        window,
        min_periods=window,
    ).mean()

    std = values.rolling(
        window,
        min_periods=window,
    ).std(ddof=0)

    z = (values - mean) / std.where(std > _EPS)

    # A constant rolling series is valid information: "no relative change".
    # Treat it as neutral 0.0 instead of turning every later row into NaN.
    constant = (
        mean.notna()
        & std.notna()
        & (std <= _EPS)
        & values.notna()
    )

    return z.where(~constant, 0.0)


def build_precision_features(m5: pd.DataFrame) -> pd.DataFrame:
    """Build the causal M5 feature set used by the selective model.

    Every value is available only after the current M5 candle closes.
    No future bar, centered statistic or future-fill is used.
    """
    df = _clean_m5(m5)

    close = df["close"].astype("float64")
    open_ = df["open"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    volume = df["tick_volume"].astype("float64")

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.ewm(
        alpha=1.0 / 14.0,
        adjust=False,
        min_periods=14,
    ).mean()

    feat: dict[str, pd.Series] = {}

    feat["body_atr"] = _safe_divide(
        close - open_,
        atr,
    )

    feat["range_atr"] = _safe_divide(
        high - low,
        atr,
    )

    candle_range = high - low
    feat["close_loc"] = _safe_divide(
        close - low,
        candle_range,
        neutral=0.5,
    )

    feat["upper_wick_atr"] = _safe_divide(
        high - pd.concat([open_, close], axis=1).max(axis=1),
        atr,
    )

    feat["lower_wick_atr"] = _safe_divide(
        pd.concat([open_, close], axis=1).min(axis=1) - low,
        atr,
    )

    for bars in (1, 2, 3, 6, 12, 24, 48, 96, 192, 288):
        feat[f"ret_{bars}"] = close.pct_change(bars)

        feat[f"ret_atr_{bars}"] = _safe_divide(
            close - close.shift(bars),
            atr,
        )

    one_bar_return = close.pct_change()

    for window in (6, 12, 24, 48, 96, 192, 288):
        feat[f"vol_{window}"] = one_bar_return.rolling(
            window,
            min_periods=window,
        ).std(ddof=0)

    for span in (5, 9, 20, 50, 100, 200):
        ema = close.ewm(
            span=span,
            adjust=False,
            min_periods=span,
        ).mean()

        feat[f"ema_dist_{span}"] = _safe_divide(
            close - ema,
            atr,
        )

        feat[f"ema_slope_{span}"] = _safe_divide(
            ema - ema.shift(3),
            atr,
        )

    feat["rsi14"] = _rsi(
        close,
        14,
    ) / 100.0

    feat["atr_pct"] = _safe_divide(
        atr,
        close,
    )

    for window in (24, 48, 96, 192, 288):
        rolling_high = high.rolling(
            window,
            min_periods=window,
        ).max()

        rolling_low = low.rolling(
            window,
            min_periods=window,
        ).min()

        width = rolling_high - rolling_low

        feat[f"range_pos_{window}"] = _safe_divide(
            close - rolling_low,
            width,
            neutral=0.5,
        )

        feat[f"range_width_atr_{window}"] = _safe_divide(
            width,
            atr,
        )

    log_volume = np.log1p(
        volume.clip(lower=0.0)
    )

    feat["vol_log"] = log_volume

    for window in (24, 96, 288):
        feat[f"vol_z_{window}"] = _rolling_z(
            log_volume,
            window,
        )

    decision_time = (
        df["time"]
        + pd.Timedelta(minutes=5)
    )

    minute_of_day = (
        decision_time.dt.hour * 60
        + decision_time.dt.minute
    )

    feat["tod_sin"] = np.sin(
        2.0
        * np.pi
        * minute_of_day
        / 1440.0
    )

    feat["tod_cos"] = np.cos(
        2.0
        * np.pi
        * minute_of_day
        / 1440.0
    )

    feat["dow_sin"] = np.sin(
        2.0
        * np.pi
        * decision_time.dt.dayofweek
        / 7.0
    )

    feat["dow_cos"] = np.cos(
        2.0
        * np.pi
        * decision_time.dt.dayofweek
        / 7.0
    )

    features = (
        pd.DataFrame(feat)
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .astype("float32")
    )

    out = pd.concat(
        [
            df[
                [
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            ].copy(),
            pd.DataFrame(
                {
                    "decision_time": (
                        decision_time
                    )
                }
            ),
            pd.DataFrame(
                {
                    "current_atr": (
                        atr.astype(
                            "float32"
                        )
                    )
                }
            ),
            features,
        ],
        axis=1,
    )

    return out


def precision_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:
    excluded = {
        "time",
        "decision_time",
        "open",
        "high",
        "low",
        "close",
        "current_atr",
        "entry_open",
        "exit_close",
        "future_move_atr",
        "target_class",
    }

    return [
        c
        for c in frame.columns
        if c not in excluded
    ]
