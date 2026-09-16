# filename: shared/tradeai_core/feature_engine.py

from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_schema import (
    FEATURE_NAMES,
    H1_BAR_MINUTES,
    M5_BAR_MINUTES,
)


class FeatureTransformer:
    """
    Canonical feature engine used by BOTH training and TradeAI.

    The multi-timeframe merge is point-in-time safe:
      * M5 rows are decision-ready at M5 open + 5 minutes.
      * H1 rows are usable only at H1 open + 60 minutes.
      * The latest H1 row available at the M5 decision time is selected.
    """

    def get_feature_list(self):
        return list(FEATURE_NAMES)

    @staticmethod
    def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "time" not in result.columns:
            raise RuntimeError("Feature input is missing 'time'")
        result["time"] = pd.to_datetime(result["time"], errors="coerce")
        return result

    def _technical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_time(df)
        df = df.sort_values("time").reset_index(drop=True)

        required = {"open", "high", "low", "close", "tick_volume"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise RuntimeError(f"Feature input missing columns: {missing}")

        if "spread" not in df.columns:
            df["spread"] = 0.0

        eps = 1e-9
        price = df["close"].replace(0, np.nan).astype(float)

        # Returns
        df["return"] = np.log(price / price.shift(1))

        # Candle range
        df["range"] = (df["high"] - df["low"]) / (price + eps)

        # RSI
        delta = price.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(span=14, adjust=False).mean()
        avg_loss = loss.ewm(span=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + eps)
        df["rsi"] = 100 - (100 / (1 + rs))

        # ATR
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - price.shift()).abs(),
                (df["low"] - price.shift()).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.ewm(span=14, adjust=False).mean()

        # Moving average trend
        df["ma5"] = price.rolling(5).mean()
        df["ma20"] = price.rolling(20).mean()
        df["trend"] = df["ma5"] - df["ma20"]

        # Volume
        df["volume_change"] = df["tick_volume"].pct_change().fillna(0)

        # ATR ratio
        df["atr_ratio"] = df["range"] / (df["atr"] + eps)

        # Position in recent range
        high20 = df["high"].rolling(20).max()
        low20 = df["low"].rolling(20).min()
        df["range_position"] = (
            (price - low20) / (high20 - low20 + eps)
        ).clip(0, 1)

        # Trend strength
        df["trend_strength"] = df["trend"] / (price + eps)

        # Momentum
        df["momentum_3"] = price - price.shift(3)
        df["momentum_10"] = price - price.shift(10)

        # Volatility
        df["volatility"] = df["return"].rolling(20).std()

        # Market structure
        df["structure_bias"] = (
            np.sign(df["ma5"] - df["ma20"]) * df["volatility"]
        )

        # Price z-score
        mean = price.rolling(1000).mean()
        std = price.rolling(1000).std()
        df["price_zscore"] = (price - mean) / (std + eps)

        # Volatility expansion
        df["volatility_change"] = (
            df["volatility"] / (df["volatility"].shift(10) + eps)
        )

        # Breakout distance
        high50 = df["high"].rolling(50).max()
        low50 = df["low"].rolling(50).min()
        df["dist_to_high_50"] = (high50 - price) / (price + eps)
        df["dist_to_low_50"] = (price - low50) / (price + eps)

        # Time encoding
        hour = df["time"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # Volatility regime
        vol = df["return"].rolling(50).std()
        df["volatility_regime"] = vol.rolling(200).rank(pct=True)

        return df

    @staticmethod
    def _alignment(m5_trend_strength, h1_trend_strength):
        return (
            np.sign(m5_trend_strength) == np.sign(h1_trend_strength)
        ).astype(np.int8)

    def build_multi_timeframe_features(
        self,
        df_m5: pd.DataFrame,
        df_h1: pd.DataFrame,
    ) -> pd.DataFrame:
        m5 = self._technical(df_m5)
        h1_full = self._technical(df_h1)

        # Decision timestamp: when the M5 candle has fully closed.
        m5["__decision_time"] = (
            m5["time"] + pd.Timedelta(minutes=M5_BAR_MINUTES)
        )

        # H1 features may not be used before the H1 candle has fully closed.
        h1 = h1_full[
            [
                "time",
                "ma20",
                "rsi",
                "trend_strength",
                "structure_bias",
            ]
        ].copy()

        h1.rename(
            columns={
                "time": "h1_source_time",
                "ma20": "h1_ma20",
                "rsi": "h1_rsi",
                "trend_strength": "h1_trend_strength",
                "structure_bias": "h1_structure_bias",
            },
            inplace=True,
        )

        h1["__h1_available_time"] = (
            h1["h1_source_time"] + pd.Timedelta(minutes=H1_BAR_MINUTES)
        )

        h1["h1_trend_bias"] = np.sign(h1["h1_trend_strength"])

        m5 = m5.sort_values("__decision_time").reset_index(drop=True)
        h1 = h1.sort_values("__h1_available_time").reset_index(drop=True)

        df = pd.merge_asof(
            m5,
            h1,
            left_on="__decision_time",
            right_on="__h1_available_time",
            direction="backward",
            allow_exact_matches=True,
        )

        # Canonical semantics: bullish+bullish OR bearish+bearish are aligned.
        df["m5_h1_alignment"] = self._alignment(
            df["trend_strength"],
            df["h1_trend_strength"],
        )

        self.assert_causal_alignment(df)

        df.drop(
            columns=["__decision_time", "__h1_available_time"],
            inplace=True,
            errors="ignore",
        )

        return self._final_clean(df)

    @staticmethod
    def assert_causal_alignment(df: pd.DataFrame) -> None:
        """Fail immediately if any M5 row uses an H1 candle from the future."""
        if "h1_source_time" not in df.columns:
            return

        valid = df["h1_source_time"].notna() & df["time"].notna()
        if not valid.any():
            return

        h1_available = (
            pd.to_datetime(df.loc[valid, "h1_source_time"])
            + pd.Timedelta(minutes=H1_BAR_MINUTES)
        )
        m5_decision = (
            pd.to_datetime(df.loc[valid, "time"])
            + pd.Timedelta(minutes=M5_BAR_MINUTES)
        )

        if (h1_available > m5_decision).any():
            raise RuntimeError(
                "CAUSALITY VIOLATION: future H1 information reached an M5 feature row"
            )

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "symbol" in df.columns:
            parts = []
            for _, group in df.groupby("symbol", sort=False):
                parts.append(self._technical(group))
            df = pd.concat(parts, ignore_index=True) if parts else df
        else:
            df = self._technical(df)

        return self._final_clean(df)

    # Kept only for compatibility with older callers. New training code uses
    # target_definition.py so target semantics are versioned independently.
    def add_target(self, df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
        result = df.copy()
        future = result["close"].shift(-horizon)
        result["target"] = np.tanh(
            (future - result["close"]) / result["close"] * 100
        )
        result.dropna(inplace=True)
        return result

    def _final_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
