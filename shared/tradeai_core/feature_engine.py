from __future__ import annotations

import numpy as np
import pandas as pd

from .feature_schema import FEATURE_NAMES, H1_BAR_MINUTES, M5_BAR_MINUTES
from .setup_engine import add_structural_setup_features


class FeatureTransformer:
    """Canonical causal feature engine for the directional opportunity model."""

    def get_feature_list(self):
        return list(FEATURE_NAMES)

    @staticmethod
    def _ensure_time(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "time" not in out.columns:
            raise RuntimeError("Feature input is missing 'time'")
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
        return out

    @staticmethod
    def _adx(high, low, close, period=14):
        eps = 1e-12
        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0), 0.0)
        minus_dm = down.where((down > up) & (down > 0), 0.0)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + eps)
        minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / (atr + eps)
        dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + eps)
        adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
        return adx, plus_di, minus_di, atr

    def _technical(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._ensure_time(df).sort_values("time").reset_index(drop=True)
        required = {"open", "high", "low", "close", "tick_volume"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise RuntimeError(f"Feature input missing columns: {missing}")
        if "spread" not in df.columns:
            df["spread"] = 0.0

        eps = 1e-12
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float).replace(0, np.nan)
        v = df["tick_volume"].astype(float)

        log_c = np.log(c)
        df["return_1"] = log_c.diff(1)
        df["return_3"] = log_c.diff(3)
        df["return_6"] = log_c.diff(6)
        df["return_12"] = log_c.diff(12)
        df["return_24"] = log_c.diff(24)
        df["return_48"] = log_c.diff(48)
        df["return_96"] = log_c.diff(96)

        candle_range = (h - l).clip(lower=0)
        body = (c - o).abs()
        upper = (h - pd.concat([o, c], axis=1).max(axis=1)).clip(lower=0)
        lower = (pd.concat([o, c], axis=1).min(axis=1) - l).clip(lower=0)
        df["range_pct"] = candle_range / (c.abs() + eps)
        df["body_pct"] = body / (candle_range + eps)
        df["upper_wick_pct"] = upper / (candle_range + eps)
        df["lower_wick_pct"] = lower / (candle_range + eps)
        df["close_location"] = ((c - l) / (candle_range + eps)).clip(0.0, 1.0)
        df["close_location_mean_6"] = df["close_location"].rolling(6).mean()

        delta = c.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        rs = avg_gain / (avg_loss + eps)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        df["rsi14"] = rsi
        df["rsi_slope_3"] = rsi.diff(3) / 100.0

        adx, plus_di, minus_di, atr = self._adx(h, l, c, 14)
        df["atr"] = atr
        df["atr_pct"] = atr / (c.abs() + eps)
        df["atr_ratio"] = candle_range / (atr + eps)
        df["atr_rank_200"] = df["atr_pct"].rolling(200).rank(pct=True)
        df["signed_body_atr"] = (c - o) / (atr + eps)
        df["wick_imbalance"] = (lower - upper) / (candle_range + eps)
        df["range_expansion_20"] = candle_range / (candle_range.rolling(20).mean() + eps)
        df["adx14"] = adx / 100.0
        df["plus_di14"] = plus_di / 100.0
        df["minus_di14"] = minus_di / 100.0
        df["di_spread14"] = (plus_di - minus_di) / 100.0

        ema8 = c.ewm(span=8, adjust=False).mean()
        ema21 = c.ewm(span=21, adjust=False).mean()
        ema55 = c.ewm(span=55, adjust=False).mean()
        df["ema8_gap_atr"] = (c - ema8) / (atr + eps)
        df["ema21_gap_atr"] = (c - ema21) / (atr + eps)
        df["ema55_gap_atr"] = (c - ema55) / (atr + eps)
        df["ema8_21_atr"] = (ema8 - ema21) / (atr + eps)
        df["ema21_55_atr"] = (ema21 - ema55) / (atr + eps)
        df["trend_slope_12_atr"] = (ema21 - ema21.shift(12)) / (atr + eps)
        df["ema21_slope_6_atr"] = (ema21 - ema21.shift(6)) / (atr + eps)
        df["ema55_slope_24_atr"] = (ema55 - ema55.shift(24)) / (atr + eps)
        df["momentum_3_atr"] = (c - c.shift(3)) / (atr + eps)
        df["momentum_12_atr"] = (c - c.shift(12)) / (atr + eps)
        df["momentum_24_atr"] = (c - c.shift(24)) / (atr + eps)
        df["momentum_48_atr"] = (c - c.shift(48)) / (atr + eps)

        abs_delta = c.diff().abs()
        df["trend_efficiency_12"] = (c - c.shift(12)).abs() / (abs_delta.rolling(12).sum() + eps)
        df["trend_efficiency_48"] = (c - c.shift(48)).abs() / (abs_delta.rolling(48).sum() + eps)
        up = (c.diff() > 0).astype(float)
        df["up_fraction_12"] = up.rolling(12).mean()
        df["up_fraction_48"] = up.rolling(48).mean()

        vol5 = df["return_1"].rolling(5).std()
        vol20 = df["return_1"].rolling(20).std()
        vol60 = df["return_1"].rolling(60).std()
        vol100 = df["return_1"].rolling(100).std()
        df["volatility_5"] = vol5
        df["volatility_20"] = vol20
        df["volatility_60"] = vol60
        df["volatility_ratio"] = vol20 / (vol100 + eps)
        df["volatility_term_5_60"] = vol5 / (vol60 + eps)

        hi20 = h.rolling(20).max()
        lo20 = l.rolling(20).min()
        hi50 = h.rolling(50).max()
        lo50 = l.rolling(50).min()
        hi100 = h.rolling(100).max()
        lo100 = l.rolling(100).min()
        df["range_position_20"] = ((c - lo20) / (hi20 - lo20 + eps)).clip(0, 1)
        df["range_position_50"] = ((c - lo50) / (hi50 - lo50 + eps)).clip(0, 1)
        df["range_position_100"] = ((c - lo100) / (hi100 - lo100 + eps)).clip(0, 1)
        prev_hi20 = h.shift(1).rolling(20).max()
        prev_lo20 = l.shift(1).rolling(20).min()
        prev_hi50 = h.shift(1).rolling(50).max()
        prev_lo50 = l.shift(1).rolling(50).min()
        df["breakout_up_20_atr"] = (c - prev_hi20) / (atr + eps)
        df["breakout_down_20_atr"] = (prev_lo20 - c) / (atr + eps)
        df["breakout_up_50_atr"] = (c - prev_hi50) / (atr + eps)
        df["breakout_down_50_atr"] = (prev_lo50 - c) / (atr + eps)

        mean20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["bb_z20"] = (c - mean20) / (std20 + eps)
        df["bb_width20"] = (4.0 * std20) / (c.abs() + eps)

        v_mean = v.rolling(20).mean()
        v_std = v.rolling(20).std()
        df["volume_z20"] = (v - v_mean) / (v_std + eps)
        v_mean50 = v.rolling(50).mean()
        v_std50 = v.rolling(50).std()
        df["volume_z50"] = (v - v_mean50) / (v_std50 + eps)
        df["volume_change"] = v.pct_change().replace([np.inf, -np.inf], np.nan)
        spread = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0).clip(lower=0.0)
        spread_med = spread.rolling(100).median()
        df["spread_relative_100"] = spread / (spread_med + eps)

        hour = df["time"].dt.hour + df["time"].dt.minute / 60.0
        dow = df["time"].dt.dayofweek.astype(float)
        df["hour_sin"] = np.sin(2*np.pi*hour/24.0)
        df["hour_cos"] = np.cos(2*np.pi*hour/24.0)
        df["dow_sin"] = np.sin(2*np.pi*dow/7.0)
        df["dow_cos"] = np.cos(2*np.pi*dow/7.0)
        h_int = df["time"].dt.hour
        df["session_asia"] = ((h_int >= 0) & (h_int < 8)).astype(np.int8)
        df["session_london"] = ((h_int >= 7) & (h_int < 16)).astype(np.int8)
        df["session_newyork"] = ((h_int >= 12) & (h_int < 21)).astype(np.int8)
        df["session_overlap_london_ny"] = ((h_int >= 12) & (h_int < 16)).astype(np.int8)

        return df

    @staticmethod
    def _alignment(m5_trend_strength, h1_trend_strength):
        return (np.sign(m5_trend_strength) == np.sign(h1_trend_strength)).astype(np.int8)

    def build_multi_timeframe_features(self, df_m5: pd.DataFrame, df_h1: pd.DataFrame) -> pd.DataFrame:
        m5 = self._technical(df_m5)
        h1_full = self._technical(df_h1)
        m5["__decision_time"] = m5["time"] + pd.Timedelta(minutes=M5_BAR_MINUTES)

        h1_cols = [
            "time", "rsi14", "rsi_slope_3", "adx14", "atr_pct",
            "ema8_21_atr", "ema21_55_atr", "trend_slope_12_atr", "momentum_12_atr",
            "volatility_ratio", "bb_z20", "di_spread14", "range_position_20", "range_position_50",
        ]
        h1 = h1_full[h1_cols].copy().rename(columns={
            "time": "h1_source_time",
            "rsi14": "h1_rsi14",
            "rsi_slope_3": "h1_rsi_slope_3",
            "adx14": "h1_adx14",
            "atr_pct": "h1_atr_pct",
            "ema8_21_atr": "h1_ema8_21_atr",
            "ema21_55_atr": "h1_ema21_55_atr",
            "trend_slope_12_atr": "h1_trend_slope_12_atr",
            "momentum_12_atr": "h1_momentum_12_atr",
            "volatility_ratio": "h1_volatility_ratio",
            "bb_z20": "h1_bb_z20",
            "di_spread14": "h1_di_spread14",
            "range_position_20": "h1_range_position_20",
            "range_position_50": "h1_range_position_50",
        })
        h1["__h1_available_time"] = h1["h1_source_time"] + pd.Timedelta(minutes=H1_BAR_MINUTES)

        m5 = m5.sort_values("__decision_time").reset_index(drop=True)
        h1 = h1.sort_values("__h1_available_time").reset_index(drop=True)
        df = pd.merge_asof(
            m5, h1,
            left_on="__decision_time", right_on="__h1_available_time",
            direction="backward", allow_exact_matches=True,
        )
        df["m5_h1_alignment"] = self._alignment(df["ema21_55_atr"], df["h1_ema21_55_atr"])
        self.assert_causal_alignment(df)
        df = add_structural_setup_features(df)
        df.drop(columns=["__decision_time", "__h1_available_time"], inplace=True, errors="ignore")
        return self._final_clean(df)

    @staticmethod
    def assert_causal_alignment(df: pd.DataFrame) -> None:
        if "h1_source_time" not in df.columns:
            return
        valid = df["h1_source_time"].notna() & df["time"].notna()
        if not valid.any():
            return
        h1_available = pd.to_datetime(df.loc[valid, "h1_source_time"]) + pd.Timedelta(minutes=H1_BAR_MINUTES)
        m5_decision = pd.to_datetime(df.loc[valid, "time"]) + pd.Timedelta(minutes=M5_BAR_MINUTES)
        if (h1_available > m5_decision).any():
            raise RuntimeError("CAUSALITY VIOLATION: future H1 information reached an M5 feature row")

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "symbol" in df.columns:
            parts = [self._technical(g) for _, g in df.groupby("symbol", sort=False)]
            out = pd.concat(parts, ignore_index=True) if parts else df.copy()
        else:
            out = self._technical(df)
        return self._final_clean(out)

    def add_target(self, df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
        out = df.copy()
        future = out["close"].shift(-horizon)
        out["target"] = np.tanh((future - out["close"]) / out["close"] * 100)
        return out.dropna().reset_index(drop=True)

    def _final_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.loc[:, ~df.columns.duplicated()].copy()
        out.replace([np.inf, -np.inf], np.nan, inplace=True)
        required = [c for c in FEATURE_NAMES if c in out.columns]
        if required:
            out.dropna(subset=required, inplace=True)
        out.reset_index(drop=True, inplace=True)
        return out
