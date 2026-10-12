"""Run: python -m unittest TradeAI.replay_sandbox.tests.test_precision_replay"""
from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from Trade_Bot_Training.precision_feature_engine import build_precision_features
from TradeAI.replay_sandbox.broker import Position
from TradeAI.replay_sandbox.data import Candle
from TradeAI.replay_sandbox.precision_replay import PrecisionPaperBroker, _prepare_decisions

UTC = timezone.utc
T0 = datetime(2026, 10, 1, 10, 0, tzinfo=UTC)
PROFILE = {"symbols": {"EURUSD": {"point": 1e-5, "digits": 5,
    "volume_min": 0.01, "volume_max": 500., "volume_step": 0.01,
    "slippage_pips": 0., "commission_per_lot": 7.}}}


def bar(i, *, o=1.1000, close=1.1001, spread=10):
    return Candle(T0 + timedelta(minutes=5*i), "EURUSD", "M5", o,
                  max(o, close)+0.0001, min(o, close)-0.0001, close, spread, 100)


class PrecisionReplayTests(unittest.TestCase):
    def test_horizon_exactly_six_bars_no_sl_tp(self):
        b = PrecisionPaperBroker(PROFILE, initial_balance=150, leverage=100)
        candles = [bar(i) for i in range(6)]
        b.update_mark(candles[0], at_open=True)
        self.assertTrue(b.open_fixed(candles[0], "BUY", "unit test"))
        self.assertIsNone(b.positions["EURUSD"].sl)
        self.assertIsNone(b.positions["EURUSD"].tp)
        for c in candles[:5]:
            b.update_mark(c, at_open=False)
            b.close_due(c, at_open=False)
            self.assertEqual(len(b.positions), 1)
        b.update_mark(candles[5], at_open=False)
        b.close_due(candles[5], at_open=False)
        self.assertFalse(b.positions)
        self.assertEqual(b.trades[0]["close_reason"], "MODEL_HORIZON_30M")
        self.assertAlmostEqual(b.balance-150, b.trades[0]["net_pnl_usd"], places=6)

    def test_no_stacking_and_minimum_lot(self):
        b = PrecisionPaperBroker(PROFILE, initial_balance=150, leverage=100)
        c = bar(0)
        b.update_mark(c, at_open=True)
        self.assertTrue(b.open_fixed(c, "BUY", ""))
        self.assertFalse(b.open_fixed(c, "SELL", ""))
        self.assertEqual(b.entry_count, 1)
        self.assertEqual(b.positions["EURUSD"].lots, 0.01)

    def test_margin_block_and_fee(self):
        b = PrecisionPaperBroker(PROFILE, initial_balance=1, leverage=100)
        c = bar(0)
        b.update_mark(c, at_open=True)
        self.assertFalse(b.open_fixed(c, "BUY", ""))
        self.assertEqual(b.rejections[-1]["reason"], "LOW_BALANCE_MARGIN_BLOCK")
        self.assertEqual(b.balance, 1)

    def test_spread_floor(self):
        b = PrecisionPaperBroker(PROFILE, spread_floor_pips=1.2)
        self.assertAlmostEqual(b.spread(bar(0, spread=0)) / b.pip("EURUSD"), 1.2)

    def test_missing_horizon_bar_exits_at_next_available_open(self):
        b = PrecisionPaperBroker(PROFILE)
        c = bar(0)
        b.update_mark(c, at_open=True)
        self.assertTrue(b.open_fixed(c, "BUY", ""))
        next_candle = bar(6)  # 10:30, last 10:25-10:30 bar is absent
        b.update_mark(next_candle, at_open=True)
        b.close_due(next_candle, at_open=True)
        self.assertEqual(b.trades[0]["close_reason"], "MODEL_HORIZON_GAP_AT_OPEN")

    def test_causal_feature_parity(self):
        bars = [bar(i, o=1.1 + i * .00001, close=1.10005 + i * .00001) for i in range(510)]
        def frame(rows):
            return pd.DataFrame({"time": [b.time for b in rows],
                 "open": [b.open for b in rows], "high": [b.high for b in rows],
                 "low": [b.low for b in rows], "close": [b.close for b in rows],
                 "tick_volume": [b.tick_volume for b in rows]})
        full = build_precision_features(frame(bars))
        prefix = build_precision_features(frame(bars[:490]))
        cols = [k for k in prefix.columns if k not in ("time", "decision_time")]
        np.testing.assert_allclose(full.loc[489, cols].to_numpy(dtype="float64"),
                                   prefix.loc[489, cols].to_numpy(dtype="float64"),
                                   rtol=1e-6, atol=1e-8, equal_nan=True)

    def test_signal_only_after_previous_candle(self):
        bars = [bar(i, o=1.1 + i*.00001, close=1.10005+i*.00001) for i in range(510)]
        feats = build_precision_features(pd.DataFrame({
            "time": [b.time for b in bars], "open": [b.open for b in bars],
            "high": [b.high for b in bars], "low": [b.low for b in bars],
            "close": [b.close for b in bars], "tick_volume": [b.tick_volume for b in bars]}))
        feature_cols = [c for c in feats.columns if c not in {"time", "decision_time", "open", "high", "low", "close", "current_atr"}]
        class FakeModel:
            symbol = "EURUSD"
            feature_columns = feature_cols
            def predict(self, frame):
                return pd.DataFrame({"signal": "UP", "tradeable": True, "historical_gate_pass": True,
                    "confidence": .8, "p_down": .1, "p_hold": .1, "p_up": .8}, index=frame.index)
        decision_time = bars[-1].ends_at
        decisions, _ = _prepare_decisions(bars, FakeModel(), bars[-5].time, decision_time+timedelta(seconds=1))
        self.assertIn(decision_time, decisions)
        self.assertEqual(decisions[decision_time]["bar_time"], bars[-1].time)
        self.assertNotIn(bars[-1].time, {t for t,d in decisions.items() if d["bar_time"] == bars[-1].time})


if __name__ == "__main__":
    unittest.main()
