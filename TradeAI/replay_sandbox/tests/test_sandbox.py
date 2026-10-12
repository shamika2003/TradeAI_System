"""Run: python -m unittest discover -s TradeAI/replay_sandbox/tests -v"""
import csv
from datetime import datetime, timedelta, timezone
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from TradeAI.replay_sandbox.broker import PaperBroker, Signal, Position
from TradeAI.replay_sandbox.data import Candle, MT5Archive
from TradeAI.replay_sandbox.runner import run_replay

START = datetime(2026, 9, 30, 0, 0, tzinfo=timezone.utc)
PROFILE = {"account_currency": "USD", "captured_at_utc": "2026-09-21T00:00:00+00:00", "symbols": {
    "EURUSD": {"point": 0.00001, "digits": 5, "slippage_pips": 0, "commission_per_lot": 7,
               "volume_min": 0.01, "volume_max": 500, "volume_step": 0.01},
    "USDJPY": {"point": 0.001, "digits": 3, "slippage_pips": 0, "commission_per_lot": 7,
               "volume_min": 0.01, "volume_max": 500, "volume_step": 0.01},
}}


def candle(index=0, o=1.10000, high=None, low=None, close=None, spread=2):
    return Candle(START + timedelta(minutes=5 * index), "EURUSD", "M5", o,
                  high if high is not None else o + .0001,
                  low if low is not None else o - .0001,
                  close if close is not None else o, spread)


class BrokerTests(unittest.TestCase):
    def test_buy_fills_at_ask_and_close_at_bid_with_commission(self):
        b = PaperBroker(PROFILE, initial_balance=150)
        c = candle()
        b.update_mark(c, at_open=True)
        b.execute(c, Signal("BUY", .01, 15, 25))
        self.assertIn("EURUSD", b.positions)
        self.assertAlmostEqual(b.positions["EURUSD"].entry_price, 1.10002)
        self.assertAlmostEqual(b.balance, 149.965)
        # Flat market exit takes the spread plus $0.07 round-turn commission.
        b._close("EURUSD", c.ends_at, 1.10000, 1.10002, "TEST")
        self.assertAlmostEqual(b.trades[0]["net_pnl_usd"], -0.09, places=5)
        self.assertAlmostEqual(b.balance, 149.91, places=5)

    def test_stop_wins_if_bar_touches_both(self):
        b = PaperBroker(PROFILE)
        c = candle()
        b.update_mark(c, at_open=True)
        b.execute(c, Signal("BUY", .01, 15, 25))
        b.check_range(candle(1, o=1.100, high=1.105, low=1.095))
        self.assertEqual(b.trades[0]["close_reason"], "SL_BOTH_TOUCHED")
        self.assertLess(b.trades[0]["net_pnl_usd"], 0)

    def test_gap_stop_not_optimistic_fill(self):
        b = PaperBroker(PROFILE)
        c = candle()
        b.update_mark(c, at_open=True)
        b.execute(c, Signal("BUY", .01, 15, 25))
        d = candle(1, o=1.096, high=1.0961, low=1.095)
        b.check_open_gap(d)
        self.assertEqual(b.trades[0]["close_reason"], "SL_GAP")
        self.assertAlmostEqual(b.trades[0]["exit_price"], 1.096)

    def test_margin_guards_small_account(self):
        b = PaperBroker(PROFILE, initial_balance=5, leverage=100)
        c = candle()
        b.update_mark(c, at_open=True)
        b.execute(c, Signal("BUY", 0.01, 15, 25))
        self.assertEqual(len(b.positions), 0)
        self.assertIn("insufficient", b.rejections[0]["reason"])

    def test_dynamic_jpy_conversion(self):
        b = PaperBroker(PROFILE)
        p = Position("USDJPY", "BUY", .01, START, 150, 149, 152, 0.035, 10)
        self.assertAlmostEqual(b.pnl(p, 151), 1000 / 151, places=6)

    def test_stop_checks_short_at_ask(self):
        b = PaperBroker(PROFILE)
        c = candle()
        b.update_mark(c, at_open=True)
        b.execute(c, Signal("SELL", .01, 15, 25))
        p = b.positions["EURUSD"]
        # The ASK HIGH crosses SL, even when BID HIGH is under it.
        high = p.sl - b.spread(c) / 2
        b.check_range(candle(1, o=1.10000, high=high, low=1.09999))
        self.assertEqual(b.trades[0]["close_reason"], "SL")


def make_archive(path: Path):
    fields = "time,symbol,timeframe,open,high,low,close,tick_volume,spread,real_volume\n"
    def rows(frame, count):
        text = fields
        for i in range(count):
            time = START + timedelta(minutes=(5 if frame == "M5" else 60) * i)
            v = 1.1 + i * .00001
            text += f"{time.isoformat(sep=' ')},EURUSD,{frame},{v},{v+.0001},{v-.0001},{v},100,2,0\n"
        return gzip.compress(text.encode())
    with ZipFile(path, "w") as z:
        z.writestr("broker_profile.json", json.dumps(PROFILE))
        z.writestr("direction_m5_raw\\EURUSD_M5.csv.gz", rows("M5", 45))
        z.writestr("direction_m5_raw\\EURUSD_H1.csv.gz", rows("H1", 5))


class RunnerTests(unittest.TestCase):
    def test_no_lookahead_decisions_at_next_candle_open(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.zip"
            make_archive(path)
            calls = []
            def decide(obs):
                self.assertLess(obs.completed_m5[-1].time, obs.decision_time)
                self.assertLessEqual(obs.completed_m5[-1].ends_at, obs.decision_time)
                self.assertTrue(all(c.ends_at <= obs.decision_time for c in obs.completed_h1))
                calls.append(obs.decision_time)
                if len(calls) == 1:
                    return Signal("BUY", .01, 15, 25)
                if len(calls) == 2:
                    return Signal("CLOSE")
                return None
            report = run_replay(
                data_zip=path, symbols=["EURUSD"], start=START + timedelta(minutes=20),
                end=START + timedelta(hours=4), strategy=decide,
                output_dir=Path(temp) / "results", exploratory=True, equity_every=1)
            self.assertEqual(report["entries_filled"], 1)
            self.assertEqual(report["trades_closed"], 1)
            self.assertEqual(report["evaluation_label"], "EXPLORATORY_NOT_OUT_OF_SAMPLE")
            with open(Path(temp) / "results" / "trades.csv", newline="") as f:
                trade = list(csv.DictReader(f))[0]
            self.assertEqual(trade["entry_time"], calls[0].isoformat())
            self.assertEqual(trade["exit_time"], calls[1].isoformat())

    def test_reject_start_before_frozen_model(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "raw.zip"
            make_archive(path)
            with self.assertRaisesRegex(ValueError, "precedes model freeze"):
                run_replay(data_zip=path, symbols=["EURUSD"], start=START,
                           end=START + timedelta(hours=2), strategy=lambda _: None,
                           output_dir=Path(temp) / "out", lock_time=START + timedelta(days=1))


if __name__ == "__main__":
    unittest.main()
