"""Offline tests: no terminal/order access required. Run from TradeAI_System root."""
import importlib
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace, ModuleType
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd

from TradeAI.replay_sandbox.collect_fresh_mt5 import (
    _csv_gz, detect_offset, model_creation_cutoff, prepare_frame, collect,
)

UTC = timezone.utc
SYMS = ("EURUSD", "GBPUSD", "USDJPY", "USDCNH")


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.now(UTC).replace(microsecond=0)
        self.cutoff = self.now - timedelta(days=1)

    def test_correct_three_hour_offset(self):
        ticks = {s: SimpleNamespace(time=int((self.now + timedelta(hours=3, seconds=-6)).timestamp())) for s in SYMS}
        self.assertEqual(detect_offset(ticks, self.now), 3)

    def test_reject_stale_quote(self):
        ticks = {s: SimpleNamespace(time=int((self.now + timedelta(hours=3, minutes=-20)).timestamp())) for s in SYMS}
        with self.assertRaisesRegex(RuntimeError, "inconsistent"):
            detect_offset(ticks, self.now)

    def test_reject_conflicting_clocks(self):
        ticks = {s: SimpleNamespace(time=int((self.now + timedelta(hours=3)).timestamp())) for s in SYMS}
        ticks["EURUSD"] = SimpleNamespace(time=int((self.now + timedelta(hours=2)).timestamp()))
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            detect_offset(ticks, self.now)

    def frame(self, sym="EURUSD", tf="M5"):
        raw = [self.now - timedelta(days=35) + timedelta(hours=3),
               self.now - timedelta(minutes=120 if tf == "H1" else 15) + timedelta(hours=3)]
        return pd.DataFrame({
            "time": pd.to_datetime(raw, utc=True), "symbol": [sym, sym], "timeframe": [tf, tf],
            "open": [1.01, 1.02], "high": [1.03, 1.03], "low": [1.00, 1.01],
            "close": [1.02, 1.02], "spread": [3, 3], "tick_volume": [150, 160],
            "real_volume": [0, 0],
        })

    def test_retains_raw_clock_and_manifest_utc(self):
        f, meta = prepare_frame(self.frame(), symbol="EURUSD", timeframe="M5", offset_hours=3,
                                now_utc=self.now, cutoff_utc=self.cutoff)
        self.assertEqual(meta["count"], 2)
        self.assertEqual(meta["last_corrected_utc"], (self.now - timedelta(minutes=15)).isoformat())
        self.assertIn("+00:00", meta["last_mt5_broker_clock_encoded_as_utc"])
        self.assertTrue(_csv_gz(f))

    def test_reject_forming_candle(self):
        f = self.frame()
        f.loc[1, "time"] = pd.Timestamp(self.now + timedelta(hours=3))
        with self.assertRaisesRegex(RuntimeError, "forming/future"):
            prepare_frame(f, symbol="EURUSD", timeframe="M5", offset_hours=3,
                          now_utc=self.now, cutoff_utc=self.cutoff)

    def test_reject_insufficient_warmup(self):
        f = self.frame()
        f.loc[0, "time"] = pd.Timestamp(self.now - timedelta(days=7, hours=-3))
        with self.assertRaisesRegex(RuntimeError, "warmup"):
            prepare_frame(f, symbol="EURUSD", timeframe="M5", offset_hours=3,
                          now_utc=self.now, cutoff_utc=self.cutoff)

    def test_archive_and_models_roundtrip_without_real_mt5(self):
        from TradeAI.replay_sandbox.data import MT5Archive
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "original.zip"
            with ZipFile(data, "w") as z:
                z.writestr("broker_profile.json", json.dumps({
                    "account_currency": "USD", "symbols": {s: {"volume_min": 0.01} for s in SYMS}
                }))
            model_root = root / "models"
            for s in SYMS:
                (model_root / s).mkdir(parents=True)
                (model_root / s / "model.json").write_text("{}")
                (model_root / s / "metadata.json").write_text(json.dumps({"created_utc": self.cutoff.isoformat()}))
            self.assertEqual(model_creation_cutoff(model_root)[0], self.cutoff)
            class FakeCollector:
                def connect(self): pass
                def disconnect(self): pass
                def _ensure_symbol(self, _): pass
                def fetch_recent_closed(self, sym, tf, n):
                    return self_outer.frame(sym, tf)
            self_outer = self
            mockmt5 = ModuleType("MetaTrader5")
            mockmt5.symbol_info_tick = lambda s: SimpleNamespace(time=int((self.now + timedelta(hours=3, seconds=-6)).timestamp()))
            mockcollector = ModuleType("Trade_Bot_Training.data_collector")
            mockcollector.MarketDataCollector = FakeCollector
            output = root / "forward.zip"
            with patch.dict(sys.modules, {"MetaTrader5": mockmt5, "Trade_Bot_Training.data_collector": mockcollector}):
                result = collect(data_zip=data, output=output, model_root=model_root, m5_count=15000, h1_count=1500)
            self.assertEqual(result["broker_offset_hours_observed_at_capture"], 3)
            self.assertTrue(output.is_file())
            archive = MT5Archive(output)
            self.assertEqual(sorted(archive.symbols), sorted(SYMS))
            self.assertEqual(len(list(archive.candles("EURUSD", "M5"))), 2)
            with ZipFile(output) as z:
                manifest = json.loads(z.read("collection_manifest.json"))
                self.assertIn("clock_warning", manifest)
            with self.assertRaises(FileExistsError):
                collect(data_zip=data, output=output, model_root=model_root, m5_count=15000, h1_count=1500)


if __name__ == "__main__":
    unittest.main()
