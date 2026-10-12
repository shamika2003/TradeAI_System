# TradeAI — safe post-model MT5 data capture

This patch **adds** `TradeAI/replay_sandbox/collect_fresh_mt5.py`. It does not edit
existing demo/live engines, model/threshold files, training code, or NIRA.

## Run (PowerShell, from TradeAI_System root)

    python -m TradeAI.replay_sandbox.collect_fresh_mt5 --data ".\TradeAI_Model_Training_Data.zip"

MT5 must be open and signed in. The script reads 15,000 closed M5 and 1,500
closed H1 candles per symbol through the existing `MarketDataCollector`.
It collects EURUSD, GBPUSD, USDJPY and USDCNH. The output is a **new**
`artifacts/replay_sandbox/mt5_fresh_broker_clock.zip` file, compatible with
`MT5Archive`. A second run will refuse overwriting it. Use `--output` with a
new name for subsequent captures.

### Critical timestamp issue

Your broker currently reports tick timestamps about **UTC+3** relative to your
PC clock. The current trained model's `tod_*`/`dow_*` features were trained on
MT5's raw clock. This collector preserves those raw timestamps and stores an
explicit broker offset plus corrected UTC date ranges in
`collection_manifest.json`. The archive's `time` values must **not** be
interpreted as true UTC until the replay strategy has been made clock-aware.

The collector checks for stale quote/clock discrepancies, incomplete candles,
missing minimum historical warmup and unexpected model files. It requires
30 days of pre-model history to provide causal warmup, but this is not a
forward training window: models are never refitted.

**The collector does not execute the replay.** A verified forward-replay wrapper
must enforce the real-UTC model cutoff, align evaluation timestamps, and audit
that no model/threshold selection used the evaluation period. Do **not** use
`--lock-time` on raw broker-clock timestamps as though they were true UTC.

Candle-only replay still cannot reproduce tick-level fills and real broker risk.
