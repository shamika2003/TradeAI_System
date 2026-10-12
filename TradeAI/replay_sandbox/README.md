# TradeAI Market Replay Sandbox (v1)

**Standalone TradeAI-owned trading-research module.** Does not edit the existing demo/live engine, trading rules, model files, TradeAI dashboard, NIRA, or the ELVARA bridge. No external packages needed. The replay code has **no MT5 import or broker-order API**.

## What it actually does

Streams historical **M5 OHLC** directly from `TradeAI_Model_Training_Data.zip` (no extraction required), with completed **H1** history available to a strategy callback. The four exported symbols are EURUSD, GBPUSD, USDJPY and USDCNH. A strategy sees only candles that have already *closed*. Its signal is filled at the **next M5 candle open**, with an isolated paper-broker ledger.

Execution simulation includes bid/ask spread from each historical bar's `spread` points, adverse slippage, round-trip commission, per-symbol minimum/maximum volume/step, one open position per symbol, maximum open positions, maximum lots, margin checks, leverage, buy/sell/reverse/close, SL/TP, forced closure, loss-aware liquidation and per-candle equity/drawdown calculations. When SL and TP fall inside one candle, **SL is assumed first**. Stop gaps fill at the worse market opening price. Both cost multipliers can be increased for stress tests.

**This is still a form of historical backtesting.** It is more comparable to a paper account than naive fixed-return backtests, but will not reproduce actual tick order, liquidity, broker fills, latency or intrabar spread changes. This is NOT 100%-accurate live trading.

## Install (Windows 11)

1. Find your `TradeAI_System` project folder (the folder containing `TradeAI/`).
2. Unzip `TradeAI_Market_Replay_Sandbox_PATCH.zip` **into that project root**. It adds only `TradeAI/replay_sandbox/` and does not overwrite your existing files.
3. Keep your existing `TradeAI_Model_Training_Data.zip` somewhere on your disk. You do not need to extract the data ZIP.
4. Open PowerShell in `TradeAI_System` (your project root).

Replace the ZIP path with its **actual** location. This command only verifies the archive:

```powershell
python -m TradeAI.replay_sandbox inspect --data "C:\path\to\TradeAI_Model_Training_Data.zip"
```

## Run the FIRST test (example strategy, not TradeAI!)

**IMPORTANT:** This command uses a simple moving-average demonstration strategy and deliberately labels the output **EXPLORATORY_NOT_OUT_OF_SAMPLE**. Its profit/loss says nothing about the trained TradeAI model.

```powershell
python -m TradeAI.replay_sandbox run `
  --data "C:\path\to\TradeAI_Model_Training_Data.zip" `
  --from 2026-09-22 --to 2026-10-05 `
  --symbols EURUSD GBPUSD USDJPY USDCNH `
  --strategy TradeAI.replay_sandbox.examples.demo_strategy:decide `
  --balance 150 --leverage 100 `
  --exploratory `
  --output artifacts/replay_sandbox/example
```

Reports in the output directory:

- `summary.json` — end balance, net profit, win rate, maximum drawdown, profit factor, commission, rejects, simulation assumptions, data/strategy hashes and date provenance.
- `trades.csv` — every closed simulated trade, including closing reason and net P&L.
- `equity.csv` — balance, mark-to-market equity, used margin and number of positions. Maximum drawdown is measured **every active candle**, independent of CSV sampling.
- `rejections.csv` — rejected signals with reasons.
- `events.csv` — opening/closing event audit log.

**Costs under stress:** re-run to a different output directory and add `--spread-multiplier 2 --slippage-multiplier 3` to double spreads and triple modeled slippage. These are **hypothetical** stress assumptions, not observed historical fills.

### Use $20 or $150 virtual capital

Change `--balance 150` to `--balance 20`. Lot sizes are never silently scaled down. Minimum volumes and margin can block entry into positions for a $20 account, as they should. The default 100:1 leverage is **ASSUMED** because account leverage is not included in the provided broker profile. Set it to your actual demo leverage.

## Connect the REAL frozen TradeAI model (pending source)

The example strategy is *not* the user's trained model. The provided data ZIP contains only market data plus broker settings, **not** your trained model artifact, feature builder, frozen threshold/policy, manager, or current demo/live engine source.

To connect the real system, implement a new callback in a **separate** TradeAI-owned Python file; do not change the sandbox broker. The callable signature is:

```python
from TradeAI.replay_sandbox.broker import Signal

def decide(observation):
    # observation.symbol, .decision_time
    # observation.completed_m5, .completed_h1 (ONLY fully closed bars)
    # observation.existing_position, .balance_usd, .equity_usd
    # Implement using VERIFIED frozen original TradeAI model, features,
    # entry/manager rules, with as-of-time feature engineering.
    # Return Signal('BUY', .01, sl_pips=15, tp_pips=25) or None
    raise NotImplementedError
```

`examples/frozen_model_adapter_TEMPLATE.py` is only a checklist/template. **Do not treat it as a working model adapter.** To implement properly, supply the latest `TRADEAI_SYSTEM_FULL_CODE.md` (or source ZIP), frozen model/feature metadata, model-freeze timestamp, exact demo entry/manager logic, and current risk/configuration. We should reuse those original components rather than reimplementing or silently changing your strategy.

Once an actual adapter exists, select a **strictly post-freeze** period and specify the freeze timestamp (**UTC with offset**, no guessing), for example:

```powershell
python -m TradeAI.replay_sandbox run `
  --data "C:\path\to\TradeAI_Model_Training_Data.zip" `
  --from YYYY-MM-DD --to YYYY-MM-DD `
  --strategy your_verified_tradeai_adapter:decide `
  --lock-time YYYY-MM-DDTHH:MM:SS+00:00 `
  --output artifacts/replay_sandbox/frozen_model
```

The timestamp above is deliberately a placeholder. The program refuses non-exploratory runs without an explicit freeze time, and rejects any replay start earlier than it. The time is **user-declared, not authenticated** against the actual model artifact yet.

## Key accuracy limitations / assumptions

1. **Data:** the ZIP contains M5 and H1 **candle** bars, *not* historical bid/ask ticks. MT5 candle OHLC is interpreted as **bid**; `spread` is read as **points**, converted with the broker's symbol `point`. One spread is held constant throughout a bar.
2. **SL/TP:** high/low have no intrabar order. The conservative SL-first resolution and modeled gap execution are estimates. Real SL/TP and margin liquidation can occur within a candle at different prices.
3. **Costs:** `commission_per_lot` is assumed to mean **round-trip USD commission**, so half is charged on entry and half on exit. Verify with your broker. The exported profile provides snapshot slippage assumptions, not historical realized slippage.
4. **Account:** snapshot lacks account leverage, broker stop-out percent and contract size. Defaults: 100:1 leverage, 50% stop-out, 100,000 FX units/lot. USD P&L for USDJPY/USDCNH uses *current closing price* for FX conversion, not the single snapshot tick value.
5. **Execution:** no liquidity, queue/depth, partial fills, overnight financing/swaps, broker trading hours, variable commission tier, stop-level overrides, portfolio currency hedging, news execution, or true latency model. The replay engine does **not** execute against MT5's actual broker trade server.
6. **Data provenance:** the user-provided M5/H1 rows extend into October 5, 2026; historical evaluation is valid as a holdout **only if** the feature-building, model-fitting, threshold selection, entry tuning and manager selection never used that future period.
7. **Audit:** data and strategy source SHA-256 hashes are recorded. Hashes do **not** establish the true training-freeze date or make an unverified adapter safe.
8. **Reporting:** mark-to-market drawdown is calculated at each M5 **bar close**, not on intra-bar equity minima; gaps and market spread spikes could create worse losses.

To improve fill fidelity, the next version should ingest native **MT5 historical tick bid/ask** data for execution while keeping feature calculations strictly causal. This dataset does not contain those ticks.

## Development sanity check

From the project root:

```powershell
python -m unittest discover -s TradeAI/replay_sandbox/tests -v
```

The included suite tests spread/commission accounting, stop ambiguity, gap stops, margin enforcement, short exits at ask, quote currency conversion, no-lookahead signal timing and post-freeze range validation.

**Ownership:** Only new files in `TradeAI/replay_sandbox/` are in this patch. No changes are made to TradeAI's existing live/demo engines or to NIRA/ELVARA dashboard and bridge project files.
