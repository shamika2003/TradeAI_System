# TradeAI Precision Model Market Replay (isolated add-on)

**Purpose:** replay the *real frozen Precision M5* TradeAI model over historical MT5 M5 candles. This is *not* the older `direction_m5` trading policy, nor the sandbox's example moving-average strategy. It executes **NO MT5 orders** and does not retrain or overwrite models, demo account state, the dashboard, or NIRA.

## Installation

1. Open your **TradeAI_System** directory in VS Code.
2. Extract the supplied `TradeAI_Precision_Model_Replay_PATCH.zip` into **the project root**. It installs two new files only (Python entry and tests) plus this README.
3. Leave `artifacts/models/precision_m5/{EURUSD,GBPUSD,USDJPY,USDCNH}/model.json` and `metadata.json` in place; the models are already on your computer. This ZIP does not contain model weights.
4. Keep `TradeAI_Model_Training_Data.zip` in the project root.
5. If needed install the project's existing model dependencies with `python -m pip install -r Trade_Bot_Training/requirements_model.txt`.
6. Run one of the commands below in **PowerShell**, copying **only** the single command line, never the `PS ...>` prompt or JSON output.

## Check tests

```powershell
python -m unittest discover -s TradeAI/replay_sandbox/tests -p "test_*.py" -v
```

## First model replay (explicitly exploratory; not forward validation)

```powershell
python -m TradeAI.replay_sandbox.precision_replay --data ".\TradeAI_Model_Training_Data.zip" --from 2026-09-22 --to 2026-10-05 --balance 150 --exploratory
```

## Stress run: impose minimum 0.8-pip spread on all four pairs

```powershell
python -m TradeAI.replay_sandbox.precision_replay --data ".\TradeAI_Model_Training_Data.zip" --from 2026-09-22 --to 2026-10-05 --balance 150 --spread-floor-pips 0.8 --exploratory
```

All outputs go to `artifacts/replay_sandbox/precision_demo_replay/`: `summary.json`, `trades.csv`, `equity.csv`, `events.csv`, `rejections.csv`. Running again in the same directory **overwrites replay reports only**. Use `--output` with another directory to compare runs. Never store credentials in these files.

## Exact connection to existing demo policy

- Original `Trade_Bot_Training.precision_feature_engine.build_precision_features` using **closed M5** bars only; vectorized precompute uses the same causal feature equations. It does not feed future prices into a decision.
- Original `Trade_Bot_Training.precision_model.PrecisionDirectionModel.predict` loads **existing** per-pair `model.json` and `metadata.json`; preserves action class map, 0.55 or metadata confidence threshold, `production_ready_by_historical_gate`. All artifacts hashed in report.
- BUY/SELL on **next contiguous M5 open** after signal-bar close. Skip HOLD/nontradeable. No reverse/stacking. Broker **minimum lot**, maximum **3 open positions**. No invented SL/TP/trailing stop.
- Close on **sixth subsequent M5 bar close**, a scheduled 30-minute model horizon; if the needed historical candle is missing, close at next available market open and label data-gap execution. Broker margin stopout is approximated from M5 close.
- Entry at **ask for BUY / bid for SELL** and opposite quote for exit, with per-candle spread, synthetic slippage, and commission.
- Account equity, balance, margin requirement and USD conversion calculated by the **existing** sandbox's `PaperBroker` methods. Default 100:1 leverage and 50% stopout are **assumptions**, not confirmed account settings.

## Known limitations / evaluation warnings

1. Original precision models were created on **2026-10-07 UTC**, while the supplied archive ends approximately **2026-10-05**. Therefore no replay from this archive can prove out-of-sample performance for those models; the final historical test period may have been used in model evaluation or selection. `--exploratory` is mandatory unless a separately verified post-lock dataset is provided.
2. The live demo's actual order fills happen via MT5 using the next available quotes; this replay uses **next candle open** after the completed decision candle, and **sixth-bar close** as the exit. These are reasonable approximations, *not identical brokerage execution*.
3. Candles cannot reconstruct intra-bar stops, precise spread fluctuations, slippage, broker rejection, swaps, server delay, or intrabar margin stop-out. This system models liquidation at M5 close. Nonconstant swap/rollover charges are **not** simulated.
4. The downloaded dataset contains per-candle spread values and a broker profile snapshot. Zero spreads can occur; use `--spread-floor-pips` for adverse cost assumptions.
5. Models are loaded from your installed `artifacts/models/precision_m5/`; never fit, retune or promote based on these results.
6. This patch intentionally does **not** touch `TradeAI/precision_demo_engine.py`, `TradeAI/precision_demo_engine_base.py`, `TradeAI/precision_live_engine.py`, NIRA or dashboard files.

The **next validation milestone** is to collect truly new market data *after* the last model freeze, run a strictly post-lock replay without altering model or thresholds, and compare its fills and signals to a contemporaneous MT5 demo run. Use `--lock-time` with independently documented actual freeze time; the simulator refuses a timestamp before any model's creation or a replay interval before the supplied lock.
