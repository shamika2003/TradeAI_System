# TradeAI Precision M5 — Pre-2026 training / Jan–Sep 2026 replay

**Type:** Isolated retrospective research study, not a new demo/live trading model.
**Created with:** Existing `Trade_Bot_Training` precision feature, label, validation,
training and production gate algorithms copied byte-for-byte into a private
workspace, followed by a separate **final refit using all pre-2026 labelled
rows**, at the validation-selected tree count. No changes to `artifacts/models/precision_m5`, live/demo engines,
NIRA UI, or existing replay outputs.

## Model and data boundaries

| Use | Window |
| --- | --- |
| Allowed model-development input history | 2020–2025-12-31, inclusive |
| Label horizon | Existing next M5 open to the sixth following M5 close |
| Internal candidate training / validation / test | Existing 70% / 15% / 15% split **within pre-2026 data** |
| Final refit of selected model | **All** eligible pre-2026 labelled decisions (through late Dec 2025) |
| Never used for training or model gate selection | 2026-01-01 onward |
| Replay feature warm-up only | 2025-12-01 to 2025-12-31 |
| Retrospective out-of-training holdout | 2026-01-01 through 2026-09-30 |

The data may start in January 2020 depending on your broker's available
history. Label generation cannot reach into 2026 because the isolated input
files have **no 2026 rows**. There is a six-bar purge around the existing
train/validation boundary. All original production readiness gates are kept;
if a retrained model fails those gates, the replay **may show zero trades**.
That is expected and must not be silently bypassed.

**Provenance caveat:** This study runs in October 2026, so the Jan–Sep 2026
results are **retrospective out-of-sample**, not literal forward trading
performed at that time. Don't pick thresholds, architectures or symbol
subsets after seeing this holdout and then call it independent.

## Install

From your TradeAI_System repository root in PowerShell, extract the patch:

```powershell
Expand-Archive -Path ".\TradeAI_2025_Holdout_PATCH.zip" -DestinationPath "." -Force
```

Your existing raw source files should be present here:

```text
artifacts/datasets/direction_m5_raw/EURUSD_M5.csv.gz
artifacts/datasets/direction_m5_raw/EURUSD_H1.csv.gz
... same M5 and H1 files for GBPUSD, USDJPY, USDCNH ...
artifacts/reports/broker_profile.json
```

This workflow **does not require** the previously downloaded
`TradeAI_Model_Training_Data.zip`. It reads your existing raw files and never
changes them. If some files are missing, restore original files **before**
continuing. The collector does not retrieve new data or interact with MT5.

## 1. Prepare the dataset, holdout replay ZIP and training-code snapshot

```powershell
python -m TradeAI.replay_sandbox.holdout_2025 prepare
```

Full raw input is processed in chunks; depending on your disk and CPU this can
take a few minutes. This should create:

```text
artifacts/research/precision_cutoff_20251231/
  cutoff_manifest.json
  replay_2026_jan_sep.zip
  Trade_Bot_Training/                  # copied original trainer + isolated refit helper
  artifacts/datasets/direction_m5_raw/ # ONLY before 2026-01-01
```

If an existing workspace is detected, preparation **refuses** to overwrite it.

## 2. Train isolated models

```powershell
python -m TradeAI.replay_sandbox.holdout_2025 train
```

The original four-symbol training workflow is executed in a *separate Python
process*, with that process's working directory and module search path set to
this isolated workspace. This may take a while depending on your machine.
It first trains and evaluates four precision internal candidates using the original
0.55 action threshold, 70/15/15 splits, early stopping and historical production
gates. Then, with **no 2026 data or selection**, it refits four final models
on all pre-2026 labelled data using the validation-selected tree counts.

The historical production pass/fail gate and internal-test metrics describe
the **initial pre-refit candidate**, not the final refitted estimator. We retain
the gate as a conservative inherited eligibility rule, but it is not fresh
validation of the refitted model. The 2026 holdout is the first direct
evaluation of each final refitted estimator.

On success, the models are written only beneath:

```text
artifacts/research/precision_cutoff_20251231/artifacts/models/precision_m5/
```

`training_complete.json` records model hashes and blocks accidental retraining
of the same candidate. Existing production artifacts remain untouched.

## 3. Replay nine months on a $150 simulated account

```powershell
python -m TradeAI.replay_sandbox.holdout_2025 replay --balance 150 --spread-floor-pips 0.8
```

The replay uses the existing **read-only Precision M5 demo policy**:
next-M5-open entry, broker minimum lot, no stacking per pair, 3 position
portfolio cap, no invented SL/TP, fixed 30-minute exit, and the model's existing
historical pass/fail gates. All execution is simulated, with no MT5 orders.

Outputs:

```text
artifacts/research/precision_cutoff_20251231/replay_results_2026/
  summary.json
  trades.csv
  equity.csv
  rejections.csv
  events.csv
```

The replay report is explicitly marked as retrospective, and shows simulated
net profit, max drawdown, number of trades, rejection counts and more.
Assumed leverage/margin behavior, approximated spreads, intrabar execution
sequence, slippage and broker outages mean this will **never be 100% live
accuracy**. Do not use results to justify real-money deployment.

## Checks and tests

```powershell
python -m pytest TradeAI/replay_sandbox/tests/test_holdout_2025.py -q
```

Preflight preparation refuses to proceed when raw source has wrong symbol,
wrong timeframe, mixed/non-UTC timestamps, missing coverage through Sep 30,
non-monotone timestamps, or inadequate data. Hash checks prevent silently
altering the raw training snapshot, candidate models, replay data archive or
snapshotted training scripts between steps. The final full-data refit is
additional to the original training workflow and requires more compute time.

To intentionally run an independent NEW study rather than overwrite a locked
one, give all three commands the same new name via `--workspace`, for example
`--workspace ".\artifacts\research\precision_cutoff_20251231_trial2"`.
But treat **all** such trials on the 2026 holdout as exploratory experiments,
not independent model validation.

**Timestamp caveat:** The source CSVs contain `+00:00` timestamps as
exported by the existing MT5 data collector. This patch does not change
the recorded times or independently prove historical broker-clock offset
accuracy; a broker-clock offset would affect the exact boundary interpretation.
