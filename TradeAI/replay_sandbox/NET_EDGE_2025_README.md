# TradeAI Net Edge 2025 – Isolated Training Research

**Status:** Offline 2025 development only; NOT a ready-to-trade system and no MT5 integration. The 2026 results have been inspected, so a later 2026 replay cannot be claimed to be an independent final test.

## Installation

Extract `TradeAI_Net_Edge_2025_PATCH.zip` into `TradeAI_System` root. Contents only add new files under `TradeAI/replay_sandbox`, and do not overwrite existing files.

Run these commands **one at a time** in PowerShell:

```powershell
python -m TradeAI.replay_sandbox.net_edge_2025 prepare
python -m TradeAI.replay_sandbox.net_edge_2025 train
```

`prepare` checks the frozen 2025 cutoff, costs policy, broker profile, feature implementation, and training-label/sample distributions. It writes `artifacts/research/precision_cutoff_20251231/net_edge_2025/prepared.json`. It **does not train**.

`train` (only after prepare) uses Jan–Jul 2025 train, Aug–Sep validation, and Oct–Dec 2025 development test. It saves 3 separate research models under `.../net_edge_2025/models/{symbol}`, and a `training_report.json`. No existing Precision M5/MT5/demo/live model is modified. The CLI does **not offer** a 2026 replay or production promotion.

## Labels & costs

- The input is each *fully closed* M5 bar's causal 64 features, plus its completed-bar spread, built using the frozen 2025 snapshot.
- The target uses the next M5 bar open and sixth future M5 bar close, only for *training labels* and historical test scoring.
- Profitability labels are 0=SELL, 1=HOLD, 2=BUY. An action must exceed **+1.0 net pip** over estimated entry/exit spread, 0.2 pip slippage each side, and full commission at broker minimum lot; otherwise the target is HOLD.
- Entry uses archived *future entry bar* spread for the training BUY label; SELL exit uses the archived *future exit bar* spread in its label. Neither future spread is a predictive feature.
- Eligible opportunities must pass the **existing 2025-frozen completed-bar spread cap**, have contiguous 30-minute history, and keep entry and projected exit outside 23:00–00:59 in the archival timestamp clock.
- Original broker timestamps are treated as archived clock fields, **not independently verified wall-clock UTC**. Historical M5 spread is a proxy, not a guarantee of fillable live spread.

## Validation and deployment safeguards

- XGBoost 3-class classifier with early stopping on Aug–Sep validation data only.
- On validation only, assesses a **predeclared** confidence grid (0.40–0.65). A threshold is accepted only if >=30 non-overlapping validation trades show >=0.5 net pip average *after estimated costs*. Otherwise the model abstains from all 2025 test trades.
- That threshold, if found, is frozen before Oct–Dec testing. Tests report non-overlapping *per-symbol* trades, net pips and month-level results. Account-level margin, cross-pair contention, stop loss and gaps are **not simulated** by this experiment.
- No automatic threshold relaxation, no forced trades, no model promotion.
- If validation fails, that outcome is meaningful: it suggests lack of demonstrated predictive *net edge* for this architecture and 30-minute horizon. Consider further research on target/horizon/other data using proper chronological validation; do not tune on already-seen 2026 outcomes.

**Dependencies:** project's existing Python pandas/numpy/scikit-learn/xgboost stack. Unit tests: `python -m pytest -q TradeAI/replay_sandbox/tests/test_net_edge_2025.py` (pytest optional).

**Reproducibility:** No `2026` values are used to generate model features or training labels. Existing cutoff workspace is not overwritten; freeze hashes checked before prepare and train. The broker profile from the archived research snapshot is used.
