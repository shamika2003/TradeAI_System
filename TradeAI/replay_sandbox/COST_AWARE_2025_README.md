# TradeAI — 2025-Cutoff Cost-Aware Research Replay

Research only. **No MT5, no live/demo orders, no retraining, no model promotion, no NIRA changes.**

Built against `TRADEAI_SYSTEM_FULL_CODE(20261010-041532).md`.

## What is changed

- `precision_replay.py`: Adds **optional** `entry_policy` callback in the normal loop. With no policy, the old code path still runs. The callbacks receive the current M5 bar, the most recent **previously completed** M5 bar, and the paper broker. All rejections are logged as `COST_POLICY_*`.
- `cost_aware_2025.py`: Creates and freezes a rule set **only from the isolated pre-2026 study's 2025 M5 spread history** and then runs an exploratory replay of 2026. Does not change model weights, confidence thresholds, entries, exits, lots, leverage, or broker accounting.
- `tests/test_cost_aware_2025.py`: Adds offline causal-safety and replay-loop tests.

## Installation

Copy ZIP into existing `TradeAI_System` root and run:

```powershell
Expand-Archive -Path ".\TradeAI_Cost_Aware_2025_PATCH.zip" -DestinationPath "." -Force
python -m TradeAI.replay_sandbox.cost_aware_2025 prepare
python -m TradeAI.replay_sandbox.cost_aware_2025 replay --balance 150 --spread-floor-pips 0.8
```

## Policy fixed before the new replay

- Hours **23:00–00:59 in the archived raw timestamp clock** are excluded, both for entry time and for the expected 30-minute exit time. Do not silently convert these hours to PC-local or live broker-local timezone.
- A trade requires one **contiguous fully completed previous M5 candle** to obtain its spread. The next candle's recorded spread is **not** permitted as a pre-entry filter because it may represent information accumulated during the candle.
- For each symbol, the cap on last-closed spread is `min(95th percentile of 2025 nonblackout spreads, 10.0 pips)`. This 10-pip absolute research risk ceiling is **a fixed, conservative budget, not a statistically optimized trading threshold**. As of historical 2025 data, USDCNH's normal-hour median was around 37 pips, so it will likely be excluded.
- The execution simulator **still charges its original spread** from the entry and exit candles, along with original slippage and commission. Its bar-period spread is not verified to be the exact quote offered at the opening tick. No hypothetical reduction in broker spreads is assumed.

## Isolation safeguards

- Requires existing `artifacts/research/precision_cutoff_20251231` prepared and trained study.
- Rechecks all frozen research model / metadata SHA-256 fingerprints, original 2025 training dataset hashes, source snapshot hashes, and the 2026 reserve ZIP checksum.
- Rejects any isolated calibration file with 2026 rows.
- Never overwrites `cost_policy_2025.json` or `replay_results_2026_cost_aware/`; choose a new study/version if experimenting further.
- Leaves the original `replay_results_2026/` unchanged.

## How to interpret results

**Jan–Sep 2026 has already been inspected.** This rerun is exploratory, even though the new parameters were calculated only from 2025 data. The rule concept (avoid rollover) was motivated by seeing 2026 losses, and therefore this is **not a genuinely blind validation**. Even a higher simulated ending balance could reflect fewer or no trades, rather than demonstrated net-positive edge.

A later strictly post-freeze paper-trading interval with actual quotes (and then broker fills) is needed to assess profitability. Do not promote these research models or this rule set to live/demo trading merely because the replay improves.
