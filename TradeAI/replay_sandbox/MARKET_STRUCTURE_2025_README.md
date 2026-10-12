# TradeAI H1/M5 Market Structure Research V1

This is a **simulation-only research extension** for the existing isolated
`precision_cutoff_20251231` workspace. It does not modify or promote any model,
policy, demo engine, live engine, dashboard, NIRA bridge, or MT5 terminal.

## Installation

Place `TradeAI_Market_Structure_2025_PATCH.zip` in the TradeAI_System project root.
In PowerShell:

```powershell
Expand-Archive -Path '.\TradeAI_Market_Structure_2025_PATCH.zip' -DestinationPath '.' -Force
```

Run tests (optional, recommended):

```powershell
python -m unittest TradeAI.replay_sandbox.tests.test_market_structure_2025 -v
```

Research on frozen pre-2026 candles with a $150 account:

```powershell
python -m TradeAI.replay_sandbox.market_structure_2025 run --balance 150
```

It writes **only** to a new folder:

`artifacts/research/precision_cutoff_20251231/market_structure_2025_v1/`

It refuses to overwrite existing results. A partial failure preserves `report.json`
for diagnosis. To rerun, move that new research directory to a different location
before executing, rather than blindly overwriting it.

## What changed and what did not

- Uses completed **H1 trend/structure, volatility and M5 price/volume** features.
  Decisions occur at full H1 close; entry fills at the **following M5 open**.
- Trains separate gradient-boosting regressors predicting **net realized R**
  for BUY and SELL, not just gross 30-minute direction.
- Target outcomes simulate BID OHLC and ASK using broker-point spreads, **0.8-pip
  minimum spread**, 0.2-pip-per-side broker slippage, and full round-trip commission.
  Stops win same-bar SL/TP ambiguity; adverse gaps fill at the worse open.
- Predeclared stop = 0.85x completed H1 ATR bounded to 8–35 pips;
  target = 1.8x stop distance; max hold = **120 minutes**.
- Excludes rollover hours `23`/`0` in the archived broker timestamp clock, including
  the projected two-hour hold. Enforces last fully closed M5 spread <=3 pips.
  The current entry candle spread is used **only to simulate fill and costs**, not
  to preselect signals.
- Research sim trades **0.01 lots maximum**, maximum planned 1.5% equity risk,
  3% closed-loss cap per day, 12% drawdown stop and a 100:1 assumed margin reserve.
  Check actual broker leverage, minimum stops, tick value and commissions separately.
- Reads only **2020–2025** frozen raw candles. 2023 and 2024 are two independent
  rolling-origin validation periods. The threshold must demonstrate >=20 trades,
  positive net USD P/L, PF>1.15 and <10% DD **in both periods**.
  Otherwise final 2025 simulation is **not run** and model promotion remains false.
- 2026 has been repeatedly inspected in prior research and is **not** an untouched
  generalization test anymore. Never report an exploratory 2026 rerun as forward.

## Archived-data smoke result

An offline check using the supplied historical TradeAI archive and broker profile
found **zero symbols passing both 2023 and 2024 validation gates** for this V1
specification. This is a functioning research/audit tool, **not a profitable strategy**.
Do not disable the gate or promote the output to live trading to obtain trades.

## Remaining limits

This is not a tick backtest. Candle OHLC cannot reveal exact stop/target event order,
precise entry spread, unobserved gaps, order rejects or tick-level quote liquidity.
There is no intratrade mark-to-market margin stopout, swap, financing, slippage spike
stress, latency, portfolio-level concurrency or trade-server lot validation.
A $150 account may have no acceptable minimum-lot trades under conservative risk.
Further research may require genuinely new features, different horizons or brokers;
none guarantees positive expectancy.
