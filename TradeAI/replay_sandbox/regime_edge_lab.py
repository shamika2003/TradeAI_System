"""TradeAI: isolated, cost-aware US-index XGBoost research.

No MetaTrader5 import, no orders, no automatic promotion. 2025 outcomes are not
read in the default `run` command. Requires the user's historical broker ZIP.
2023 tuning and 2024 evaluation are exploratory (previously inspected in this
project); only later truly forward data can be independent confirmation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from . import _broker_replay_core as core

SYMBOLS = ("US500Cash", "US100Cash")
HOLDING_M5 = 24  # fixed 2 hours, no horizon optimisation
TRAIN_YEARS = (2020, 2021, 2022)
TUNE_YEAR = 2023
CHECK_YEAR = 2024
SEALED_YEAR = 2025
SPREAD_COMMISSION_USD = 0.20  # hypothetical EXTRA round-trip CFD fee for minimum volume
STRESS_COMMISSION_USD = 0.40
THRESHOLDS_R = (0.05, 0.10, 0.20)  # fixed before examining scores
MIN_VALIDATION_TRADES = 30
MIN_PF = 1.15
MAX_VALIDATION_DD_PCT = 10.0
FEATURES = (
    "ema_separation_atr", "close_ema20_atr", "atr_relative_50",
    "close_location", "body_atr", "range_atr", "upper_wick_atr",
    "lower_wick_atr", "rsi_fraction", "delta3_atr", "zscore_atr",
    "breakout20_distance_atr", "breakdown20_distance_atr",
    "log_tick_volume", "entry_spread_to_atr", "hour_sin", "hour_cos",
    "weekday_sin", "weekday_cos", "past_1h_return_atr",
)


def features_from_eligible(h: pd.DataFrame) -> pd.DataFrame:
    """All inputs are computed from the most recently completed H1 candle.

    `h.time` is the H1 candle OPEN; `h.time_entry` = h.time + one hour.
    No shift(-1), future price/return or future spread appears in FEATURES.
    """
    atr = h["atr"].to_numpy(dtype=float)
    op = h["open"].to_numpy(dtype=float)
    close = h["close"].to_numpy(dtype=float)
    high = h["high"].to_numpy(dtype=float)
    low = h["low"].to_numpy(dtype=float)
    ema20 = h["ema20"].to_numpy(dtype=float)
    ema50 = h["ema50"].to_numpy(dtype=float)
    scale = np.maximum(atr, 1.e-8)
    t = h["time_entry"].dt
    frac = t.hour.to_numpy() + t.minute.to_numpy()/60
    dow = t.dayofweek.to_numpy()
    bottom = np.minimum(op, close)
    top = np.maximum(op, close)
    df = pd.DataFrame({
        "ema_separation_atr": (ema20-ema50)/scale,
        "close_ema20_atr": (close-ema20)/scale,
        # pct-relative volatility regime is a causal 50-bar ATR rolling mean, see below.
        "atr_relative_50": np.zeros(len(h),dtype=float),
        "close_location": (close-low)/np.maximum(high-low,1.e-8),
        "body_atr": (close-op)/scale,
        "range_atr": (high-low)/scale,
        "upper_wick_atr": (high-top)/scale,
        "lower_wick_atr": (bottom-low)/scale,
        "rsi_fraction": h["rsi"].to_numpy(dtype=float)/100.0,
        "delta3_atr": h["delta3"].to_numpy(dtype=float),
        "zscore_atr": h["z"].to_numpy(dtype=float),
        "breakout20_distance_atr": (close-h["avg_high20"].to_numpy(dtype=float))/scale,
        "breakdown20_distance_atr": (close-h["avg_low20"].to_numpy(dtype=float))/scale,
        "log_tick_volume": np.log1p(np.maximum(h["tick_volume"].to_numpy(dtype=float),0.0)),
        "entry_spread_to_atr": h["spread_entry"].to_numpy(dtype=float)/scale,
        "hour_sin": np.sin(2*np.pi*frac/24),
        "hour_cos": np.cos(2*np.pi*frac/24),
        "weekday_sin": np.sin(2*np.pi*dow/7),
        "weekday_cos": np.cos(2*np.pi*dow/7),
        "past_1h_return_atr": (close-h["closeprev"].to_numpy(dtype=float))/scale,
    })
    # Eligible rows can have gaps; compute ATR trailing reference on ALL available
    # H1 bars using a separate causal merge in load_training_rows below.
    return df


def make_causal_features(z: ZipFile, symbol: str, eligible: pd.DataFrame) -> pd.DataFrame:
    # Use complete H1 history (including warmup bars) for rolling computation.
    h = core.add_signals(core.load(z, symbol, "H1"))
    h = h.loc[h.time.dt.year <= CHECK_YEAR].copy()
    h["atr_relative_50"] = h.atr / h.atr.rolling(50, min_periods=50).mean()
    ref = h[["time_entry", "atr_relative_50"]]
    df = features_from_eligible(eligible)
    lookup = ref.set_index("time_entry")["atr_relative_50"].reindex(eligible.time_entry)
    df["atr_relative_50"] = lookup.to_numpy(dtype=float)
    return df


def add_commission(pre: dict, amount: float) -> dict:
    """Copies cost outcomes: never modifies source replay data in place."""
    return {side: {**p,
        "net": p["net"].copy() - amount,
        "risk_usd": p["risk_usd"].copy() + amount}
        for side, p in pre.items()}


def actions_from_scores(buy: np.ndarray, sell: np.ndarray, threshold: float) -> np.ndarray:
    # No HOLD score needed: trade only when best predicted net R is positive.
    # Need a 0.03R difference to avoid random flipping between near-tied sides.
    out = np.zeros(len(buy), dtype=np.int8)
    out[(buy >= threshold) & (buy > sell+0.03)] = 1
    out[(sell >= threshold) & (sell > buy+0.03)] = -1
    return out


def acceptable(result: dict) -> bool:
    return bool(result["trades"] >= MIN_VALIDATION_TRADES
       and result["net_usd"] > 0
       and (result["profit_factor"] or 0) >= MIN_PF
       and result["max_dd_pct"] <= MAX_VALIDATION_DD_PCT
       and not result["risk_stopped"])


def verify_manifest(z: ZipFile) -> dict:
    manifest = json.loads(z.read("collection_manifest.json"))
    # Verify actual compressed bytes against hashes frozen by the MT5 collector.
    for symbol in SYMBOLS:
        for tf in ("M5", "H1"):
            key=f"{symbol}_{tf}"
            item=manifest.get("data_files", {}).get(key)
            name=f"market_data/{key}.csv.gz"
            if not item or item.get("archive_member") != name or name not in z.namelist():
                raise ValueError(f"Missing or mismatched archive manifest entry: {key}")
            expected=item.get("sha256")
            if not expected or hashlib.sha256(z.read(name)).hexdigest()!=expected:
                raise ValueError(f"Archive SHA256 integrity check failed: {key}")
    return manifest


def extract_eligible(z: ZipFile, manifest: dict, symbol: str):
    eligible, m, spec, quality = core.prepare(z, manifest, symbol)
    eligible = eligible.loc[eligible.time_entry.dt.year <= CHECK_YEAR].copy().reset_index(drop=True)
    x = make_causal_features(z, symbol, eligible)
    raw = core.precompute(eligible, m, spec, HOLDING_M5)
    if raw is None:
        raise RuntimeError(f"No rows for {symbol}")
    pre = add_commission(raw, SPREAD_COMMISSION_USD)
    valid = np.isfinite(x.to_numpy(dtype=float)).all(axis=1)
    valid &= pre[1]["valid"] & pre[-1]["valid"]
    # Year-specific path closure prevents training labels crossing into the next year.
    year = eligible.time_entry.dt.year.to_numpy()
    for side in (1,-1):
        end_year = pd.to_datetime(pre[side]["exit_time"],unit="s",utc=True).year
        valid &= end_year == year
    print(f"{symbol}: eligible={len(eligible):,}, causal_valid={sum(valid):,}, features={len(FEATURES)}",flush=True)
    y = {}
    for side in (1,-1):
        # Outcome in units of stop-equivalent money, after spread, slippage and fee.
        y[side] = np.clip(pre[side]["net"] / np.maximum(pre[side]["risk_usd"],.01), -4., 4.)
    return eligible, x, y, pre, valid, m, spec, quality


def fit_model(x, labels):
    return XGBRegressor(n_estimators=160,max_depth=2,learning_rate=.045,
        subsample=.8,colsample_bytree=.8,min_child_weight=65,
        reg_lambda=15,objective="reg:squarederror",n_jobs=2,random_state=42,
        tree_method="hist").fit(x,labels,verbose=False)


def concise(r):
    return {k:r[k] for k in ("trades","net_usd","profit_factor", "max_dd_pct", "risk_stopped", "win_rate_pct", "entry_signals", "skip_risk_cost_missing", "skip_daily_cap")}


def train_one(z, manifest, symbol, outdir):
    h, x, y, pre, valid, m, spec, quality = extract_eligible(z, manifest, symbol)
    year = h.time_entry.dt.year.to_numpy()
    train = valid & np.isin(year, TRAIN_YEARS)
    tune = valid & (year==TUNE_YEAR)
    check = valid & (year==CHECK_YEAR)
    if min(sum(train),sum(tune),sum(check)) < 500:
        raise ValueError(f"Too few valid rows for {symbol}")
    # Train uses 2020..22 labels only; thresholds are tuned only on 2023.
    models = {side:fit_model(x.loc[train,list(FEATURES)],y[side][train]) for side in (1,-1)}
    scores = {side:models[side].predict(x.loc[:,list(FEATURES)]) for side in (1,-1)}
    trials=[]
    for threshold in THRESHOLDS_R:
        flag=actions_from_scores(scores[1], scores[-1], threshold)
        val=core.simulate(h,pre,flag,TUNE_YEAR)
        # No 2024 evaluated in the trial loop; choose 2023 threshold first.
        trials.append({"threshold_R":threshold,"validation_2023":concise(val),"passed":acceptable(val)})
    candidates=[r for r in trials if r["passed"]]
    selected=max(candidates,key=lambda q:q["validation_2023"]["net_usd"]) if candidates else None
    chosen_threshold=selected["threshold_R"] if selected else None
    later=None
    stress=None
    if selected is not None:
        flags=actions_from_scores(scores[1],scores[-1],chosen_threshold)
        later=concise(core.simulate(h,pre,flags,CHECK_YEAR))
        # stress: all other assumptions unchanged, doubled slippage, 1.5x spread
        # and $0.40 round-turn commission. Holdout not touched.
        stressed=add_commission(core.precompute(h,m,spec,HOLDING_M5,stress=True),STRESS_COMMISSION_USD)
        stress=concise(core.simulate(h,stressed,flags,CHECK_YEAR))
    gate=bool(selected and later and stress and acceptable(later) and acceptable(stress))
    symbol_report={
        "symbol":symbol,"years":{"fit":[2020,2021,2022],"threshold_selection":TUNE_YEAR,
                            "locked_evaluation":CHECK_YEAR,"sealed_unopened":SEALED_YEAR},
        "train_rows":int(train.sum()),"tune_rows":int(tune.sum()),"locked_eval_rows":int(check.sum()),
        "feature_count":len(FEATURES),"features":list(FEATURES),
        "threshold_trials_2023":trials,"chosen_threshold_R":chosen_threshold,
        "2024_locked_validation":later,"2024_cost_stress":stress,
        "strict_gate_passed":gate,"not_live_approved":True,
        "rules":{"hold_minutes":120,"min_lot":spec['volume_min'],
                 "risk_fraction":core.MAX_RISK,"daily_loss_cap":core.MAX_DAILY,
                 "max_drawdown":core.MAX_DD,"spread_slippage": "bar-based estimates, not tick quotes",
                 "commission_base_extra_usd_roundtrip":SPREAD_COMMISSION_USD,
                 "commission_stress_extra_usd_roundtrip":STRESS_COMMISSION_USD,
                 "capital":core.INITIAL}}
    # Save research-only models, even failed candidates, for audit and reproducibility.
    model_dir=outdir/"models"/symbol
    model_dir.mkdir(parents=True,exist_ok=True)
    for side,mdl in models.items():
        mdl.save_model(model_dir/("buy_research_only.json" if side==1 else "sell_research_only.json"))
    (model_dir/"metadata.json").write_text(json.dumps({"approved_for_live":False,
        "strict_gate_passed":gate,"features":list(FEATURES),"selected_threshold_R":chosen_threshold,
        "warning":"RESEARCH ONLY: historical 2023/2024 previously inspected, 2025 not evaluated"},indent=2),encoding="utf-8")
    print(f"{symbol}: 2023 threshold {chosen_threshold}; 2024 {later}; STRESS {stress}; GATE {'PASS' if gate else 'FAIL'}",flush=True)
    return symbol_report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("run", "tests"),default="run",nargs="?")
    parser.add_argument("--zip",dest="archive",type=Path,default=Path("TradeAI_NEW_MARKETS_2020_2025.zip"))
    parser.add_argument("--out",type=Path,default=Path("artifacts/research/regime_edge_lab_v1"))
    args=parser.parse_args(argv)
    if args.command=="tests":
        import unittest
        suite=unittest.defaultTestLoader.discover(
            start_dir=str(Path(__file__).resolve().parent/"tests"),
            pattern="test_regime_edge_lab.py")
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.archive.is_file():
        raise SystemExit(f"Market-data ZIP not found: {args.archive}. Keep ZIP in project root.")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Refusing to overwrite prior research: {args.out}")
    args.out.mkdir(parents=True,exist_ok=True)
    report={"experiment":"TradeAI_US_INDEX_NET_R_ML_RESEARCH_V1",
      "archive_sha256":hashlib.sha256(args.archive.read_bytes()).hexdigest(),
      "status":"RESEARCH_SIMULATION_ONLY_NO_MT5_ORDERS_NO_AUTOMATIC_PROMOTION",
      "selection_honesty":"2023/2024 were previously inspected by other experiments; results exploratory",
      "model":"separate BUY / SELL XGBoost net-R regressors", 
      "unknown_costs":"CFD commission/swap schedules are not supplied; extra hypothetical fees included, no tick fills",
      "rules":"predefined two-hour hold, next H1 open M5 entry, same-candle stop first, no 2025 read",
      "symbols":{}}
    with ZipFile(args.archive) as z:
        manifest=verify_manifest(z)
        for symbol in SYMBOLS:
            report["symbols"][symbol]=train_one(z,manifest,symbol,args.out)
    report["strict_gate_passed_all"] = any(s["strict_gate_passed"] for s in report["symbols"].values())
    report["ready_for_live"] = False
    (args.out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:",args.out/"report.json")
    print("APPROVED FOR REAL MONEY: NO. 2025 HOLDOUT UNOPENED. NO ORDERS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
