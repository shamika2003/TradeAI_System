"""TradeAI V3: integrated historical BUY/SELL entry-ranking, ML, risk and portfolio replay.

RESEARCH ONLY: no broker API, order submission or production-model promotion.
2020-21 fit -> 2022 probability calibration -> 2023 threshold selection ->
2024 chronological evaluation. 2025 is not evaluated. Historical 2023/24 have
been inspected previously, so none of these results are independent forward proof.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier, XGBRegressor

from . import _broker_replay_core as core
from . import regime_edge_lab as base

SYMBOLS = ("US500Cash", "US100Cash")
HORIZONS = (24, 48)  # 2h and 4h, chosen BEFORE checking V3 outcomes
TRAIN = (2020, 2021)
CALIBRATION = 2022
SELECT = 2023
CHECK = 2024
UNTOUCHED_EVAL = 2025
INITIAL_USD = 150.0
TRADE_RISK_CAP = 0.015
DAILY_LOSS_CAP = 0.03
MONTHLY_LOSS_CAP = 0.08
MAX_CLOSED_DD = 0.12
MIN_VALID_TRADES = 30
MIN_PF = 1.15
MAX_PASSING_DD = 0.10
THRESHOLDS_R = (0.05, 0.15, 0.30)  # fixed before tests
EXTRA_FEE_USD = 0.20  # hypothetical round trip charge; actual CFD fees unknown
COST_STRESS_FEE_USD = 0.40  # total round trip fee in stress (incl core's 0.20)
CONSERVATIVE_LEVERAGE = 100  # margin RESERVE estimate, not broker margin assertion
EPS = 1e-8
SETUPS = ("trend_breakout", "trend_pullback", "mean_reversion", "momentum")
FEATURES = base.FEATURES + (
    "side", "horizon_4h", "is_us100", "side_ema_separation", "side_zscore",
    "side_delta3", "side_rsi_from_50", "side_body", "side_breakout",
    "side_breakdown", "side_close_ema", "setup_breakout", "setup_pullback",
    "setup_reversion", "setup_momentum", "votes_with", "votes_against",
    "has_setup", "trend_strength", "high_vol_regime", "cost_to_risk",
    "min_lot_risk_fraction", "margin_to_capital",
)


def derive_features(h: pd.DataFrame, x: pd.DataFrame, side: int,
                    horizon: int, symbol: str, risk: np.ndarray,
                    costs: np.ndarray, margin: np.ndarray) -> pd.DataFrame:
    """Only completed H1 features, known nominal entry costs and contract metadata."""
    d = x.copy().reset_index(drop=True)
    raw = np.column_stack([h[k].to_numpy(dtype=int) for k in SETUPS])
    same = (raw == side)
    opposing = (raw == -side)
    d["side"] = float(side)
    d["horizon_4h"] = float(horizon == 48)
    d["is_us100"] = float(symbol == "US100Cash")
    d["side_ema_separation"] = side * d["ema_separation_atr"]
    d["side_zscore"] = side * d["zscore_atr"]
    d["side_delta3"] = side * d["delta3_atr"]
    d["side_rsi_from_50"] = side * (d["rsi_fraction"] - .5)
    d["side_body"] = side * d["body_atr"]
    d["side_breakout"] = side * d["breakout20_distance_atr"]
    d["side_breakdown"] = side * d["breakdown20_distance_atr"]
    d["side_close_ema"] = side * d["close_ema20_atr"]
    for j, name in enumerate(("setup_breakout", "setup_pullback", "setup_reversion", "setup_momentum")):
        d[name] = same[:, j].astype(float)
    d["votes_with"] = same.sum(axis=1).astype(float)
    d["votes_against"] = opposing.sum(axis=1).astype(float)
    d["has_setup"] = (raw != 0).any(axis=1).astype(float)
    d["trend_strength"] = np.abs(d["ema_separation_atr"])
    d["high_vol_regime"] = (d["atr_relative_50"] > 1.1).astype(float)
    d["cost_to_risk"] = costs / np.maximum(risk, EPS)
    d["min_lot_risk_fraction"] = risk / INITIAL_USD
    d["margin_to_capital"] = margin / INITIAL_USD
    return d.loc[:, FEATURES]


def _year_end_ok(entry: pd.Series, exit_time: np.ndarray) -> np.ndarray:
    # Prevent all labels from straddling evaluation boundaries.
    yy = entry.dt.year.to_numpy()
    return pd.to_datetime(exit_time, unit="s", utc=True).year.to_numpy() == yy


def assemble_candidates(z: ZipFile, manifest: dict, symbol: str) -> pd.DataFrame:
    """Construct causal candidate features for BOTH sides, without hindsight entries."""
    h, m, spec, _ = core.prepare(z, manifest, symbol)
    h = h.loc[h.time_entry.dt.year <= CHECK].copy().reset_index(drop=True)
    x = base.make_causal_features(z, symbol, h)
    b = np.asarray(m.open, dtype=float)
    mi = h.mi.to_numpy(dtype=int)
    margin = b[mi] * float(spec["volume_min"]) * float(spec["trade_contract_size"]) / CONSERVATIVE_LEVERAGE
    entry_time = h.time_entry.copy().reset_index(drop=True)
    entry_epoch = core.unix_seconds(entry_time)
    all_d = []
    for horizon in HORIZONS:
        pre = core.precompute(h, m, spec, horizon)
        stressed = core.precompute(h, m, spec, horizon, stress=True)
        if pre is None or stressed is None:
            continue
        # core.precompute(stress=True) ALREADY deducts 0.20 USD. Add only
        # 0.20 more to arrive at the documented 0.40 stress fee total.
        for side in (1, -1):
            p, s = pre[side], stressed[side]
            risk = p["risk_usd"] + EXTRA_FEE_USD
            stressed_risk = s["risk_usd"] + EXTRA_FEE_USD
            valid = p["valid"] & (risk <= TRADE_RISK_CAP * INITIAL_USD)
            valid &= _year_end_ok(entry_time, p["exit_time"])
            valid &= _year_end_ok(entry_time, s["exit_time"])
            feat = derive_features(h, x, side, horizon, symbol, risk,
                                   p["cost_est"] + EXTRA_FEE_USD, margin)
            valid &= np.isfinite(feat.to_numpy(dtype=float)).all(axis=1)
            j = np.flatnonzero(valid)
            if not len(j):
                continue
            df = feat.iloc[j].copy().reset_index(drop=True)
            df["symbol"] = symbol
            df["entry_ts"] = entry_epoch[j]
            df["exit_ts"] = np.asarray(p["exit_time"], dtype=np.int64)[j]
            df["stress_exit_ts"] = np.asarray(s["exit_time"], dtype=np.int64)[j]
            df["year"] = entry_time.dt.year.to_numpy()[j]
            df["month"] = entry_time.dt.strftime("%Y-%m").to_numpy()[j]
            df["side_int"] = int(side)
            df["holding_m5"] = int(horizon)
            df["risk_usd"] = risk[j]
            df["stress_risk_usd"] = stressed_risk[j]
            df["margin_usd"] = margin[j]
            df["net_usd"] = np.asarray(p["net"], dtype=float)[j] - EXTRA_FEE_USD
            df["stress_net_usd"] = np.asarray(s["net"], dtype=float)[j] - EXTRA_FEE_USD
            df["stress_valid"] = np.asarray(s["valid"], dtype=bool)[j]
            df["exit_reason"] = np.asarray(p["reason"])[j]
            all_d.append(df)
    if not all_d:
        return pd.DataFrame(columns=list(FEATURES) + ["symbol", "year"])
    out = pd.concat(all_d, ignore_index=True)
    # Label columns, future exits and realised R are deliberately never in FEATURES.
    out["realized_R"] = np.clip(out["net_usd"] / out["risk_usd"], -4, 4)
    out["win"] = (out["net_usd"] > 0).astype(int)
    return out


def fit_models(d: pd.DataFrame):
    tr = d.loc[d.year.isin(TRAIN)]
    ca = d.loc[d.year == CALIBRATION]
    if len(tr) < 1500 or len(ca) < 400:
        raise RuntimeError("Insufficient distinct historical events for fit/calibration")
    xx = tr.loc[:, FEATURES]
    model_params = dict(n_estimators=220, max_depth=3, learning_rate=.035,
                        subsample=.80, colsample_bytree=.85, min_child_weight=65,
                        reg_alpha=3.0, reg_lambda=30.0, tree_method="hist", n_jobs=2)
    r1 = XGBRegressor(**model_params, objective="reg:squarederror", random_state=42)
    r2 = XGBRegressor(**model_params, objective="reg:squarederror", random_state=139)
    clf = XGBClassifier(**model_params, objective="binary:logistic", random_state=71,
                        eval_metric="logloss")
    r1.fit(xx, tr.realized_R.to_numpy())
    r2.fit(xx, tr.realized_R.to_numpy())
    clf.fit(xx, tr.win.to_numpy())
    # CALIBRATION year is later than model FIT years; thresholds not selected until 2023.
    calib_probs = clf.predict_proba(ca.loc[:, FEATURES])[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(calib_probs, ca.win.to_numpy())
    wins = tr.realized_R.to_numpy()
    mean_win = float(wins[wins > 0].mean())
    mean_loss = float(-wins[wins <= 0].mean())
    if not (mean_win > 0 and mean_loss > 0):
        raise RuntimeError("Unable to estimate both win and loss payoffs from fit years")
    return (r1, r2, clf, iso), {"train_rows": len(tr), "calibration_rows": len(ca),
         "train_avg_win_R": mean_win, "train_avg_loss_R": mean_loss,
         "train_positive_fraction": float(tr.win.mean())}


def predict_score(d: pd.DataFrame, models, stats: dict) -> pd.DataFrame:
    """Scores depend ONLY on feature columns and models trained before 2023."""
    r1, r2, clf, iso = models
    x = d.loc[:, FEATURES]
    p = iso.predict(clf.predict_proba(x)[:, 1])
    ev_cls = p * stats["train_avg_win_R"] - (1-p) * stats["train_avg_loss_R"]
    a, b = r1.predict(x), r2.predict(x)
    scored = d.copy()
    scored["ev_reg1_R"] = a
    scored["ev_reg2_R"] = b
    scored["p_win_calibrated"] = p
    scored["ev_probability_R"] = ev_cls
    # All 3 estimators must agree on positive net expectancy; choose more cautious.
    scored["score_R"] = np.minimum(np.minimum(a, b), ev_cls)
    return scored


def simulate_portfolio(d: pd.DataFrame, threshold: float, year: int,
                       stressed: bool = False, no_ai: bool = False) -> dict:
    """One position across assets; conservative stop-first outcomes from M5 paths.

    Trade results are computed offline from future paths but never consulted to
    rank/choose entries; only score/known risk/margin/time are consulted.
    """
    subset = d.loc[d.year == year].copy()
    if no_ai:
        # Rule baseline is NOT hindsight-optimised: prioritize aligned setup votes.
        subset["rank_priority"] = subset["votes_with"] - subset["votes_against"]
        subset = subset.loc[subset.votes_with > 0]
    else:
        subset = subset.loc[subset.score_R >= threshold]
        subset["rank_priority"] = subset.score_R
    subset = subset.sort_values(["entry_ts", "rank_priority", "symbol", "side_int", "holding_m5"],
                                ascending=[True, False, True, False, True], kind="stable")
    capital = INITIAL_USD
    peak = capital
    max_dd = 0.0
    busy_until = -1
    date_start = INITIAL_USD
    month_start = INITIAL_USD
    current_day = current_month = None
    positive = negative = 0.0
    rejections = Counter()
    logs = []
    risk_stopped = False
    for row in subset.itertuples(index=False):
        ts = int(row.entry_ts)
        if ts < busy_until:
            rejections["EXISTING_POSITION"] += 1
            continue
        day = ts // 86400
        if day != current_day:
            current_day = day
            date_start = capital
        if row.month != current_month:
            current_month = row.month
            month_start = capital
        if capital <= date_start * (1 - DAILY_LOSS_CAP):
            rejections["DAILY_LOSS_CAP"] += 1
            continue
        if capital <= month_start * (1 - MONTHLY_LOSS_CAP):
            rejections["MONTHLY_LOSS_CAP"] += 1
            continue
        if stressed and not row.stress_valid:
            rejections["STRESS_COST_TOO_HIGH"] += 1
            continue
        risk = float(row.stress_risk_usd if stressed else row.risk_usd)
        if risk <= 0 or risk > TRADE_RISK_CAP * capital:
            rejections["MIN_LOT_RISK"] += 1
            continue
        if capital - risk < row.margin_usd + 2.0:
            rejections["MARGIN_RESERVE"] += 1
            continue
        net = float(row.stress_net_usd if stressed else row.net_usd)
        capital += net
        positive += max(net, 0.0)
        negative += max(-net, 0.0)
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak)
        busy_until = int(row.stress_exit_ts if stressed else row.exit_ts)
        logs.append({"time": pd.Timestamp(ts, unit="s", tz="UTC").isoformat(),
                     "symbol": row.symbol, "side": "BUY" if row.side_int == 1 else "SELL",
                     "horizon_h": int(row.holding_m5 / 12),
                     "predicted_net_R": round(float(row.rank_priority), 5),
                     "net_usd": round(net, 5), "balance_usd": round(capital, 5),
                     "risk_usd": round(risk, 5), "exit_reason": str(row.exit_reason)})
        if capital <= 0 or max_dd >= MAX_CLOSED_DD:
            risk_stopped = True
            break
    n = len(logs)
    months = defaultdict(float)
    for row in logs:
        months[row["time"][:7]] += row["net_usd"]
    return {
        "year": year, "trades": n, "net_usd": round(capital - INITIAL_USD, 4),
        "final_balance_usd": round(capital, 4),
        "profit_factor": round(positive / negative, 4) if negative else (999.0 if n else None),
        "win_rate_pct": round(sum(t["net_usd"] > 0 for t in logs) * 100 / n, 2) if n else None,
        "max_closed_dd_pct": round(100 * max_dd, 3), "risk_stopped": risk_stopped,
        "signal_candidates": len(subset), "rejections": dict(rejections),
        "monthly_net_usd": {m: round(v, 4) for m, v in sorted(months.items())},
        "trades_log": logs,
    }


def passes(r: dict) -> bool:
    return bool(r["trades"] >= MIN_VALID_TRADES and r["net_usd"] > 0 and
                (r["profit_factor"] or 0) >= MIN_PF and
                r["max_closed_dd_pct"] <= 100 * MAX_PASSING_DD and
                not r["risk_stopped"])


def summarize(r: dict) -> dict:
    return {k: v for k, v in r.items() if k != "trades_log"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", nargs="?", default="run", choices=("run", "tests"))
    ap.add_argument("--zip", type=Path, default=Path("TradeAI_NEW_MARKETS_2020_2025.zip"))
    ap.add_argument("--out", type=Path, default=Path("artifacts/research/portfolio_lab_v3"))
    args = ap.parse_args(argv)
    if args.action == "tests":
        import unittest
        suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"),
                                                        pattern="test_portfolio_lab_v3.py")
        ok = unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
        return 0 if ok else 1
    if not args.zip.is_file():
        raise SystemExit(f"Market data archive missing: {args.zip}")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"Refusing to overwrite previous research: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    print("TRADEAI V3 | Integrated entry + three-model AI + portfolio risk + simulator", flush=True)
    print("RESEARCH ONLY. 2025 is not scored or evaluated. No MT5 orders.", flush=True)
    with ZipFile(args.zip) as z:
        manifest = base.verify_manifest(z)
        parts = []
        for symbol in SYMBOLS:
            candidates = assemble_candidates(z, manifest, symbol)
            print(f"{symbol}: {len(candidates):,} affordable side/horizon events (2020-2024)", flush=True)
            if not candidates.empty:
                parts.append(candidates)
    if not parts:
        raise RuntimeError("No historically affordable candidates")
    allrows = pd.concat(parts, ignore_index=True)
    print("Candidate counts by year:", {str(k):int(v) for k,v in allrows.year.value_counts().sort_index().items()}, flush=True)
    models, stats = fit_models(allrows)
    print(f"TRAIN 2020-21: {stats['train_rows']:,}; CALIBRATION 2022: {stats['calibration_rows']:,}", flush=True)
    evalrows = allrows.loc[allrows.year.isin((SELECT, CHECK))].copy()
    scored = predict_score(evalrows, models, stats)
    # Descriptive scores: no post-hoc threshold adjustment and no 2025 data.
    score_audit = {}
    for year in (SELECT, CHECK):
        peryear = scored.loc[scored.year == year]
        arr = peryear.score_R.to_numpy(dtype=float)
        score_audit[str(year)] = {
            "candidates":len(peryear),
            "mean_predicted_net_R":round(float(arr.mean()),5),
            "max_predicted_net_R":round(float(arr.max()),5),
            "p95_predicted_net_R":round(float(np.percentile(arr,95)),5),
            "individual_regressor_1_positive":int((peryear.ev_reg1_R > 0).sum()),
            "individual_regressor_2_positive":int((peryear.ev_reg2_R > 0).sum()),
            "calibrated_probability_EV_positive":int((peryear.ev_probability_R > 0).sum()),
            "all_models_positive":int((arr > 0).sum()),
            "realized_mean_net_R_all_candidates_POSTHOC_ONLY":round(float(peryear.realized_R.mean()),5),
        }
        print(f"{year} score audit: {score_audit[str(year)]['all_models_positive']} positive of {len(peryear)} "
              f"(max {score_audit[str(year)]['max_predicted_net_R']:+.3f}R)", flush=True)
    baselines = {str(y):summarize(simulate_portfolio(scored, 0, y, no_ai=True))
                 for y in (SELECT, CHECK)}
    trials = []
    for t in THRESHOLDS_R:
        a = simulate_portfolio(scored, t, SELECT)
        b = simulate_portfolio(scored, t, CHECK)
        stress = simulate_portfolio(scored, t, CHECK, stressed=True)
        trials.append({"min_predicted_net_R": t, "selection_2023":summarize(a),
                       "diagnostic_2024":summarize(b),
                       "cost_stress_2024":summarize(stress),
                       "passes_selection":passes(a),
                       "passes_2024_and_stress":bool(passes(a) and passes(b) and passes(stress))})
        print(f"Threshold {t:.2f}R | 2023 {a['trades']} trades {a['net_usd']:+.2f} USD | "
              f"2024 {b['trades']} trades {b['net_usd']:+.2f} USD | "
              f"stress {stress['net_usd']:+.2f} USD", flush=True)
    selected_candidates = [v for v in trials if v["passes_selection"]]
    chosen = max(selected_candidates, key=lambda v:v["selection_2023"]["net_usd"]) if selected_candidates else None
    selected_t = chosen["min_predicted_net_R"] if chosen else None
    approved = bool(chosen and chosen["passes_2024_and_stress"])
    modeldir = args.out / "models_RESEARCH_ONLY"
    modeldir.mkdir()
    for model, name in zip(models[:3], ("net_r_ensemble_a.json", "net_r_ensemble_b.json", "win_probability.json")):
        model.save_model(modeldir / name)
    # Isotonic calibration is stored as data points, not arbitrary pickle code.
    (modeldir / "calibration.json").write_text(json.dumps({
        "x_thresholds": models[3].X_thresholds_.tolist(),
        "y_thresholds": models[3].y_thresholds_.tolist()}, indent=2), encoding="utf-8")
    report = {
        "experiment":"TRADEAI_V3_INTEGRATED_PORTFOLIO_RESEARCH",
        "not_live_approved":True, "research_only":True,
        "forward_independence_warning":"2023 and 2024 have been examined in earlier experiments, so results are exploratory; 2025 not evaluated in this run, but earlier strategy families have inspected 2025. Future data necessary.",
        "years":{"fit":list(TRAIN),"calibration":CALIBRATION,"threshold_selection":SELECT,
                 "later_evaluation":CHECK,"no_evaluation":UNTOUCHED_EVAL},
        "configuration":{"symbols":SYMBOLS,"horizons_m5":HORIZONS,"features":len(FEATURES),
            "model":"Two XGBoost net-R regressors and an independently calibrated XGBoost win-probability classifier",
            "candidate_entry":"Both BUY and SELL every eligible completed H1; H1 setups are features, not hindsight constraints",
            "ranking":"minimum of ensemble predicted after-cost net R and calibrated payoff-based EV; one position across both assets",
            "account_usd":INITIAL_USD,"risk_fraction":TRADE_RISK_CAP,
            "daily_loss_fraction":DAILY_LOSS_CAP,"monthly_loss_fraction":MONTHLY_LOSS_CAP,
            "closed_trade_drawdown_fraction":MAX_CLOSED_DD,"stop_atr_multiplier":core.RISK_STOP_ATR,
            "target_stop_multiple":core.TP_MULT,"precommitted_thresholds_R":THRESHOLDS_R,
            "cost_assumptions":"M5 BID OHLC broker bar spread + adverse hypothetical slippage + nominal 0.20 USD fee; stress doubles slippage, widens spread 1.5x, fee total 0.40 USD; no ticks/swap/actual fee schedule",
            "estimated_leverage_for_margin_reserve":CONSERVATIVE_LEVERAGE,
            "known_limitations":"Closed-trade drawdown only; no intratrade equity, historical funding charges or exact tick fills. 2023/24 previously seen."},
        "fit":stats,
        "prediction_score_audit":score_audit,
        "trained_regressor_feature_importance_top10":[
            {"feature":str(FEATURES[i]),"importance":round(float(v),5)}
            for i,v in sorted(enumerate(models[0].feature_importances_),
                              key=lambda t:t[1],reverse=True)[:10]
        ],
        "qualified_candidate_rows_by_symbol_year":{str((s,int(y))):int(k) for (s,y),k in allrows.groupby(["symbol","year"]).size().items()},
        "entry_rule_only_benchmark":baselines,
        "threshold_trials":trials,
        "selected_threshold_R":selected_t,"passes_all_research_gates":approved,
        "production_approved":False,
    }
    (args.out / "report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("REPORT:", args.out / "report.json")
    print("RESEARCH GATE:", "PASS (requires future confirmation)" if approved else "FAIL")
    print("PRODUCTION APPROVED: NO. NO MT5 ORDERS.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
