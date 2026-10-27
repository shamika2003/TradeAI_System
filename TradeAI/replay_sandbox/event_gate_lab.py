"""TradeAI event-conditioned, cost-aware ML entry research, isolated from production.

Strategy: entry rules nominate an action from *closed* H1 bars; a trained model
estimates net R for that nominated action; risk/execution simulator decides whether
entry is affordable and simulates stops, takes, day-loss and drawdown constraints.
Training 2020-22 -> threshold selection 2023 -> locked check 2024. 2025
is not evaluated or used for any target/model selection; the archive reader
may load 2025 candles in memory before filtering to 2024.
2023/24 were inspected in previous work, so this is *exploratory*, not a virgin test.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

from . import _broker_replay_core as core
from . import regime_edge_lab as base

SYMBOLS=("US500Cash", "US100Cash")
SETUPS=("trend_breakout", "trend_pullback", "mean_reversion", "momentum")
TRAIN=(2020, 2021, 2022)
TUNE=2023
CHECK=2024
SEALED=2025
THRESHOLDS=(0.0, 0.05, 0.10, 0.20)
MIN_TRADES=30
MIN_PF=1.15
MAX_DD_PERCENT=10.0
FEATURES=base.FEATURES+(
    "side", "side_ema_separation_atr", "side_close_ema20_atr",
    "side_body_atr", "side_delta3_atr", "side_zscore_atr",
    "side_breakout20_distance_atr", "side_rsi_from50",
    "setup_trend_breakout", "setup_trend_pullback", "setup_mean_reversion",
    "setup_momentum", "aligned_votes", "conflicting_votes",
    "estimated_initial_risk_fraction",
)


def make_events(h:pd.DataFrame, x:pd.DataFrame, pre:dict) -> tuple[np.ndarray,pd.DataFrame,dict]:
    """Nominate one side from independent, causal H1 events; no outcomes used."""
    if len(h)!=len(x):raise ValueError('Data/feature length mismatch')
    raw=np.column_stack([h[c].to_numpy(dtype=np.int8) for c in SETUPS])
    up=(raw==1).sum(axis=1)
    dn=(raw==-1).sum(axis=1)
    # Opposing evidence -> HOLD; no majority lookahead/tie breaking by outcomes.
    side=np.where((up>0)&(dn==0),1,np.where((dn>0)&(up==0),-1,0)).astype(np.int8)
    events=x.copy().reset_index(drop=True)
    events['side']=side.astype(float)
    events['side_ema_separation_atr']=side*events['ema_separation_atr']
    events['side_close_ema20_atr']=side*events['close_ema20_atr']
    events['side_body_atr']=side*events['body_atr']
    events['side_delta3_atr']=side*events['delta3_atr']
    events['side_zscore_atr']=side*events['zscore_atr']
    events['side_breakout20_distance_atr']=side*events['breakout20_distance_atr']
    events['side_rsi_from50']=side*(events['rsi_fraction']-.5)
    for j,name in enumerate(SETUPS):
        events['setup_'+name]=(raw[:,j]==side).astype(float)*(side!=0)
    events['aligned_votes']=np.maximum(up,dn).astype(float)
    events['conflicting_votes']=np.minimum(up,dn).astype(float)
    risk=np.where(side==1,pre[1]['risk_usd'],pre[-1]['risk_usd'])
    events['estimated_initial_risk_fraction']=risk/core.INITIAL
    diagnostics={'no_setup':int((up+dn==0).sum()),
       'conflicts':int(((up>0)&(dn>0)).sum()),
       'nominated':int((side!=0).sum())}
    return side,events,diagnostics


def make_fit_mask(h, events, side, pre, valid):
    # Train only entries that could have been filled at the initial $150 balance.
    # Actual check at every trade uses current equity and is more conservative.
    risk=np.where(side==1,pre[1]['risk_usd'],pre[-1]['risk_usd'])
    mask=(side!=0)&valid&(risk<=core.INITIAL*core.MAX_RISK)
    mask &= np.isfinite(events.loc[:,FEATURES].to_numpy(dtype=float)).all(axis=1)
    return mask


def train(events, side, pre, mask, years):
    rows=mask & np.isin(years,TRAIN)
    if int(rows.sum()) < 250: raise ValueError(f'Only {int(rows.sum())} affordable training events; insufficient data')
    # Same-candle stop-first, spread/slip & assumed commission incorporated in label.
    raw_y=np.where(side==1,pre[1]['net'],pre[-1]['net'])
    raw_risk=np.where(side==1,pre[1]['risk_usd'],pre[-1]['risk_usd'])
    labels=np.clip(raw_y/np.maximum(raw_risk,.01),-4.,4.)
    model=XGBRegressor(n_estimators=110,max_depth=2,learning_rate=.035,
        min_child_weight=35,subsample=.85,colsample_bytree=.8,
        reg_lambda=25,reg_alpha=2.5,objective='reg:squarederror',
        random_state=173,n_jobs=2,tree_method='hist')
    model.fit(events.loc[rows,FEATURES],labels[rows],verbose=False)
    return model,int(rows.sum()),labels


def train_profit_classifier(events, side, pre, mask, years):
    """Independent cost-aware estimate: probability of *net profitable fill*."""
    rows=mask & np.isin(years,TRAIN)
    outcome=np.where(side==1,pre[1]['net'],pre[-1]['net'])
    risk=np.where(side==1,pre[1]['risk_usd'],pre[-1]['risk_usd'])
    realized_r=outcome/np.maximum(risk,.01)
    profitable=(realized_r[rows]>0).astype(int)
    if len(np.unique(profitable))!=2:
        raise ValueError('Cannot fit profit classifier: only one outcome class')
    model=XGBClassifier(n_estimators=120,max_depth=2,learning_rate=.04,
        min_child_weight=35,subsample=.8,colsample_bytree=.8,
        reg_lambda=20,reg_alpha=2,random_state=11,tree_method='hist',n_jobs=2)
    model.fit(events.loc[rows,FEATURES],profitable)
    mean_win=float(realized_r[rows][profitable==1].mean())
    mean_loss=float(-realized_r[rows][profitable==0].mean())
    # Used as a *model assumption*, not knowledge of future payoff distribution.
    break_even_probability=mean_loss/(mean_loss+mean_win)
    return model, {'training_mean_win_R':mean_win,
        'training_mean_loss_R':mean_loss,
        'training_break_even_win_probability':break_even_probability}


def select(h,pre,side,mask,scores,threshold,year):
    # Model entries must be *positive after costs*. No hindsight trade selection.
    actions=np.where(mask & (scores>=threshold),side,0).astype(np.int8)
    r=core.simulate(h,pre,actions,year)
    # See rejection reasons and execution coverage separately from model signal count.
    return {k:r[k] for k in (
        'year','trades','net_usd','ending_usd','profit_factor',
        'win_rate_pct','max_dd_pct','risk_stopped','active_days','stops',
        'entry_signals','skip_risk_cost_missing','skip_daily_cap')}


def accepted(r):
    return (r['trades']>=MIN_TRADES and r['net_usd']>0
       and (r['profit_factor'] or 0)>=MIN_PF
       and r['max_dd_pct']<=MAX_DD_PERCENT and not r['risk_stopped'])


def work_one(z,manifest,symbol,outdir):
    h,x,_,pre,valid,m,spec,quality=base.extract_eligible(z,manifest,symbol)
    side,features,event_diag=make_events(h,x,pre)
    mask=make_fit_mask(h,features,side,pre,valid)
    years=h.time_entry.dt.year.to_numpy()
    eligible={str(y):int((mask & (years==y)).sum()) for y in (*TRAIN,TUNE,CHECK)}
    train_available=int(sum(eligible[str(y)] for y in TRAIN))
    if train_available < 250:
        report={'symbol':symbol,'history_range':'2020-2024 outcomes only; no 2025 test',
           'train_event_rows':train_available,'affordable_events_by_year':eligible,
           'setup_diagnostics_all_years':event_diag,'tuning_2023':[],
           'threshold_R':None,'check_2024':None,'stress_2024':None,
           'passes_all_gates':False,'research_only':True,
           'skip_reason':'INSUFFICIENT_AFFORDABLE_TRAINING_EVENTS',
           'minimum_training_events':250}
        print(f"{symbol}: SKIP. Only {train_available} affordable event rows vs 250 required. {eligible}",flush=True)
        return report
    model,train_count,labels=train(features,side,pre,mask,years)
    classifier,payoff=train_profit_classifier(features,side,pre,mask,years)
    regression_scores=np.full(len(h),-np.inf,dtype=float)
    classification_scores=np.full(len(h),-np.inf,dtype=float)
    probabilities=np.zeros(len(h),dtype=float)
    # Predicted net R by two complementary cost-aware estimators.
    # Require agreement; disagreement or non-positive estimate -> HOLD.
    regression_scores[mask]=model.predict(features.loc[mask,FEATURES])
    probabilities[mask]=classifier.predict_proba(features.loc[mask,FEATURES])[:,1]
    classification_scores[mask]=(probabilities[mask]*payoff['training_mean_win_R']
        -(1-probabilities[mask])*payoff['training_mean_loss_R'])
    scores=np.minimum(regression_scores,classification_scores)
    baseline_2023=select(h,pre,side,mask,np.ones(len(h)),0.0,TUNE)
    baseline_2024=select(h,pre,side,mask,np.ones(len(h)),0.0,CHECK)
    predictions={}
    outcome_diagnostics={}
    for year in (TUNE,CHECK):
        year_mask=mask&(years==year)
        p=scores[year_mask]
        outcomes=np.where(side==1,pre[1]['net']/np.maximum(pre[1]['risk_usd'],.01),
                          pre[-1]['net']/np.maximum(pre[-1]['risk_usd'],.01))
        predictions[str(year)]={'regression_pred_R_max':round(float(regression_scores[year_mask].max()),4),
           'classifier_breakeven_probability':round(payoff['training_break_even_win_probability'],4),
           'classifier_max_predicted_profit_probability':round(float(probabilities[year_mask].max()),4),
           'classifier_positive_net_R_predictions':int((classification_scores[year_mask]>0).sum()),
           'predicted_net_R_min':round(float(p.min()),4),
           'predicted_net_R_median':round(float(np.median(p)),4),
           'predicted_net_R_max':round(float(p.max()),4),
           'positive_predictions':int((p>0).sum())}
        outcome_diagnostics[str(year)]={'affordable_events':int(year_mask.sum()),
           'posthoc_mean_net_R_all_events':round(float(outcomes[year_mask].mean()),4)}
    trials=[]
    for threshold in THRESHOLDS:
        r=select(h,pre,side,mask,scores,threshold,TUNE)
        trials.append({'min_predicted_net_R':threshold,'validation_2023':r,'passed':accepted(r)})
    passing=[trial for trial in trials if trial['passed']]
    chosen=max(passing,key=lambda q:q['validation_2023']['net_usd']) if passing else None
    validation_2024=None;cost_stress_2024=None
    if chosen:
        th=chosen['min_predicted_net_R']
        validation_2024=select(h,pre,side,mask,scores,th,CHECK)
        stressed=base.add_commission(core.precompute(h,m,spec,base.HOLDING_M5,stress=True),base.STRESS_COMMISSION_USD)
        cost_stress_2024=select(h,stressed,side,mask,scores,th,CHECK)
    passed=bool(chosen and accepted(validation_2024) and accepted(cost_stress_2024))
    model_dir=outdir/'models'/symbol
    model_dir.mkdir(parents=True,exist_ok=True)
    model.save_model(model_dir/'event_net_r_RESEARCH_ONLY.json')
    classifier.save_model(model_dir/'event_profit_probability_RESEARCH_ONLY.json')
    metadata={
       'train_years':TRAIN,'threshold_selection_year':TUNE,'independent_forward_test':False,
       'prior_inspection_2023_2024':True,'2025_outcomes_evaluated_or_used_to_select':False,
       'production_enabled':False,'selected_threshold':chosen['min_predicted_net_R'] if chosen else None,
       'features':list(FEATURES),'strict_gate_passed':passed,
       'dual_model_gate':'Regressor and classifier-derived EV must both be positive after costs','classifier_payoffs':payoff,
    }
    (model_dir/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report={'symbol':symbol,'history_range':'2020-2024 outcomes only; no 2025 test',
       'train_event_rows':train_count,'affordable_events_by_year':eligible,
       'setup_diagnostics_all_years':event_diag,
       'no_ml_entry_baseline_2023':baseline_2023,
       'no_ml_entry_baseline_2024':baseline_2024,
       'model_score_distribution':predictions,
       'posthoc_outcome_diagnostics_not_for_selection':outcome_diagnostics,
       'classifier_assumed_payoffs_learned_from_2020_2022':payoff,
       'tuning_2023':trials,'threshold_R':metadata['selected_threshold'],
       'check_2024':validation_2024,'stress_2024':cost_stress_2024,
       'passes_all_gates':passed,'research_only':True}
    print(f"{symbol}: train={train_count}, 2023 baseline={baseline_2023['net_usd']:+.2f} USD, 2024 baseline={baseline_2024['net_usd']:+.2f} USD; 2023 model positives={predictions['2023']['positive_predictions']}; 2023 gate_passes={len(passing)}; LIVE=NO",flush=True)
    return report


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',nargs='?',choices=('run','tests'),default='run')
    ap.add_argument('--zip',type=Path,default=Path('TradeAI_NEW_MARKETS_2020_2025.zip'))
    ap.add_argument('--out',type=Path,default=Path('artifacts/research/event_gate_lab_v2'))
    args=ap.parse_args(argv)
    if args.action=='tests':
        import unittest
        suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent/'tests'),pattern='test_event_gate_lab.py')
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if not args.zip.is_file():raise SystemExit(f'Data archive absent: {args.zip}')
    if args.out.exists() and any(args.out.iterdir()):raise SystemExit(f'Refusing to overwrite existing results: {args.out}')
    args.out.mkdir(parents=True,exist_ok=True)
    report={'experiment':'TradeAI_EVENT_CONDITIONED_ML_ENTRY_RESEARCH_V2',
       'no_order_path':True,'production_models_unchanged':True,
       'validation_note':'2023/2024 were viewed in earlier research, NOT independent forward tests',
       '2025_holdout_status':'NOT EVALUATED; raw archive reader loads full source but no 2025 outcomes used to fit or select',
       'risk_rules':{'starting_usd':core.INITIAL,'max_position_risk_fraction':core.MAX_RISK,
            'max_daily_loss_fraction':core.MAX_DAILY,'max_closed_trade_drawdown_fraction':core.MAX_DD},
       'strategy':{'entry':'four causal H1 setups, no conflicting votes; one trade at a time',
           'model':'XGBRegressor net R + XGBClassifier profit probability on affordable rule-triggered events; require both EV estimates positive',
           'hold_bars_m5':base.HOLDING_M5,'commission_round_trip_proxy_usd':base.SPREAD_COMMISSION_USD,
           'historical_broker_spreads':'M5 bar approximation, not tick fills',
           'unknown_costs':'CFD swap/true commission/tick execution not provided'},
       'symbols':{}}
    with ZipFile(args.zip) as z:
        manifest=base.verify_manifest(z)
        for symbol in SYMBOLS:
            report['symbols'][symbol]=work_one(z,manifest,symbol,args.out)
    report['live_approved']=False
    report['any_passed_all_gates']=any(r['passes_all_gates'] for r in report['symbols'].values())
    (args.out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('REPORT:', args.out/'report.json')
    print('LIVE TRADING APPROVAL: NO. 2025 NOT EVALUATED. NO MT5 ORDERS.')
    return 0

if __name__=='__main__':raise SystemExit(main())
