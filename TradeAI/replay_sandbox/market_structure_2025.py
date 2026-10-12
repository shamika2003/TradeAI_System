"""Isolated H1/M5 stop-path net-R research. NO broker/MT5 API, NO model promotion.

Run: python -m TradeAI.replay_sandbox.market_structure_2025 run --balance 150

Study date policy, locked before test:
  train through 2022 -> validate 2023
  train through 2023 -> validate 2024
  refit through 2024 -> final 2025 report
2026 has been examined in previous research and is NOT an independent holdout.

Historical MT5 M5 bars are BID OHLC, historical `spread` is broker *points*;
there are no ticks, so intrabar stop/target order is unknown (stop first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import holdout_2025 as h

NAME = 'TRADEAI_H1_NET_R_STRUCTURE_V1'
SYMBOLS = ('EURUSD', 'GBPUSD', 'USDJPY')
# Frozen study choices. These are NOT selected using 2025 or 2026 results.
HORIZON_BARS = 24        # 2h maximum, no rollover crossing
STOP_ATR_MULT = 0.85    # Completed H1 ATR; capped to keep small account risk feasible
REWARD_R = 1.8
RISK_FRACTION = 0.015  # 1.5% max planned stop risk including commission, 0.01 min lot
MAX_DRAWDOWN = 0.12    # Hard global simulated drawdown kill-switch
DAILY_LOSS_CAP = 0.03   # Stop admitting entries for the day after 3% closed loss
ASSUMED_LEVERAGE = 100.0
MIN_TRADES_EACH_VALIDATION = 20
PREDICTED_NET_R_GATES = (0.15, 0.25, 0.40, 0.60)
SPREAD_FLOOR = 0.8
BLACKOUT_HOURS = frozenset({0, 23})
START, END = pd.Timestamp('2020-01-01', tz='UTC'), pd.Timestamp('2026-01-01', tz='UTC')
SPLITS = (
    ('2023', pd.Timestamp('2023-01-01',tz='UTC'), pd.Timestamp('2024-01-01',tz='UTC')),
    ('2024', pd.Timestamp('2024-01-01',tz='UTC'), pd.Timestamp('2025-01-01',tz='UTC')),
)
TEST_BEGIN = pd.Timestamp('2025-01-01', tz='UTC')
FEATURES = (
    'h1_ema_20_50_atr','h1_close_ema20_atr','h1_ret_3_atr',
    'h1_ret_12_atr','h1_range_pos_24','h1_atr_rel_50','h1_body_atr',
    'h1_wick_ratio','m5_ret_3_atr','m5_range_atr','m5_rsi_14',
    'm5_relative_vol','spread_pips','hour_sin','hour_cos','weekday_sin','weekday_cos',
)


def sha(path: Path) -> str:
    z=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): z.update(b)
    return z.hexdigest()


def spec_for(profile: dict, sym: str) -> dict:
    p=dict(profile['symbols'][sym]);p['symbol']=sym
    for k in ('point','digits','volume_min','volume_step','volume_max','commission_per_lot','slippage_pips'):
        if k not in p:raise RuntimeError(f'{sym} missing broker field {k}')
    if float(p['volume_min'])!=0.01:raise ValueError('Study requires 0.01 broker min lot')
    return p


def pip_size(spec: dict) -> float:
    return float(spec['point'])*(10 if int(spec['digits']) in (3,5) else 1)


def pip_usd(spec: dict, exit_price: float, lots: float) -> float:
    contract=float(spec.get('contract_size') or 100000)
    if spec['symbol'] in ('EURUSD','GBPUSD'): return contract*lots*pip_size(spec)
    if spec['symbol']=='USDJPY':return contract*lots*pip_size(spec)/exit_price
    raise ValueError('Only EURUSD, GBPUSD, USDJPY')


def _load(workspace: Path, sym: str, tf: str) -> pd.DataFrame:
    p=workspace/'artifacts'/'datasets'/'direction_m5_raw'/f'{sym}_{tf}.csv.gz'
    if not p.is_file():raise FileNotFoundError(p)
    x=pd.read_csv(p,compression='gzip',usecols=['time','open','high','low','close','tick_volume','spread'])
    x['time']=pd.to_datetime(x['time'],utc=True,errors='raise')
    x=x.loc[(x.time>=START)&(x.time<END)].copy().sort_values('time').reset_index(drop=True)
    if x.time.duplicated().any() or not x.time.is_monotonic_increasing:raise ValueError(f'{sym} {tf} invalid times')
    if ((x.low<=0)|(x.high<x[['open','close','low']].max(axis=1))|
        (x.low>x[['open','close','high']].min(axis=1))|(x.spread<0)).any():
        raise ValueError(f'{sym} {tf} invalid OHLC/spread')
    return x


def _h1_features(h1:pd.DataFrame) -> pd.DataFrame:
    """H1 rows become observable at time+1h; never use open H1 candles."""
    x=h1.copy(); c=x['close']; prev=c.shift(1)
    tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    e20=c.ewm(span=20,adjust=False,min_periods=20).mean()
    e50=c.ewm(span=50,adjust=False,min_periods=50).mean()
    hi=x.high.shift(1).rolling(24,min_periods=24).max()
    lo=x.low.shift(1).rolling(24,min_periods=24).min()
    span=(hi-lo).clip(lower=1e-9)
    a=pd.DataFrame({
        'available_at':x.time+pd.Timedelta(hours=1),
        'h1_atr_pips_raw':atr,
        'h1_ema_20_50_atr':(e20-e50)/atr,
        'h1_close_ema20_atr':(c-e20)/atr,
        'h1_ret_3_atr':(c-c.shift(3))/atr,
        'h1_ret_12_atr':(c-c.shift(12))/atr,
        'h1_range_pos_24':(c-lo)/span,
        'h1_atr_rel_50':atr/atr.rolling(50,min_periods=50).median(),
        'h1_body_atr':(c-x.open)/atr,
        'h1_wick_ratio':(x.high-x.low)/(atr.clip(lower=1e-9)),
    })
    # Reject rows built on discontinuous H1 history (weekends/holiday gaps).
    contiguous=(x.time-x.time.shift(1)==pd.Timedelta(hours=1))
    good=contiguous.rolling(15,min_periods=15).sum()==15
    cols=list(a.columns.difference(['available_at']))
    a.loc[~good,cols]=np.nan
    return a.sort_values('available_at').reset_index(drop=True)


def _m5_features(m5:pd.DataFrame) -> pd.DataFrame:
    c=m5.close; prev=c.shift(1)
    tr=pd.concat([m5.high-m5.low,(m5.high-prev).abs(),(m5.low-prev).abs()],axis=1).max(axis=1)
    a=tr.ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    delta=c.diff();up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False,min_periods=14).mean()
    rsi=100-100/(1+up/(down+1e-12))
    vol=m5.tick_volume.astype(float)
    out=pd.DataFrame({
        'decision_time':m5.time+pd.Timedelta(minutes=5),
        'm5_atr_raw':a,
        'm5_ret_3_atr':(c-c.shift(3))/a,
        'm5_range_atr':(m5.high-m5.low)/a,
        'm5_rsi_14':rsi/100,
        'm5_relative_vol':vol/(vol.rolling(96,min_periods=96).median()+1),
        'last_spread_points':m5.spread,
    })
    # Risk features must reflect uninterrupted recent 5m history.
    consecutive=(m5.time.diff()==pd.Timedelta(minutes=5))
    valid=consecutive.rolling(14,min_periods=14).sum()==14
    out.loc[~valid,['m5_atr_raw','m5_ret_3_atr','m5_range_atr','m5_rsi_14','m5_relative_vol']]=np.nan
    return out


def _cost_spread(points:float,spec:dict) -> float:
    return max(SPREAD_FLOOR,float(points)*float(spec['point'])/pip_size(spec))


def _trade_path(arr:dict[str,np.ndarray], idx:int, side:str, stop_pips:float, spec:dict,
                horizon:int=HORIZON_BARS, lots:float=0.01) -> dict:
    """Simulate single BUY/SELL against BID candles and ASK=BID+spread.

    Stop wins ambiguous candles; gaps through stop fill at adverse opening quote.
    Spread/slippage are realized from the actual future bar ONLY for historical labels.
    """
    pip=pip_size(spec);slip=float(spec['slippage_pips'])*pip
    open0=float(arr['open'][idx]);entryspread=_cost_spread(arr['spread'][idx],spec)*pip
    buy=side=='BUY'
    entry=open0+entryspread+slip if buy else open0-slip
    stop=entry + (-stop_pips if buy else stop_pips)*pip
    target=entry + (stop_pips*REWARD_R if buy else -stop_pips*REWARD_R)*pip
    exitprice=None;reason='TIME';end_idx=idx+horizon-1
    for j in range(idx,idx+horizon):
        barsp=_cost_spread(arr['spread'][j],spec)*pip
        start_bid=float(arr['open'][j]);start_close_side=start_bid if buy else start_bid+barsp
        highside=float(arr['high'][j])+(0 if buy else barsp)
        lowside=float(arr['low'][j])+(0 if buy else barsp)
        adverse_open=(start_close_side<=stop) if buy else (start_close_side>=stop)
        if adverse_open:
            exitprice=start_close_side+(-slip if buy else slip)
            reason='GAP_STOP' if j>idx else 'OPEN_SPREAD_STOP';end_idx=j;break
        favorable_open=(start_close_side>=target) if buy else (start_close_side<=target)
        if favorable_open:
            exitprice=target+(-slip if buy else slip);reason='GAP_TAKE';end_idx=j;break
        stop_hit=(lowside<=stop) if buy else (highside>=stop)
        take_hit=(highside>=target) if buy else (lowside<=target)
        if stop_hit:
            exitprice=stop+(-slip if buy else slip);reason='STOP_BOTH' if take_hit else 'STOP';end_idx=j;break
        if take_hit:
            exitprice=target+(-slip if buy else slip);reason='TAKE';end_idx=j;break
    if exitprice is None:
        barsp=_cost_spread(arr['spread'][end_idx],spec)*pip
        exitprice=float(arr['close'][end_idx])+(0 if buy else barsp)+(-slip if buy else slip)
    contract=float(spec.get('contract_size') or 100000)
    direction=1 if buy else -1
    gross=direction*(exitprice-entry)*contract*lots*(1 if spec['symbol'].endswith('USD') else 1/max(exitprice,1e-9))
    commission=lots*float(spec['commission_per_lot'])
    net=gross-commission
    risk=stop_pips*pip_usd(spec,entry,lots)+2*float(spec['slippage_pips'])*pip_usd(spec,entry,lots)+commission
    return {'net_usd':float(net),'net_r':float(net/max(risk,1e-9)), 'reason':reason,
            'entry':float(entry),'exit':float(exitprice),'entry_idx':idx,'exit_idx':end_idx,
            'stop_pips':float(stop_pips),'risk_usd':float(risk), 'lots':float(lots), 'side':side}



def batch_paths(arr:dict[str,np.ndarray],indices:np.ndarray,stops:np.ndarray,side:str,spec:dict,
                horizon:int=HORIZON_BARS) -> dict[str,np.ndarray]:
    """Vectorized stop/target first-touch BID/ASK event simulation, stop wins ties."""
    assert len(indices)==len(stops)
    idx=np.asarray(indices,dtype=np.int64);stops=np.asarray(stops,dtype=float)
    n=len(idx);pip=pip_size(spec);slip=float(spec['slippage_pips'])*pip
    point=float(spec['point']);buy=side=='BUY'
    entry_bid=np.asarray(arr['open'][idx],dtype=float)
    entry_spread=np.maximum(SPREAD_FLOOR,arr['spread'][idx]*point/pip)*pip
    entry=entry_bid+entry_spread+slip if buy else entry_bid-slip
    stop=entry+(-stops if buy else stops)*pip
    take=entry+(stops*REWARD_R if buy else -stops*REWARD_R)*pip
    final=idx+horizon-1
    endspread=np.maximum(SPREAD_FLOOR,arr['spread'][final]*point/pip)*pip
    exitprice=np.asarray(arr['close'][final],dtype=float)+(0 if buy else endspread)+(-slip if buy else slip)
    exitidx=final.copy()
    reason=np.full(n,'TIME',dtype='<U20')
    alive=np.ones(n,dtype=bool)
    for k in range(horizon):
        j=idx+k
        sp=np.maximum(SPREAD_FLOOR,arr['spread'][j]*point/pip)*pip
        o=arr['open'][j]+(0 if buy else sp)
        hi=arr['high'][j]+(0 if buy else sp)
        lo=arr['low'][j]+(0 if buy else sp)
        gstop=(o<=stop) if buy else (o>=stop)
        gtake=(o>=take) if buy else (o<=take)
        istop=(lo<=stop) if buy else (hi>=stop)
        itake=(hi>=take) if buy else (lo<=take)
        # Stop wins if both target and stop touched in a single M5 candle.
        hit=alive & (gstop | gtake | istop | itake)
        if not hit.any():continue
        gs=hit & gstop
        gt=hit & ~gstop & gtake
        st=hit & ~gstop & ~gtake & istop
        tp=hit & ~gstop & ~gtake & ~istop & itake
        exitprice[gs]=o[gs]+(-slip if buy else slip)
        exitprice[gt]=take[gt]+(-slip if buy else slip)
        exitprice[st]=stop[st]+(-slip if buy else slip)
        exitprice[tp]=take[tp]+(-slip if buy else slip)
        exitidx[hit]=j[hit]
        reason[gs]='GAP_STOP' if k else 'OPEN_SPREAD_STOP'
        reason[gt]='GAP_TAKE'
        reason[st]='STOP'
        reason[st & itake]='STOP_BOTH'
        reason[tp]='TAKE'
        alive[hit]=False
        if not alive.any():break
    lot=0.01;contract=float(spec.get('contract_size') or 100000)
    convert=np.ones(n) if spec['symbol'].endswith('USD') else 1/np.maximum(exitprice,1e-9)
    net=((exitprice-entry)*(1 if buy else -1)*contract*lot*convert-
         lot*float(spec['commission_per_lot']))
    risk=stops*contract*lot*pip*(1 if spec['symbol'].endswith('USD') else 1/entry)
    risk+=(2*float(spec['slippage_pips'])*contract*lot*pip*
           (1 if spec['symbol'].endswith('USD') else 1/entry))
    risk+=lot*float(spec['commission_per_lot'])
    return {'net_usd':net,'net_r':net/np.maximum(risk,1e-9),'reason':reason,
            'entry':entry,'exit':exitprice,'exit_idx':exitidx,'stop_pips':stops,'risk_usd':risk}

def create_rows(workspace:Path,sym:str,spec:dict) -> pd.DataFrame:
    m5=_load(workspace,sym,'M5');h1=_load(workspace,sym,'H1')
    if len(m5)<100_000 or len(h1)<5000:raise RuntimeError(f'{sym}: insufficient data')
    mf=_m5_features(m5); hf=_h1_features(h1)
    allfeatures=pd.merge_asof(mf.sort_values('decision_time'),hf,
                  left_on='decision_time',right_on='available_at',direction='backward',allow_exact_matches=True)
    allfeatures['spread_pips']=np.maximum(allfeatures['last_spread_points'].to_numpy()*float(spec['point'])/pip_size(spec),SPREAD_FLOOR)
    t=allfeatures.decision_time
    allfeatures['hour_sin']=np.sin(2*np.pi*t.dt.hour/24)
    allfeatures['hour_cos']=np.cos(2*np.pi*t.dt.hour/24)
    allfeatures['weekday_sin']=np.sin(2*np.pi*t.dt.dayofweek/7)
    allfeatures['weekday_cos']=np.cos(2*np.pi*t.dt.dayofweek/7)
    # H1 decisions only: last fully closed M5 bar is xx:55, H1 is available at xx+1:00.
    eligible=(t.dt.minute==0)&(allfeatures.available_at==t)
    # Reject ANY rollover exposure across maximum 2h holding window.
    for z in range(HORIZON_BARS//12+1):
        eligible &= ~((t+pd.Timedelta(hours=z)).dt.hour.isin(BLACKOUT_HOURS))
    # Costs are broker-local; caps based on fixed normal-session economic budget,
    # not derived from a previously inspected 2026 test.
    eligible &= allfeatures.spread_pips<=3.0
    # Future bar continuity check, so no weekend holiday crossing.
    dt=m5.time
    # Features at completed M5 index i are visible at entry OPEN of i+1.
    idx=np.flatnonzero(eligible.to_numpy()) + 1
    idx=idx[(idx>120)&(idx+HORIZON_BARS<=len(m5))]
    if not len(idx):raise RuntimeError(f'{sym} no entry candidates')
    starts=dt.iloc[idx].reset_index(drop=True)
    # Each candidate decision is the ENTRY M5 open; needs a continuous complete horizon.
    ends=dt.iloc[idx+HORIZON_BARS-1].reset_index(drop=True)
    expected=starts+pd.Timedelta(minutes=5*(HORIZON_BARS-1))
    idx=idx[(ends==expected).to_numpy()]
    arr={k:m5[k].to_numpy() for k in ('open','high','low','close','spread')}
    # Do not build features from or enter on missing next M5 candle.
    previous_idx=idx-1
    aligned=(m5.time.iloc[idx].reset_index(drop=True).equals(
        allfeatures.decision_time.iloc[previous_idx].reset_index(drop=True)))
    if not aligned:
        # For weekend/holiday gaps, filter individual rows rather than refusing the whole study.
        good=(m5.time.iloc[idx].reset_index(drop=True).to_numpy()==
              allfeatures.decision_time.iloc[previous_idx].reset_index(drop=True).to_numpy())
        idx=idx[good];previous_idx=idx-1
    x=allfeatures
    feats=x.iloc[previous_idx][list(FEATURES)].to_numpy(dtype=float)
    good=np.isfinite(feats).all(axis=1)
    idx=idx[good];previous_idx=idx-1
    if not len(idx):raise RuntimeError(f'{sym}: no complete causal H1/M5 rows')
    stops=np.clip(x.iloc[previous_idx]['h1_atr_pips_raw'].to_numpy(dtype=float)/pip_size(spec)*STOP_ATR_MULT,8,35)
    buy=batch_paths(arr,idx,stops,'BUY',spec)
    sell=batch_paths(arr,idx,stops,'SELL',spec)
    rows=x.iloc[previous_idx][['decision_time',*FEATURES]].copy().reset_index(drop=True)
    rows['symbol']=sym
    rows['exit_deadline']=rows.decision_time+pd.Timedelta(minutes=5*HORIZON_BARS)
    for name,values in (('buy',buy),('sell',sell)):
        rows[name+'_net_r']=values['net_r']
        rows[name+'_net_usd']=values['net_usd']
        rows[name+'_exit_at']=m5.time.iloc[values['exit_idx']].reset_index(drop=True)+pd.Timedelta(minutes=5)
        rows[name+'_reason']=values['reason']
        rows[name+'_risk_usd']=values['risk_usd']
        rows[name+'_entry']=values['entry']
    rows['stop_pips']=stops
    if len(rows)<1000:raise RuntimeError(f'{sym}: insufficient candidate samples')
    return rows


def predict_pair(train:pd.DataFrame,valid:pd.DataFrame) -> tuple[np.ndarray,np.ndarray]:
    from xgboost import XGBRegressor
    preds=[]
    for side in ('buy','sell'):
        model=XGBRegressor(n_estimators=220,max_depth=3,learning_rate=0.035,
                 min_child_weight=40,subsample=0.90,colsample_bytree=0.85,
                 reg_lambda=25.0,reg_alpha=0.20,tree_method='hist',random_state=2025,n_jobs=4)
        model.fit(train[list(FEATURES)].astype('float32'),train[side+'_net_r'].clip(-3,3))
        preds.append(np.asarray(model.predict(valid[list(FEATURES)].astype('float32')),dtype=float))
    return preds[0],preds[1]


def simulate(scored:pd.DataFrame,threshold:float,capital:float) -> dict:
    """Single-symbol time-ordered paper accounting, NO simultaneous positions.

    One min lot unless risk permits bigger size; v1 deliberately caps at 0.01.
    Costs, fills and exits are precomputed from raw candles, not model scores.
    """
    balance=capital;peak=capital;maxdd=0;trades=[];reasons={}; next_free=pd.Timestamp('1900-01-01',tz='UTC')
    current_day=None;day_start_balance=capital
    stopped=False
    for r in scored.itertuples(index=False):
        if r.decision_time<next_free: continue
        day=r.decision_time.date()
        if day!=current_day:
            current_day=day;day_start_balance=balance
        if balance <= day_start_balance*(1-DAILY_LOSS_CAP):
            reasons['DAILY_LOSS_CAP']=reasons.get('DAILY_LOSS_CAP',0)+1
            continue
        b,s=float(r.p_buy),float(r.p_sell)
        if not (np.isfinite(b) and np.isfinite(s)) or max(b,s)<threshold:continue
        side='buy' if b>s else 'sell'
        risk=float(getattr(r,side+'_risk_usd'))
        if risk>RISK_FRACTION*balance:
            reasons['RISK_MIN_LOT_TOO_HIGH']=reasons.get('RISK_MIN_LOT_TOO_HIGH',0)+1
            continue
        # Conservative estimate of USD required margin at 100:1 leverage.
        notional=1000.0 if getattr(r,'symbol','EURUSD')=='USDJPY' else 1000*float(getattr(r,side+'_entry'))
        margin=notional/ASSUMED_LEVERAGE
        if balance-risk <= margin:
            reasons['MARGIN_RESERVE_BLOCK']=reasons.get('MARGIN_RESERVE_BLOCK',0)+1
            continue
        # Conservative no-leverage-hunting approach: minimum lot (0.01) only.
        result=float(getattr(r,side+'_net_usd'))
        nxt=getattr(r,side+'_exit_at')
        # Account-level arithmetic is realized at close, not when signal appears.
        balance+=result;peak=max(peak,balance)
        maxdd=max(maxdd,(peak-balance)/peak)
        trades.append({'time':r.decision_time.isoformat(),'exit_at':nxt.isoformat(),
                       'side':side.upper(),'net_usd':round(result,5),'balance':round(balance,5),
                       'predicted_net_r':round(max(b,s),4),'reason':getattr(r,side+'_reason')})
        next_free=nxt
        if balance<1 or maxdd>=MAX_DRAWDOWN:
            stopped=True;break
    pos=[t['net_usd'] for t in trades if t['net_usd']>0]
    neg=[t['net_usd'] for t in trades if t['net_usd']<0]
    pf=sum(pos)/abs(sum(neg)) if neg else None  # Undefined if no losses
    return {'trades':len(trades),'winners':len(pos),'win_rate_pct':round(100*len(pos)/len(trades),2) if trades else None,
            'net_usd':round(balance-capital,4),'ending_balance':round(balance,4),
            'max_drawdown_pct':round(100*maxdd,3),'profit_factor':pf,
            'stopped_by_risk':stopped,'rejections':reasons,'trades_log':trades}


def _fold(rows:pd.DataFrame,train_end:pd.Timestamp,valid_end:pd.Timestamp):
    t=rows.decision_time;e=rows.exit_deadline
    tr=rows.loc[(t<train_end)&(e<=train_end)].copy()
    v=rows.loc[(t>=train_end)&(t<valid_end)&(e<=valid_end)].copy()
    if len(tr)<3000 or len(v)<1000:raise ValueError('Insufficient fold examples')
    return tr,v


def _prepare_study(workspace:Path):
    workspace=h._reject_live_workspace(workspace)
    manifest=h._manifest(workspace)
    if manifest['train_exclusive_end']!='2026-01-01':raise ValueError('Pre-2026 isolation required')
    archive=workspace/'replay_2026_jan_sep.zip'
    if not archive.is_file() or sha(archive)!=manifest['replay_archive_sha256']:
        raise RuntimeError('Original research archive changed')
    from zipfile import ZipFile
    with ZipFile(archive) as z: profile=json.loads(z.read('broker_profile.json'))
    if profile['account_currency']!='USD':raise RuntimeError('Expected USD account')
    return workspace,manifest,profile


def run(workspace:Path,capital:float) -> dict:
    if capital<=0:raise ValueError('Positive balance required')
    workspace,manifest,profile=_prepare_study(workspace)
    dest=workspace/'market_structure_2025_v1'
    if dest.exists():raise FileExistsError(f'Study already exists: {dest}; refusing to overwrite results')
    dest.mkdir(parents=True,exist_ok=False)
    root_report={'study':NAME,'status':'SIMULATION_RESEARCH_ONLY_NOT_LIVE_APPROVED',
       'evidence':'2020-2024 DEVELOPMENT, 2025 FINAL RETROSPECTIVE TEST; 2026 ALREADY INSPECTED',
       'source_manifest_sha256':sha(workspace/'cutoff_manifest.json'),
       'source_archive_sha256':manifest['replay_archive_sha256'],
       'params':{'horizon_bars':HORIZON_BARS,'stop_atr_mult':STOP_ATR_MULT,
                 'reward_to_risk':REWARD_R,'risk_fraction':RISK_FRACTION,
                 'max_drawdown':MAX_DRAWDOWN,'daily_loss_cap':DAILY_LOSS_CAP,
                 'assumed_leverage':ASSUMED_LEVERAGE,'validation_thresholds':PREDICTED_NET_R_GATES,
                 'feature_names':list(FEATURES)},
       'caveats':['Historical BID M5 OHLC and bar-based spreads, not tick/true fills',
          'Stop-first ambiguous intrabar event ordering; gaps approximated',
          'One symbol per account; multi-symbol portfolio interaction not included',
          'No intratrade mark-to-market, margin stopout, financing/swap or latency simulation',
          '2025 test must not be used to adjust model/thresholds without a fresh new test',
          'Never automatically promotes models; no MT5 imports/orders'],
       'symbols':{}}
    (dest/'research_design.json').write_text(json.dumps({k:root_report[k] for k in ('study','params','caveats')},indent=2)+'\n')
    try:
        for sym in SYMBOLS:
            spec=spec_for(profile,sym)
            rows=create_rows(workspace,sym,spec)
            folds={};failed=[]
            fold_scores={}
            for name,start,end in SPLITS:
                tr,va=_fold(rows,start,end)
                b,s=predict_pair(tr,va)
                scored=va.copy();scored['p_buy']=b;scored['p_sell']=s
                fold_scores[name]=scored
                folds[name]={'train':len(tr),'validation':len(va),'score_ranges':{
                  'buy_mean':float(b.mean()),'sell_mean':float(s.mean())}}
            grid=[]
            for gate in PREDICTED_NET_R_GATES:
                fs={}
                for name in fold_scores:
                    result=simulate(fold_scores[name],gate,capital)
                    fs[name]={k:v for k,v in result.items() if k!='trades_log'}
                okay=all(z['trades']>=MIN_TRADES_EACH_VALIDATION and z['net_usd']>0 and
                         ((z['profit_factor'] is not None and z['profit_factor']>1.15) or
                          (z['profit_factor'] is None and z['winners']==z['trades'])) and
                         z['max_drawdown_pct']<10 and not z['stopped_by_risk'] for z in fs.values())
                grid.append({'threshold':gate,'passed':bool(okay),'folds':fs})
            choices=[x for x in grid if x['passed']]
            chosen=choices[0]['threshold'] if choices else None
            test_rows=rows.loc[(rows.decision_time>=TEST_BEGIN)&(rows.exit_deadline<END)].copy()
            if len(test_rows)<1000:raise ValueError('Insufficient final test rows')
            if chosen is not None:
                train_rows=rows.loc[rows.exit_deadline<=TEST_BEGIN].copy()
                b,s=predict_pair(train_rows,test_rows)
                test_rows['p_buy']=b;test_rows['p_sell']=s
                final=simulate(test_rows,chosen,capital)
                final_trades=final.pop('trades_log')
                pd.DataFrame(final_trades).to_csv(dest/f'{sym}_2025_trades.csv',index=False)
            else:
                final={'not_evaluated':True,'reason':'NO_THRESHOLD_PASSED_BOTH_2023_AND_2024'}
            status={'fold_details':folds,'validation_grid':grid,'locked_threshold':chosen,
                'gate_passed':chosen is not None,'2025_final_test':final,
                'n_2025_eligible_rows':len(test_rows),'promotion_to_live':False}
            root_report['symbols'][sym]=status
            (dest/f'{sym}.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
            print(sym,'validation PASS' if chosen else 'validation FAIL',
                  '| locked threshold',chosen,
                  '| 2025',final.get('net_usd','NOT TESTED'),flush=True)
    finally:
        (dest/'report.json').write_text(json.dumps(root_report,indent=2,allow_nan=False)+'\n')
    print('REPORT:',(dest/'report.json').resolve())
    print('SIMULATION ONLY; no orders, no deployment, no model promotion')
    return root_report


def main():
    pa=argparse.ArgumentParser(description=__doc__)
    pa.add_argument('command',choices=['run'])
    pa.add_argument('--workspace',type=Path,default=h.DEFAULT_WORKSPACE)
    pa.add_argument('--balance',type=float,default=150)
    args=pa.parse_args()
    run(args.workspace,args.balance)

if __name__=='__main__':main()
