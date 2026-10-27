"""Research-only execution simulator extracted from the verified TradeAI fixed-rule benchmark.
No MT5 import, no account access, no orders. Historical candle costs are proxies.
"""
"""Transparent historical hypothesis test; NO MT5 orders; new market OHLC only.
2020-2022 setup development, 2023 & 2024 walk-forward screening; 2025 sealed unless validation passes.
Research simplifications: M5 bar spread proxy, no ticks/swap, conservative same-bar stop first.
"""
import zipfile,gzip,json,hashlib,math,os
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
ZIP=Path(os.environ.get('TRADEAI_MARKET_ZIP', str(BASE_DIR/'TradeAI_NEW_MARKETS_2020_2025.zip')))
OUT=Path(os.environ.get('TRADEAI_RESEARCH_OUT', str(BASE_DIR/'new_market_research')))
# Import-safe: no directories are created by this library until main() is requested.
SYMBOLS=['GOLD','US500Cash','US100Cash','US30Cash','SILVER']
def unix_seconds(timestamps):
    """Return UTC epoch seconds independent of pandas datetime storage unit.

    Pandas 3 can parse CSV timestamps to datetime64[us] instead of ns.
    Dividing astype(int64) by 1e9 is correct only for nanosecond storage.
    Converting the DatetimeIndex to seconds first works with ns/us/ms.
    """
    return pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).as_unit("s").asi8


HORIZONS=[12,24,48] # M5 count 1,2,4 h
STRATS=['trend_breakout','trend_pullback','mean_reversion','momentum']
INITIAL=150.0
MAX_DD=.12
MAX_DAILY=.04
MAX_RISK=.015
RISK_STOP_ATR=.7
TP_MULT=1.8
START_HOUR=12
LAST_HOUR=18
# hypothetical adverse per-side slippage in price units. Not historical fills.
SLIP={'GOLD':.05,'US500Cash':.08,'US100Cash':.30,'US30Cash':.40,'SILVER':.015}


def load(z,s,tf):
 with gzip.GzipFile(fileobj=z.open(f'market_data/{s}_{tf}.csv.gz')) as f: d=pd.read_csv(f,parse_dates=['time'])
 d=d.sort_values('time').reset_index(drop=True)
 return d

def add_signals(h):
 c=h.close; op=h.open; hi=h.high; lo=h.low
 prev=c.shift(1)
 h['atr']=pd.concat([hi-lo,(hi-prev).abs(),(lo-prev).abs()],axis=1).max(axis=1).ewm(alpha=1/14,adjust=False,min_periods=30).mean()
 h['ema20']=c.ewm(span=20,adjust=False,min_periods=40).mean()
 h['ema50']=c.ewm(span=50,adjust=False,min_periods=60).mean()
 h['avg_high20']=hi.shift(1).rolling(20,min_periods=20).max()
 h['avg_low20']=lo.shift(1).rolling(20,min_periods=20).min()
 r=c.diff(); rise=r.clip(lower=0).ewm(alpha=1/14,adjust=False,min_periods=30).mean(); fall=-r.clip(upper=0).ewm(alpha=1/14,adjust=False,min_periods=30).mean(); h['rsi']=100*rise/(rise+fall)
 h['delta3']=(c-c.shift(3))/h.atr
 h['z']=(c-h.ema20)/h.atr
 h['closeprev']=c.shift(1); h['ema20prev']=h.ema20.shift(1)
 h['valid_h1_prev']=(h.time - h.time.shift(1)==pd.Timedelta(hours=1))
 h['time_entry']=h.time + pd.Timedelta(hours=1)
 # signals are based solely on COMPLETED H1 at h.time + 1 hour (entry).
 tr_up=(h.ema20>h.ema50); tr_dn=(h.ema20<h.ema50)
 h['trend_breakout']=np.where(tr_up & (c>h.avg_high20),1,np.where(tr_dn & (c<h.avg_low20),-1,0))
 h['trend_pullback']=np.where(tr_up & (h.closeprev<=h.ema20prev)&(c>h.ema20),1,np.where(tr_dn & (h.closeprev>=h.ema20prev)&(c<h.ema20),-1,0))
 h['mean_reversion']=np.where((h.z<-1.5)&(h.rsi<32),1,np.where((h.z>1.5)&(h.rsi>68),-1,0))
 h['momentum']=np.where((h.delta3>1.2)&tr_up,1,np.where((h.delta3<-1.2)&tr_dn,-1,0))
 # avoid H1 session breaks in most recent 3 hours
 h['valid']=h['valid_h1_prev']&(h.time.shift(1)-h.time.shift(2)==pd.Timedelta(hours=1))&(h.time.shift(2)-h.time.shift(3)==pd.Timedelta(hours=1))
 return h

def prepare(z,manifest,s):
 spec=manifest['symbols'][s]
 h=add_signals(load(z,s,'H1')); m=load(z,s,'M5')
 t_m=unix_seconds(m.time); t_h=unix_seconds(h.time_entry)
 indices=np.searchsorted(t_m,t_h); safe=np.minimum(indices,len(m)-1)
 known=(indices<len(m))&(t_m[safe]==t_h)
 prev=np.maximum(0,safe-1)
 prior_spread=m.spread.to_numpy(dtype=float)[prev]*float(spec['point'])
 floor=float(m.spread.loc[(m.spread>0)&(m.time.dt.year<=2022)].quantile(.05))*float(spec['point'])
 spr=np.maximum(prior_spread,floor)
 # remove extreme prev-spread (use only 2020-2022 calibrated 95th percentile spread)
 cap=float(m.spread.loc[(m.spread>0)&(m.time.dt.year<=2022)].quantile(.95))*float(spec['point'])
 # hourly starts need preceding fully closed M5 spread record
 prior_ok=(t_m[safe]-t_m[prev]==300)
 mask=known & prior_ok & h.valid.to_numpy() & np.isfinite(h.atr.to_numpy()) & (spr<=cap)
 mask &= (h.time_entry.dt.hour>=START_HOUR).to_numpy() & (h.time_entry.dt.hour<=LAST_HOUR).to_numpy()
 # don't include rollover in 4h windows and require complete continuous M5 for every requested exit horizon
 entry_idx=safe[mask]; hh=h.loc[mask].copy().reset_index(drop=True); hh['mi']=entry_idx; hh['spread_entry']=spr[mask]
 return hh,m,spec,{'spread_floor_price_units':floor,'spread_cap_price_units':cap,'raw_rows_m5':len(m),'hour_decisions':len(hh)}

def precompute(h,m,spec,holding,stress=False):
 # Vectorized 5min path outcomes for every hourly decision (both BUY and SELL).
 ix=h.mi.to_numpy(dtype=int)
 if not len(ix):return None
 n=len(m); ti=unix_seconds(m.time); idx=np.minimum(ix[:,None]+np.arange(holding)[None,:],n-1)
 full=(ix+holding-1<n)&(ti[np.minimum(ix+holding-1,n-1)]-ti[ix]==(holding-1)*300)
 valid=h['atr'].to_numpy()>0
 valid &= full
 # reject windows including 23:00-00:59 or crossing session break
 end_t=ti[np.minimum(ix+holding-1,n-1)]+300
 end_hours=pd.to_datetime(end_t,unit='s',utc=True).hour
 valid &= (end_hours != 23)&(end_hours != 0)
 b_open=m.open.to_numpy(dtype=float); b_close=m.close.to_numpy(dtype=float); b_high=m.high.to_numpy(dtype=float); b_low=m.low.to_numpy(dtype=float)
 spread=np.maximum(m.spread.to_numpy(dtype=float)*float(spec['point']),0)
 # pandas copy-on-write can expose a read-only view: stress must never
 # modify the source market data or a shared DataFrame-backed array.
 start_spread=h.spread_entry.to_numpy(dtype=float,copy=True)
 if stress:start_spread*=1.5
 slip=SLIP[spec['name']]*(2 if stress else 1)
 units=float(spec['volume_min'])*float(spec['trade_contract_size'])
 stop=h.atr.to_numpy(dtype=float)*RISK_STOP_ATR
 # optional spread-related guard: full round-turn cost should not exceed 20% ATR stop
 cost_est=(start_spread+2*slip)*units
 max_risk=(stop+start_spread+2*slip)*units
 valid &= (start_spread+2*slip <= stop*.2)
 # enforce too tight stop vs recorded spread at entry
 buy_entry=b_open[ix]+start_spread+slip
 sell_entry=b_open[ix]-slip
 # consistent reference stops and targets based on BID price, cost deducted through actual fill prices
 results={}
 for side in (1,-1):
  # Stop/TP price levels relative to open, so realized R uses actual fill cost separately.
  stop_bid=b_open[ix]-side*stop
  take_bid=b_open[ix]+side*stop*TP_MULT
  hi=b_high[idx]; lo=b_low[idx]; op=b_open[idx]
  # sell closes at ask = bid + recorded spread (bar-associated approximate); plus adverse slip
  spr_path=np.maximum(spread[idx],start_spread[:,None]*.5)
  if stress:spr_path*=1.5
  if side==1:
   stop_hit=lo<=stop_bid[:,None]; take_hit=hi>=take_bid[:,None]
   gap_hit=op<stop_bid[:,None]
  else:
   ask_high=hi+spr_path; ask_low=lo+spr_path; ask_open=op+spr_path
   stop_ask=b_open[ix]+stop
   take_ask=b_open[ix]-stop*TP_MULT
   stop_hit=ask_high>=stop_ask[:,None];take_hit=ask_low<=take_ask[:,None]
   gap_hit=ask_open>stop_ask[:,None]
  events=stop_hit|take_hit
  first=np.where(events.any(axis=1),events.argmax(axis=1),holding-1)
  hit_stop=stop_hit[np.arange(len(ix)),first]
  hit_take=(~hit_stop)&take_hit[np.arange(len(ix)),first]
  exit_ix=idx[np.arange(len(ix)),first]
  if side==1:
   # Stop first; stop gap slips beyond stop price at bar open
   stop_px=np.minimum(stop_bid,b_open[exit_ix])
   exit_px=np.where(hit_stop,stop_px,np.where(hit_take,take_bid,b_close[exit_ix]))-slip
   net=(exit_px-buy_entry)*units
  else:
   exit_spr=np.maximum(spread[exit_ix],start_spread*.5)*(1.5 if stress else 1)
   stop_px=np.maximum(b_open[ix]+stop,b_open[exit_ix]+exit_spr)
   exit_px=np.where(hit_stop,stop_px,np.where(hit_take,b_open[ix]-stop*TP_MULT,b_close[exit_ix]+exit_spr))+slip
   net=(sell_entry-exit_px)*units
  # extra stress commission per completed min-lot order; commission otherwise UNKNOWN.
  net-=(.20 if stress else 0)
  reason=np.where(hit_stop,'STOP',np.where(hit_take,'TAKE','TIME'))
  results[side]={'net':net,'exit_time':end_t if holding==1 else ti[exit_ix]+300,'valid':valid,'reason':reason,'cost_est':cost_est,'risk_usd':max_risk}
 return results

def simulate(h,pre,side_flags,year,stress=False):
 sel=np.flatnonzero((h.time_entry.dt.year.to_numpy()==year)&(side_flags!=0))
 balance=INITIAL;peak=INITIAL; day_start=INITIAL;cur_day=None; busy_until=0
 ddmax=0.0; n=0; pos=0;neg=0;gross_pos=0.; gross_neg=0.;sumusd=0.; stop_count=0; cap_reject=0; risk_reject=0; daily_reject=0; trades=[]; stopped=False
 days=set()
 for k in sel:
  entry_ts=int(h.time_entry.iloc[k].timestamp()); day=entry_ts//86400
  if cur_day!=day:cur_day=day;day_start=balance
  if entry_ts<busy_until:continue
  if (day_start-balance)/max(day_start,1e-9) >= MAX_DAILY: daily_reject+=1;continue
  sd=int(side_flags[k]); p=pre[sd]
  if (not p['valid'][k]) or (p['risk_usd'][k] > balance*MAX_RISK):risk_reject+=1;continue
  res=float(p['net'][k]); balance+=res;sumusd+=res;n+=1;busy_until=int(p['exit_time'][k]); days.add(day)
  gross_pos+=max(res,0);gross_neg+=max(-res,0)
  pos+=res>0;neg+=res<=0;stop_count+=p['reason'][k]=='STOP'
  trades.append({'time':h.time_entry.iloc[k].isoformat(),'net_usd':round(res,4),'reason':str(p['reason'][k]),'equity':round(balance,4)})
  peak=max(peak,balance);ddmax=max(ddmax,(peak-balance)/peak)
  if ddmax>=MAX_DD:stopped=True;break
 return {'year':year,'trades':n,'net_usd':round(sumusd,4),'ending_usd':round(balance,4),'profit_factor':round(gross_pos/gross_neg,4) if gross_neg else (None if not n else 999.0),'win_rate_pct':round(pos/n*100,2) if n else None,'max_dd_pct':round(ddmax*100,3),'risk_stopped':stopped,'active_days':len(days),'stops':stop_count,'entry_signals':len(sel),'skip_risk_cost_missing':risk_reject,'skip_daily_cap':daily_reject,'trades_log':trades}

def passed(result):
 return result['trades']>=30 and result['net_usd']>0 and (result['profit_factor'] or 0)>=1.15 and result['max_dd_pct']<=10 and not result['risk_stopped']

