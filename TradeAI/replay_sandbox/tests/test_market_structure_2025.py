"""Run: python -m unittest TradeAI.replay_sandbox.tests.test_market_structure_2025"""
import unittest
import numpy as np
import pandas as pd
from TradeAI.replay_sandbox.market_structure_2025 import (
    FEATURES,_trade_path,_h1_features,_m5_features,simulate,pip_size,
)

SPEC=dict(symbol='EURUSD',point=0.00001,digits=5,volume_min=0.01,
          volume_step=0.01,volume_max=50,commission_per_lot=7.,slippage_pips=0.2)

class MarketStructureTests(unittest.TestCase):
    def mkt(self, values, spread=10):
        a=np.asarray(values,dtype=float)
        return dict(open=a,high=a+0.00002,low=a-0.00002,close=a,spread=np.full(len(a),spread))

    def test_no_orders_no_mt5_import(self):
        import inspect
        from TradeAI.replay_sandbox import market_structure_2025
        source=inspect.getsource(market_structure_2025)
        self.assertNotIn('import MetaTrader5',source)
        self.assertEqual(len(FEATURES),17)

    def test_broker_pip(self):
        self.assertAlmostEqual(pip_size(SPEC),0.0001)

    def test_profitable_target_and_commission(self):
        arr=self.mkt([1.10000]+[1.10250]*24,spread=10)
        t=_trade_path(arr,0,'BUY',10,SPEC)
        self.assertEqual(t['reason'],'GAP_TAKE')
        self.assertGreater(t['net_usd'],0)
        self.assertEqual(t['exit_idx'],1)

    def test_stop_first_when_both_touched(self):
        arr=self.mkt([1.10000]*24,spread=10)
        arr['high'][0]=1.10250;arr['low'][0]=1.09700
        t=_trade_path(arr,0,'BUY',10,SPEC)
        self.assertEqual(t['reason'],'STOP_BOTH')
        self.assertLess(t['net_usd'],0)

    def test_spread_cost_visible(self):
        a=self.mkt([1.10000]*24,spread=40)
        t=_trade_path(a,0,'BUY',10,SPEC)
        self.assertLess(t['net_usd'],0)

    def test_gap_stop_worse_than_planned(self):
        a=self.mkt([1.10000]+[1.09500]*23,spread=10)
        t=_trade_path(a,0,'BUY',10,SPEC)
        self.assertEqual(t['reason'],'GAP_STOP')
        self.assertLess(t['net_usd'],-2)

    def test_h1_feature_availability(self):
        n=160
        start=pd.Timestamp('2022-01-03',tz='UTC')
        t=pd.date_range(start,periods=n,freq='h')
        c=1.2+0.0001*np.arange(n)
        x=pd.DataFrame(dict(time=t,open=c,high=c+0.0002,low=c-0.0002,close=c+0.00002))
        h=_h1_features(x)
        self.assertEqual(h.available_at.iloc[0],start+pd.Timedelta(hours=1))
        self.assertTrue(np.isfinite(h.h1_ema_20_50_atr.iloc[-1]))

    def test_m5_decision_uses_closed_candle(self):
        n=130;t=pd.date_range('2022-01-03',periods=n,freq='5min',tz='UTC');c=1.2+0.00001*np.arange(n)
        d=pd.DataFrame(dict(time=t,open=c,high=c+0.0001,low=c-0.0001,close=c+0.00002,spread=1,tick_volume=100))
        mf=_m5_features(d)
        self.assertEqual(mf.decision_time.iloc[-1],t[-1]+pd.Timedelta(minutes=5))

    def test_risk_abstain(self):
        t=pd.Timestamp('2023-03-01 02:00',tz='UTC')
        df=pd.DataFrame([dict(decision_time=t,p_buy=0.5,p_sell=0.1,buy_risk_usd=20.,
           sell_risk_usd=20.,buy_net_usd=3.,sell_net_usd=-2.,buy_exit_at=t+pd.Timedelta(hours=1),
           sell_exit_at=t+pd.Timedelta(hours=1),buy_reason='TAKE',sell_reason='STOP')])
        out=simulate(df,0.25,150)
        self.assertEqual(out['trades'],0)
        self.assertEqual(out['rejections'].get('RISK_MIN_LOT_TOO_HIGH'),1)

if __name__=='__main__':unittest.main()

class ParityTests(unittest.TestCase):
    def test_vectorized_matches_scalar_bid_ask_fills(self):
        from TradeAI.replay_sandbox.market_structure_2025 import batch_paths
        r=np.random.default_rng(42);n=80
        px=1.105+np.cumsum(r.normal(0,0.00015,n))
        opened=px+r.normal(0,0.00003,n)
        z=dict(open=opened,high=np.maximum(px,opened)+0.0004,
               low=np.minimum(px,opened)-0.0004,close=px,spread=r.integers(0,80,n))
        ix=np.array([1,5,12,23,37,45]);stops=np.array([8,10,14,25,33,12.])
        for sd in ('BUY','SELL'):
            bat=batch_paths(z,ix,stops,sd,SPEC)
            for j,i in enumerate(ix):
                one=_trade_path(z,i,sd,stops[j],SPEC)
                for key in ('entry','exit','risk_usd','net_usd','net_r'):
                    self.assertAlmostEqual(bat[key][j],one[key],places=8)
                self.assertEqual(bat['exit_idx'][j],one['exit_idx'])
                self.assertEqual(bat['reason'][j],one['reason'])
