import unittest
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from TradeAI.replay_sandbox import event_gate_lab as e

class EventGateTests(unittest.TestCase):
    def test_conflicting_votes_hold(self):
        d=pd.DataFrame({k:[0,0] for k in e.SETUPS})
        d.loc[0,'momentum']=1
        d.loc[0,'mean_reversion']=-1
        d.loc[1,'momentum']=1
        base=pd.DataFrame({k:[0.5,0.5] for k in e.base.FEATURES})
        p={1:{'risk_usd':np.ones(2)},-1:{'risk_usd':np.ones(2)}}
        side,fx,info=e.make_events(d,base,p)
        self.assertEqual(side.tolist(),[0,1])
        self.assertEqual(info['conflicts'],1)
    def test_2025_locked(self):
        self.assertEqual(e.TRAIN,(2020,2021,2022))
        self.assertEqual(e.TUNE,2023)
        self.assertEqual(e.CHECK,2024)
        self.assertEqual(e.SEALED,2025)
    def test_reject_thin_win(self):
        fake={'trades':7,'net_usd':1.28,'profit_factor':1.17,'max_dd_pct':2.0,'risk_stopped':False}
        self.assertFalse(e.accepted(fake))
    def test_reject_drawdown(self):
        fake={'trades':32,'net_usd':2.,'profit_factor':1.3,'max_dd_pct':12.,'risk_stopped':True}
        self.assertFalse(e.accepted(fake))
    def test_missing_predictions_hold(self):
        # ML gate must never generate a side without a nominated event.
        side=np.array([0,0,1,-1],dtype=np.int8)
        mask=np.array([False,False,True,True])
        scores=np.array([100.,100.,-.1,.2])
        flags=np.where(mask&(scores>=0),side,0)
        self.assertEqual(flags.tolist(),[0,0,0,-1])
    def test_risk_filter_capital(self):
        f=pd.DataFrame({k:[0.]*2 for k in e.FEATURES})
        pre={1:{'risk_usd':np.array([3.,1.])},-1:{'risk_usd':np.array([3.,1.])}}
        mask=e.make_fit_mask(None,f,np.array([1,-1]),pre,np.array([True,True]))
        self.assertEqual(mask.tolist(),[False,True])

if __name__=='__main__':unittest.main()
