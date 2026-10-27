"""Unit tests for the research-only US index model and replay rules."""
import unittest
import numpy as np
import pandas as pd
from TradeAI.replay_sandbox import _broker_replay_core as core
from TradeAI.replay_sandbox.regime_edge_lab import (
    FEATURES, TRAIN_YEARS, TUNE_YEAR, CHECK_YEAR, SEALED_YEAR,
    actions_from_scores, add_commission, acceptable, features_from_eligible,
)

class TestModelSafety(unittest.TestCase):
    def test_threshold_requires_positive_prediction(self):
        flags=actions_from_scores(np.array([.14,-.2,.06,.10]),np.array([.0,.14,.06,.08]),.10)
        self.assertEqual(flags.tolist(),[1,-1,0,0])

    def test_fee_nonzero_and_source_unchanged(self):
        p={1:{'net':np.array([1.,-1.]),'risk_usd':np.array([2.,2.])},
           -1:{'net':np.array([2.,-2.]),'risk_usd':np.array([2.,2.])}}
        q=add_commission(p,.20)
        self.assertAlmostEqual(q[1]['net'][0],.8)
        self.assertAlmostEqual(q[-1]['risk_usd'][0],2.2)
        self.assertAlmostEqual(p[1]['net'][0],1.)

    def test_strict_validation_does_not_accept_thin_profit(self):
        good={'trades':50,'net_usd':4.0,'profit_factor':1.2,'max_dd_pct':4.0,'risk_stopped':False}
        self.assertTrue(acceptable(good))
        self.assertFalse(acceptable({**good,'trades':4}))
        self.assertFalse(acceptable({**good,'net_usd':-.01}))
        self.assertFalse(acceptable({**good,'risk_stopped':True}))
        self.assertFalse(acceptable({**good,'profit_factor':1.0}))

    def test_chronology_fixed_and_2025_sealed(self):
        self.assertEqual(TRAIN_YEARS,(2020,2021,2022))
        self.assertLess(max(TRAIN_YEARS),TUNE_YEAR)
        self.assertLess(TUNE_YEAR,CHECK_YEAR)
        self.assertLess(CHECK_YEAR,SEALED_YEAR)

    def test_features_use_closed_h1_values(self):
        t=pd.to_datetime(['2022-01-03T15:00:00Z'])
        d=pd.DataFrame({'time':pd.to_datetime(['2022-01-03T14:00:00Z']),
            'time_entry':t,'atr':[10.], 'open':[100.], 'high':[110.], 'low':[95.],
            'close':[105.], 'ema20':[102.], 'ema50':[101.], 'rsi':[54.],
            'delta3':[.3], 'z':[.3], 'avg_high20':[108.], 'avg_low20':[94.],
            'tick_volume':[80], 'spread_entry':[.6], 'closeprev':[100.]})
        x=features_from_eligible(d)
        self.assertEqual(set(x.columns),set(FEATURES))
        self.assertAlmostEqual(x.body_atr.iloc[0],.5)
        self.assertAlmostEqual(x.entry_spread_to_atr.iloc[0],.06)
        self.assertAlmostEqual(x.past_1h_return_atr.iloc[0],.5)

    def test_epoch_conversion_independent_of_ns_us_ms(self):
        raw = pd.Series(pd.to_datetime([
            "2023-05-02T12:00:00Z", "2023-05-02T12:05:00Z",
            "2023-05-02T12:10:00Z",
        ], utc=True))
        expected = None
        for unit in ("ns", "us", "ms"):
            values = core.unix_seconds(raw.dt.as_unit(unit))
            self.assertEqual(np.diff(values).tolist(), [300, 300])
            if expected is None:
                expected = values
            else:
                np.testing.assert_array_equal(values, expected)

    def test_precompute_microsecond_timestamps(self):
        ti = pd.date_range("2023-05-02T12:00:00Z", periods=25, freq="5min").as_unit("us")
        bars = pd.DataFrame({"time":ti, "open":100., "close":100.,
                             "high":100., "low":100., "spread":5})
        hour = pd.DataFrame({"time_entry":[ti[0]], "mi":[0],
                             "spread_entry":[.05], "atr":[4.]})
        spec = {"name":"US500Cash", "volume_min":.1,
                "trade_contract_size":1., "point":.01}
        result = core.precompute(hour, bars, spec, 12)
        self.assertIsNotNone(result)
        self.assertTrue(bool(result[1]["valid"][0]))
        self.assertLess(float(result[1]["net"][0]), 0.0)

    def test_same_candle_stop_first(self):
        ti=pd.date_range('2023-05-02T12:00:00Z',periods=25,freq='5min')
        bars=pd.DataFrame({'time':ti,'open':100.,'close':100.,'high':100.,'low':100.,'spread':5})
        bars.loc[0,'high']=110.;bars.loc[0,'low']=90.
        hour=pd.DataFrame({'time_entry':[ti[0]], 'mi':[0], 'spread_entry':[.05], 'atr':[4.]})
        spec={'name':'US500Cash','volume_min':.1,'trade_contract_size':1.,'point':.01}
        p=core.precompute(hour,bars,spec,12)
        self.assertEqual(p[1]['reason'][0],'STOP')
        self.assertEqual(p[-1]['reason'][0],'STOP')
        self.assertLess(p[1]['net'][0],0)
        self.assertLess(p[-1]['net'][0],0)

    def test_constant_market_loses_after_costs(self):
        ti=pd.date_range('2023-05-02T12:00:00Z',periods=25,freq='5min')
        bars=pd.DataFrame({'time':ti,'open':100.,'close':100.,'high':100.,'low':100.,'spread':50})
        hour=pd.DataFrame({'time_entry':[ti[0]],'mi':[0],'spread_entry':[.5],'atr':[10.]})
        spec={'name':'US500Cash','volume_min':.1,'trade_contract_size':1.,'point':.01}
        p=core.precompute(hour,bars,spec,12)
        self.assertLess(p[1]['net'][0],0)
        self.assertLess(p[-1]['net'][0],0)

if __name__=='__main__':unittest.main()
