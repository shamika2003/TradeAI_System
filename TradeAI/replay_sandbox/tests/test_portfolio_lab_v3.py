"""Synthetic integrity tests of integrated V3 research selection and risk gates."""
import unittest
import numpy as np
import pandas as pd
from TradeAI.replay_sandbox import portfolio_lab_v3 as v


def fake_rows(entries, score=.5, outcome=-.4, risk=1.5):
    rows=[]
    for i in range(entries):
        ts=int(pd.Timestamp('2023-01-03T12:00:00Z').timestamp())+86400*i
        rows.append({'entry_ts':ts,'exit_ts':ts+3600,'stress_exit_ts':ts+3600,
            'year':2023,'month':pd.Timestamp(ts,unit='s',tz='UTC').strftime('%Y-%m'),
            'score_R':score,'votes_with':1,'votes_against':0,
            'symbol':'US500Cash','side_int':1,'holding_m5':24,
            'risk_usd':risk,'stress_risk_usd':risk,'margin_usd':5.,
            'net_usd':outcome,'stress_net_usd':outcome-.25,'stress_valid':True,
            'exit_reason':'STOP'})
    return pd.DataFrame(rows) if rows else fake_rows(1).iloc[0:0].copy()

class PortfolioTests(unittest.TestCase):
    def test_training_excludes_future_outcome_columns(self):
        self.assertNotIn('net_usd',v.FEATURES)
        self.assertNotIn('realized_R',v.FEATURES)
        self.assertNotIn('exit_ts',v.FEATURES)
        self.assertNotIn('win',v.FEATURES)
    def test_years_chronological_and_2025_not_used(self):
        self.assertLess(max(v.TRAIN),v.CALIBRATION)
        self.assertLess(v.CALIBRATION,v.SELECT)
        self.assertLess(v.SELECT,v.CHECK)
        self.assertLess(v.CHECK,v.UNTOUCHED_EVAL)
    def test_no_signal_is_no_trade(self):
        r=v.simulate_portfolio(fake_rows(3,score=-1),0.05,2023)
        self.assertEqual(r['trades'],0)
        self.assertEqual(r['net_usd'],0)
        self.assertFalse(v.passes(r))
    def test_reject_oversized_minimum_lot(self):
        r=v.simulate_portfolio(fake_rows(3,risk=3.5),0.05,2023)
        self.assertEqual(r['trades'],0)
        self.assertGreater(r['rejections'].get('MIN_LOT_RISK',0),0)
    def test_max_drawdown_terminates(self):
        r=v.simulate_portfolio(fake_rows(100, outcome=-1.4),.05,2023)
        self.assertTrue(r['risk_stopped'])
        self.assertLess(r['trades'],100)
        self.assertLess(r['net_usd'],0)
    def test_daily_loss_cap(self):
        d=fake_rows(5,outcome=-3)
        # repeated same day, spaced hourly, so daily cap blocks after 2nd trade
        t=d.entry_ts.iloc[0]
        d['entry_ts']=[t+j*7200 for j in range(5)]
        d['exit_ts']=d.entry_ts+3600
        d['stress_exit_ts']=d.exit_ts
        # 3 USD loss, daily 3% = 4.50 USD, so 2 fills then blocked
        r=v.simulate_portfolio(d,.05,2023)
        self.assertEqual(r['trades'],2)
        self.assertEqual(r['rejections'].get('DAILY_LOSS_CAP'),3)
    def test_monthly_loss_cap(self):
        d=fake_rows(20,outcome=-2.0)
        r=v.simulate_portfolio(d,.05,2023)
        self.assertGreater(r['rejections'].get('MONTHLY_LOSS_CAP',0),0)
    def test_spread_cost_stress_reject(self):
        d=fake_rows(2)
        d['stress_valid']=False
        r=v.simulate_portfolio(d,.05,2023,stressed=True)
        self.assertEqual(r['trades'],0)
        self.assertEqual(r['rejections']['STRESS_COST_TOO_HIGH'],2)
    def test_stress_decreases_net(self):
        d=fake_rows(5, outcome=2.0)
        normal=v.simulate_portfolio(d,.05,2023)
        stress=v.simulate_portfolio(d,.05,2023,stressed=True)
        self.assertLess(stress['net_usd'],normal['net_usd'])
    def test_candidates_same_timestamp_one_position(self):
        d=fake_rows(2, outcome=1.0)
        d.loc[1,'entry_ts']=d.loc[0,'entry_ts']
        r=v.simulate_portfolio(d,.05,2023)
        self.assertEqual(r['trades'],1)
        self.assertEqual(r['rejections']['EXISTING_POSITION'],1)
    def test_no_profit_from_zero_trades(self):
        r=v.simulate_portfolio(fake_rows(0),.05,2023)
        self.assertFalse(v.passes(r))

    def test_stressed_precompute_accepts_readonly_spread_without_mutation(self):
        # With CoW enabled, Pandas exposes a read-only array; without CoW,
        # mutating that view silently corrupts the subsequent research input.
        from TradeAI.replay_sandbox import _broker_replay_core as core
        saved = pd.options.mode.copy_on_write
        try:
            for cow in (False, True):
                with self.subTest(copy_on_write=cow):
                    pd.options.mode.copy_on_write = cow
                    times = pd.date_range('2024-06-03T12:00:00Z', periods=30, freq='5min')
                    bars = pd.DataFrame({
                        'time': times, 'open': 5000.0, 'high': 5000.0,
                        'low': 5000.0, 'close': 5000.0, 'spread': 60.0,
                    })
                    h = pd.DataFrame({'mi': [1], 'atr': [10.0], 'spread_entry': [0.60]})
                    original = h['spread_entry'].to_numpy(copy=True)
                    spec = {'name':'US500Cash', 'point':0.01,
                            'volume_min':0.1, 'trade_contract_size':1.0}
                    ordinary = core.precompute(h, bars, spec, 12, stress=False)
                    stressed = core.precompute(h, bars, spec, 12, stress=True)
                    np.testing.assert_array_equal(original, h['spread_entry'].to_numpy())
                    self.assertTrue(bool(ordinary[1]['valid'][0]))
                    self.assertTrue(bool(stressed[1]['valid'][0]))
                    self.assertLess(float(stressed[1]['net'][0]),
                                    float(ordinary[1]['net'][0]))
        finally:
            pd.options.mode.copy_on_write = saved

if __name__=='__main__':unittest.main()
