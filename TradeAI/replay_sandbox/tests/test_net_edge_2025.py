"""Research safeguards and cost-label sanity checks. No market-data/MT5 dependency."""
import numpy as np
import pandas as pd
from TradeAI.replay_sandbox.net_edge_2025 import net_cost_labels, split_rows, select_nonoverlap


def candles(first='2025-03-03 10:00:00+00:00', n=13, spread=15):
    t = pd.date_range(first, periods=n, freq='5min')
    return pd.DataFrame({'time':t,'open':1.10000 + np.arange(n)*0.00010,
                         'close':1.10005 + np.arange(n)*0.00010,'spread':np.full(n,spread)})


def spec():
    return dict(symbol='EURUSD',point=0.00001,digits=5,volume_min=0.01,
                contract_size=100000,commission_per_lot=7,slippage_pips=0.2)


def test_buy_net_accounting_and_no_future_as_observable():
    raw=candles()
    labels=net_cost_labels(raw,spec(),max_spread_pips=2)
    # +5.5 pip from next entry open to sixth future exit close; 1.5 spread, .4 slip, .7 commission.
    assert labels.iloc[0]['net_buy_pips']==pytest_approx(5.5-1.5-0.4-0.7)
    assert labels.iloc[0]['target_class']==2
    # Changing a FUTURE spread affects the label, never the currently completed spread feature.
    changed=raw.copy(); changed.loc[1,'spread']=200
    other=net_cost_labels(changed,spec(),max_spread_pips=2)
    assert labels.iloc[0]['observed_spread_pips']==other.iloc[0]['observed_spread_pips']
    assert other.iloc[0]['net_buy_pips'] < labels.iloc[0]['net_buy_pips']
    assert other.iloc[0]['label_valid']


def pytest_approx(x):
    import pytest
    return pytest.approx(x,abs=1e-5)


def test_spread_and_rollover_block_entries():
    x=net_cost_labels(candles(spread=40),spec(),max_spread_pips=2)
    assert not x['label_valid'].any()
    night=net_cost_labels(candles(first='2025-03-03 22:50:00+00:00'),spec(),2)
    assert not night.iloc[0]['label_valid']  # projected exit hits 23h


def test_gap_cannot_generate_a_30m_label():
    x=candles()
    x.loc[2:,'time'] += pd.Timedelta(minutes=5)
    labels=net_cost_labels(x,spec(),2)
    assert not labels.iloc[0]['label_valid']


def test_split_disallows_train_future_horizon_into_validation():
    times=pd.to_datetime(['2025-07-31 23:55+00:00','2025-08-01 01:00+00:00',
                          '2025-09-30 23:55+00:00','2025-10-01 01:00+00:00',
                          '2025-12-29 01:00+00:00','2025-05-05 01:00+00:00'])
    times=pd.Series(times)
    exits=times+pd.Timedelta(minutes=30)
    df=pd.DataFrame({'decision_time':times,'exit_time':exits})
    a=split_rows_min(df)
    assert len(a['train'])==1
    assert len(a['validation'])==1
    assert len(a['test'])==2


def split_rows_min(df):
    # Same split predicate, without minimum sample-count check used for real history.
    from TradeAI.replay_sandbox import net_edge_2025 as ne
    t=df['decision_time']; e=df['exit_time']
    return {
     'train':df.loc[(t<pd.Timestamp(ne.TRAIN_END,tz='UTC'))&(e<=pd.Timestamp(ne.TRAIN_END,tz='UTC'))],
     'validation':df.loc[(t>=pd.Timestamp(ne.TRAIN_END,tz='UTC'))&(t<pd.Timestamp(ne.VALIDATION_END,tz='UTC'))&(e<=pd.Timestamp(ne.VALIDATION_END,tz='UTC'))],
     'test':df.loc[(t>=pd.Timestamp(ne.VALIDATION_END,tz='UTC'))&(t<pd.Timestamp(ne.TEST_END,tz='UTC'))&(e<pd.Timestamp(ne.TEST_END,tz='UTC'))]
    }


def test_no_overlap():
    t=pd.date_range('2025-11-03 13:00+00:00',periods=5,freq='5min')
    d=pd.DataFrame({'entry_time':t, 'exit_time':t+pd.Timedelta(minutes=30),
                    'predicted_class':[2,0,1,2,2], 'confidence':[.60,.90,.99,.99,.99]})
    chosen=select_nonoverlap(d)
    assert len(chosen)==1


def test_validation_gate_rejects_negative_expectancy():
    from TradeAI.replay_sandbox.net_edge_2025 import validation_threshold
    t=pd.date_range('2025-08-05 10:00+00:00',periods=100,freq='60min')
    d=pd.DataFrame({'decision_time':t,'entry_time':t,'exit_time':t+pd.Timedelta(minutes=30),
                    'net_buy_pips':[-3.0]*100, 'net_sell_pips':[-2.0]*100})
    probs=np.tile([0.1,0.1,0.8],(len(d),1))
    threshold,trials=validation_threshold(d,probs)
    assert threshold is None
    assert not any(t['passed'] for t in trials)


def test_validation_gate_selects_positive_net():
    from TradeAI.replay_sandbox.net_edge_2025 import validation_threshold
    t=pd.date_range('2025-08-05 10:00+00:00',periods=100,freq='60min')
    d=pd.DataFrame({'decision_time':t,'entry_time':t,'exit_time':t+pd.Timedelta(minutes=30),
                    'net_buy_pips':[2.0]*100, 'net_sell_pips':[-3.0]*100})
    probs=np.tile([0.1,0.1,0.8],(len(d),1))
    threshold,trials=validation_threshold(d,probs)
    assert threshold == 0.4
    assert all(t['passed'] for t in trials)
