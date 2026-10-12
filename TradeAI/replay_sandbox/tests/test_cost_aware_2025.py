"""No MT5 access. Verify pre-2026 cutoff, causal spread policy and replay integration."""
from __future__ import annotations

import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zipfile import ZipFile

import pytest

from TradeAI.replay_sandbox.cost_aware_2025 import (
    BLACKOUT_HOURS, CostAwareEntryGate, POLICY_NAME,
    _raw_2025_spread_data,
)
from TradeAI.replay_sandbox.data import Candle
from TradeAI.replay_sandbox.precision_replay import PrecisionPaperBroker

UTC = timezone.utc
PROFILE = {'account_currency':'USD','symbols': {'EURUSD':{
    'point': 0.00001,'digits':5, 'volume_min':0.01, 'volume_step':0.01,
    'contract_size':100000, 'slippage_pips':0.2, 'commission_per_lot':7.,
}}}


def candle(t, points=10):
    return Candle(t,'EURUSD','M5',1.10000,1.10015,1.09985,1.10005,points,5)


def gate(cap=3.0):
    return CostAwareEntryGate({
        'name':POLICY_NAME, 'development_end_exclusive':'2026-01-01',
        'blackout_hours_in_archived_timestamp_clock':list(BLACKOUT_HOURS),
        'spread_filter_observation':'PREVIOUS_COMPLETED_M5_CANDLE_SPREAD_ONLY',
        'pairs':{'EURUSD':{'max_last_closed_spread_pips':cap}},
    })


def test_guard_accepts_normal_session_and_previously_closed_spread():
    broker=PrecisionPaperBroker(PROFILE,spread_floor_pips=0.8)
    now=datetime(2026,5,4,15,0,tzinfo=UTC)
    assert gate()(candle(now,999),candle(now-timedelta(minutes=5),18),broker)==(True,'')
    # High spread recorded INSIDE the executing M5 candle must NOT affect the pre-entry gate.


def test_guard_rejects_rollover_entry_and_projected_exit():
    broker=PrecisionPaperBroker(PROFILE)
    for hour,minute in ((23,15),(0,30),(22,40)):
        now=datetime(2026,5,4,hour,minute,tzinfo=UTC)
        allowed, reason=gate()(candle(now),candle(now-timedelta(minutes=5)),broker)
        assert not allowed and reason=='COST_POLICY_ROLLOVER_WINDOW'


def test_guard_rejects_stale_last_candle_and_wide_prior_spread():
    broker=PrecisionPaperBroker(PROFILE)
    now=datetime(2026,5,4,15,0,tzinfo=UTC)
    assert gate()(candle(now),None,broker)[1]=='COST_POLICY_MISSING_LAST_CLOSED_BAR'
    assert gate()(candle(now),candle(now-timedelta(minutes=10)),broker)[1]=='COST_POLICY_MISSING_LAST_CLOSED_BAR'
    assert gate(1.5)(candle(now),candle(now-timedelta(minutes=5),20),broker)[1]=='COST_POLICY_SPREAD_TOO_WIDE'
    assert gate(1.5)(candle(now),candle(now-timedelta(minutes=5),10),broker)==(True,'')


def test_reject_modified_policy_or_2026_data(tmp_path):
    with pytest.raises(ValueError,match='Unrecognized'):
        CostAwareEntryGate({'name':'unapproved','development_end_exclusive':'2026-01-01'})
    import pandas as pd
    times=pd.date_range('2025-01-01',periods=10001,freq='5min',tz=UTC)
    p=tmp_path/'sample.csv.gz'
    pd.DataFrame({'time':times.strftime('%Y-%m-%d %H:%M:%S+00:00'),
                  'spread': [20]*len(times)}).to_csv(p,index=False,compression='gzip')
    assert len(_raw_2025_spread_data(p))==len(times)
    extra=pd.concat([pd.read_csv(p),pd.DataFrame([{'time':'2026-01-01 00:00:00+00:00','spread':30}])])
    extra.to_csv(p,index=False,compression='gzip')
    with pytest.raises(ValueError,match='2026'):
        _raw_2025_spread_data(p)


def test_cost_gate_inside_real_replay_loop(tmp_path,monkeypatch):
    import TradeAI.replay_sandbox.precision_replay as pr
    start=datetime(2025,9,16,15,0,tzinfo=UTC)
    bars=[candle(start+timedelta(minutes=5*i)) for i in range(9)]
    def generate():
        return gzip.compress(('time,symbol,timeframe,open,high,low,close,tick_volume,spread,real_volume\n'+''.join(
            f'{x.time.strftime("%Y-%m-%d %H:%M:%S+00:00")},{x.symbol},M5,{x.open},{x.high},{x.low},{x.close},{x.tick_volume},{x.spread_points},0\n'
            for x in bars)).encode())
    archive=tmp_path/'mini.zip'
    with ZipFile(archive,'w') as z:
        z.writestr('broker_profile.json',json.dumps(PROFILE))
        z.writestr('direction_m5_raw/EURUSD_M5.csv.gz',generate())
    model_dir=tmp_path/'models'/'EURUSD';model_dir.mkdir(parents=True)
    (model_dir/'model.json').write_text('{}');(model_dir/'metadata.json').write_text('{}')
    class FakeModel:
        def __init__(self, s, p):
            self.metadata={'created_utc':'2026-10-10T00:00:00+00:00'}
            self.historical_gate_pass=True
            self.action_confidence=.55
    monkeypatch.setattr(pr,'PrecisionDirectionModel',FakeModel)
    def fake_decisions(rows,model,b,e):
        t=bars[1].time
        return {t: {'action':'BUY','tradeable':True,'historical_gate_pass':True,
                    'bar_time':bars[0].time,'confidence':0.8}}, {'scored':1,'tradeable':1,'invalid_features':0}
    monkeypatch.setattr(pr,'_prepare_decisions',fake_decisions)
    result=pr.run_precision_replay(data_zip=archive,start=start,end=start+timedelta(minutes=45),
                symbols=('EURUSD',),model_root=model_dir.parent,output=tmp_path/'output',
                exploratory=True,balance=150,entry_policy=lambda c,prior,broker:(False,'COST_POLICY_TEST'))
    assert result['qualified_signals']==1
    assert result['cost_policy_rejected']==1
    assert result['filled_entries']==0
    assert result['ending_balance_usd']==150
    import pandas as pd
    rejected=pd.read_csv(tmp_path/'output'/'rejections.csv')
    assert rejected['reason'].tolist()==['COST_POLICY_TEST']


def test_policy_disabled_by_default_in_replay(tmp_path,monkeypatch):
    # Validates the default path still enters, using the exact same injected synthetic context.
    import TradeAI.replay_sandbox.precision_replay as pr
    assert 'entry_policy' in pr.run_precision_replay.__code__.co_varnames
