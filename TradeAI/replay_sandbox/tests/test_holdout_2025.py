"""Safety regression tests for the isolated retrospective 2025-cutoff study."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from TradeAI.replay_sandbox import holdout_2025 as h


def test_workspace_not_outside_research(tmp_path):
    with pytest.raises(ValueError, match='SUBFOLDER'):
        h._reject_live_workspace(tmp_path)
    with pytest.raises(ValueError, match='SUBFOLDER'):
        h._reject_live_workspace(h.ROOT)
    with pytest.raises(ValueError, match='SUBFOLDER'):
        h._reject_live_workspace(h.ROOT / 'artifacts' / 'research')


def test_raw_time_must_be_explicit_utc():
    assert h._utc_iso_field('2025-12-31 23:55:00+00:00') == '2025-12-31 23:55:00'
    with pytest.raises(ValueError):
        h._utc_iso_field('2025-12-31 23:55:00+03:00')
    with pytest.raises(ValueError):
        h._utc_iso_field('2025-12-31 23:55:00')


def test_split_raw_hard_cutoff_and_2026_holdout(tmp_path, monkeypatch):
    # Use small test-specific minimum counts but exercise the REAL partition code.
    # This never affects production code, which keeps its >=10k/1k guards.
    orig_train, orig_replay, orig_warmup = h.TRAIN_END, h.REPLAY_END, h.WARMUP_START
    monkeypatch.setattr(h, 'TRAIN_END', '2026-01-01')
    monkeypatch.setattr(h, 'REPLAY_END', '2026-10-01')
    monkeypatch.setattr(h, 'WARMUP_START', '2025-12-01')
    # Reach normal 10,000 M5 threshold using repeated bar times that are distinct.
    import pandas as pd
    train_times = pd.date_range('2025-11-15', '2025-12-31 23:55', freq='5min', tz='UTC')
    test_times = pd.date_range('2026-01-01', periods=10001, freq='5min', tz='UTC')
    test_times = test_times.append(pd.DatetimeIndex(['2026-09-30 23:55:00+00:00']))
    bars = train_times.append(test_times)
    src = tmp_path / 'EURUSD_M5.csv.gz'
    with gzip.open(src, 'wt', encoding='utf8') as file:
        file.write('time,symbol,timeframe,open,high,low,close,tick_volume,spread,real_volume\n')
        for t in bars:
            file.write(f'{t.strftime("%Y-%m-%d %H:%M:%S+00:00")},EURUSD,M5,1.1,1.2,1.0,1.1,10,2,0\n')
    with ZipFile(tmp_path / 'out.zip', 'w') as archive:
        result = h.split_raw(src, tmp_path / 'isolated.gz', archive, 'EURUSD', 'M5')
    assert result['holdout_rows'] == 10002
    assert result['last_train'] == '2025-12-31 23:55:00'
    assert result['last_holdout'] == '2026-09-30 23:55:00'
    with gzip.open(tmp_path / 'isolated.gz', 'rt') as file:
        data = file.read()
        assert '2026-01-01' not in data
        assert '2025-12-31' in data
    with ZipFile(tmp_path / 'out.zip') as archive:
        assert archive.testzip() is None
        with gzip.open(archive.open('direction_m5_raw/EURUSD_M5.csv.gz'), 'rt') as file:
            data = file.read()
            assert '2026-09-30' in data
            assert '2025-11-15' not in data


def test_training_requires_prepared_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'ROOT', tmp_path)
    workspace = tmp_path / 'artifacts/research/not-prepared'
    with pytest.raises(FileNotFoundError, match='prepare'):
        h.train(workspace)


def test_replay_requires_frozen_training_record(tmp_path, monkeypatch):
    monkeypatch.setattr(h, 'ROOT', tmp_path)
    workspace = tmp_path / 'artifacts/research/study'
    workspace.mkdir(parents=True)
    (workspace / h.MANIFEST_NAME).write_text(json.dumps({
        'train_exclusive_end': h.TRAIN_END,
        'replay_end_exclusive': h.REPLAY_END,
        'snapshotted_scripts': {}, 'raw_files': {},
    }), encoding='utf8')
    with pytest.raises(FileNotFoundError, match='Run train first'):
        h.replay(workspace, 150, 0.8)


def test_refit_requires_internal_candidate(tmp_path, monkeypatch):
    from TradeAI.replay_sandbox import refit_2025 as r
    monkeypatch.setattr(r, 'model_dir', lambda symbol: tmp_path / 'missing')
    with pytest.raises(FileNotFoundError, match='Missing initial internal-test candidate'):
        r.refit_symbol('EURUSD')

def test_final_refit_cutoff_is_locked():
    from TradeAI.replay_sandbox import refit_2025 as r
    assert r.CUTOFF == h.TRAIN_END == '2026-01-01'
