"""Isolated, retrospective 2025-cutoff Precision M5 training and 2026 replay.

No MT5 APIs, live trading, model promotion, or production artifact writes.
Run from the existing TradeAI_System repository root.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import io
import pandas as pd
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'artifacts' / 'datasets' / 'direction_m5_raw'
DEFAULT_WORKSPACE = ROOT / 'artifacts' / 'research' / 'precision_cutoff_20251231'
TRAIN_END = '2026-01-01'  # first FORBIDDEN date in training data, exclusive
REPLAY_START = '2026-01-01'
REPLAY_END = '2026-10-01'  # excludes October; January through September 2026
WARMUP_START = '2025-12-01'
SYMBOLS = ('EURUSD', 'GBPUSD', 'USDJPY', 'USDCNH')
TIMEFRAMES = ('M5', 'H1')
SNAPSHOT_MODULES = (
    '__init__.py', 'precision_config.py', 'precision_dataset_builder.py',
    'precision_feature_engine.py', 'train_precision_model.py', 'precision_workflow.py',
)
MANIFEST_NAME = 'cutoff_manifest.json'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for part in iter(lambda: file.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def _reject_live_workspace(workspace: Path) -> Path:
    w = workspace.resolve()
    # The workspace MUST reside beneath artifacts/research and cannot be the research dir itself.
    allowed = (ROOT / 'artifacts' / 'research').resolve()
    if w == allowed or allowed not in w.parents:
        raise ValueError(f'Workspace must be a SUBFOLDER of {allowed}; got {w}')
    return w


def _utc_iso_field(value: str) -> str:
    # Source files have GMT/UTC ISO timestamps; refuse unknown broker-clock semantics.
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.utcoffset() is None or d.utcoffset().total_seconds() != 0:
        raise ValueError(f'Raw candle timestamp must be in UTC: {value!r}')
    return d.strftime('%Y-%m-%d %H:%M:%S')


def split_raw(source: Path, train_file: Path, holdout_zip: ZipFile, symbol: str, timeframe: str) -> dict:
    """Chunked, lossless-string raw-history partition with cutoff audit."""
    if not source.is_file():
        raise FileNotFoundError(f'Missing original raw history: {source}')
    train_file.parent.mkdir(parents=True, exist_ok=True)
    counts = {'train_rows': 0, 'replay_rows': 0, 'holdout_rows': 0}
    previous, first_holdout, last_holdout, last_train = None, None, None, None
    arcname = f'direction_m5_raw/{symbol}_{timeframe}.csv.gz'
    with gzip.open(train_file, 'wt', encoding='utf-8', newline='') as train, \
         holdout_zip.open(arcname, 'w', force_zip64=True) as raw_zip, \
         gzip.GzipFile(fileobj=raw_zip, mode='wb', mtime=0) as compressed, \
         io.TextIOWrapper(compressed, encoding='utf-8', newline='') as replay:
        required = {'time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'tick_volume', 'spread'}
        train_header, replay_header = True, True
        # dtype=str preserves OHLC decimal strings exactly; buffers avoid millions
        # of individual zlib calls that make row-at-a-time gzip slow.
        for chunk in pd.read_csv(source, compression='gzip', chunksize=50000,
                                 dtype=str, keep_default_na=False):
            if not required.issubset(chunk.columns):
                raise ValueError(f'{source.name}: missing required fields')
            times = chunk['time'].str.slice(0, 19)
            # Decline broker-clock data or any mixed offset. No implicit timezone fix.
            is_utc = chunk['time'].str.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?\+00:00')
            if not bool(is_utc.all()):
                raise ValueError(f'{source.name}: raw times must be explicit UTC (+00:00)')
            if not bool(times.is_monotonic_increasing) or bool(times.duplicated().any()):
                raise ValueError(f'{source.name}: non-increasing candle time inside chunk')
            if previous is not None and str(times.iloc[0]) <= previous:
                raise ValueError(f'{source.name}: out-of-order or overlapping chunks')
            previous = str(times.iloc[-1])
            if not bool(chunk['symbol'].eq(symbol).all()) or not bool(chunk['timeframe'].eq(timeframe).all()):
                raise ValueError(f'{source.name}: wrong symbol/timeframe')
            train_part = chunk.loc[times < TRAIN_END]
            replay_part = chunk.loc[(times >= WARMUP_START) & (times < REPLAY_END)]
            test_times = times.loc[(times >= REPLAY_START) & (times < REPLAY_END)]
            if not train_part.empty:
                train_part.to_csv(train, index=False, header=train_header, lineterminator='\n')
                train_header = False
                counts['train_rows'] += len(train_part)
                last_train = str(times.loc[train_part.index[-1]])
            if not replay_part.empty:
                replay_part.to_csv(replay, index=False, header=replay_header, lineterminator='\n')
                replay_header = False
                counts['replay_rows'] += len(replay_part)
            if not test_times.empty:
                counts['holdout_rows'] += len(test_times)
                first_holdout = first_holdout or str(test_times.iloc[0])
                last_holdout = str(test_times.iloc[-1])
        if train_header or replay_header:
            raise RuntimeError(f'{source}: empty partition')
    minimum = 10000 if timeframe == 'M5' else 1000
    if counts['train_rows'] < minimum or counts['holdout_rows'] < minimum:
        raise ValueError(f'{symbol} {timeframe}: insufficient pre-2026 or 2026 history: {counts}')
    if last_train is None or last_train[:10] < '2025-12-15':
        raise ValueError(f'{symbol} {timeframe}: train data does not reach December 2025')
    if last_holdout is None or last_holdout[:10] < '2026-09-30':
        raise ValueError(f'{symbol} {timeframe}: missing complete holdout through Sep 30')
    counts.update(first_holdout=first_holdout, last_holdout=last_holdout,
                  last_train=last_train, source_sha256=sha256(source),
                  train_sha256=sha256(train_file))
    return counts


def _manifest(workspace: Path) -> dict:
    p = workspace / MANIFEST_NAME
    if not p.is_file():
        raise FileNotFoundError(f'Run prepare first: {p}')
    d = json.loads(p.read_text(encoding='utf8'))
    if d['train_exclusive_end'] != TRAIN_END or d['replay_end_exclusive'] != REPLAY_END:
        raise ValueError('Cutoff manifest does not match this locked study')
    for rel, original_hash in d['snapshotted_scripts'].items():
        if sha256(workspace / rel) != original_hash:
            raise RuntimeError(f'Snapshotted training code changed since preparation: {rel}')
    for rel, info in d['raw_files'].items():
        t = workspace / 'artifacts' / 'datasets' / 'direction_m5_raw' / rel
        if not t.is_file() or sha256(t) != info['train_sha256']:
            raise RuntimeError(f'Isolated training raw file missing or changed: {t}')
    return d


def prepare(workspace: Path, raw_root: Path, broker_profile: Path) -> dict:
    workspace = _reject_live_workspace(workspace)
    if workspace.exists():
        raise FileExistsError(f'Preserving existing study: {workspace}. Choose a NEW --workspace')
    if not broker_profile.is_file():
        raise FileNotFoundError(f'Missing broker profile: {broker_profile}')
    profile_bytes = broker_profile.read_bytes()
    profile = json.loads(profile_bytes)
    if profile.get('account_currency') != 'USD' or not set(SYMBOLS).issubset(profile.get('symbols', {})):
        raise ValueError('Broker profile must have USD account and all four symbols')
    # Build under a temporary sibling and only publish when the full audit passed.
    staging = workspace.with_name(workspace.name + '.building')
    if staging.exists():
        raise FileExistsError(f'Incomplete previous staging exists: {staging}; review before removing it')
    staging.mkdir(parents=True)
    try:
        snapshot = {}
        for name in SNAPSHOT_MODULES:
            src = ROOT / 'Trade_Bot_Training' / name
            dst = staging / 'Trade_Bot_Training' / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            snapshot[str(dst.relative_to(staging)).replace('\\', '/')] = sha256(dst)
        refit_src = ROOT / 'TradeAI' / 'replay_sandbox' / 'refit_2025.py'
        refit_dest = staging / 'Trade_Bot_Training' / 'refit_precision_2025.py'
        shutil.copy2(refit_src, refit_dest)
        snapshot[str(refit_dest.relative_to(staging)).replace('\\', '/')] = sha256(refit_dest)
        raw_stats = {}
        archive_path = staging / 'replay_2026_jan_sep.zip'
        with ZipFile(archive_path, 'w', compression=ZIP_STORED, allowZip64=True) as replay_zip:
            replay_zip.writestr('broker_profile.json', profile_bytes)
            for sym in SYMBOLS:
                for tf in TIMEFRAMES:
                    fname = f'{sym}_{tf}.csv.gz'
                    raw_stats[fname] = split_raw(raw_root / fname,
                                staging / 'artifacts' / 'datasets' / 'direction_m5_raw' / fname,
                                replay_zip, sym, tf)
        info = {
            'study': 'RETROSPECTIVE_PRECISION_M5_2025_CUTOFF_V1',
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'train_exclusive_end': TRAIN_END,
            'train_inclusive_end': '2025-12-31',
            'replay_start_inclusive': REPLAY_START,
            'replay_end_exclusive': REPLAY_END,
            'replay_warmup_from': WARMUP_START,
            'source_raw_dir': str(raw_root.resolve()),
            'source_broker_profile': str(broker_profile.resolve()),
            'broker_profile_sha256': sha256(broker_profile),
            'snapshotted_scripts': snapshot,
            'raw_files': raw_stats,
            'replay_archive_sha256': sha256(archive_path),
            'label_boundary': 'All labels generated inside isolated pre-2026 raw history, including 6 future M5 bars',
            'classification': 'RETROSPECTIVE_2026_HOLDOUT_NOT_REALTIME_FORWARD',
        }
        (staging / MANIFEST_NAME).write_text(json.dumps(info, indent=2)+'\n', encoding='utf8')
        staging.rename(workspace)
        return info
    except BaseException:
        # Do not delete a partially written study automatically; it is safe to inspect.
        raise


def train(workspace: Path) -> dict:
    workspace = _reject_live_workspace(workspace)
    info = _manifest(workspace)
    if (workspace / 'training_complete.json').exists():
        raise FileExistsError('This study is already trained. Never overwrite a frozen candidate.')
    if (workspace / 'artifacts' / 'models' / 'precision_m5').exists():
        raise FileExistsError('Partial or existing model artifacts present. Inspect before retry.')
    environment = os.environ.copy()
    # Start in workspace with an isolated module path; do not import the root project's code.
    environment['PYTHONPATH'] = str(workspace)
    command = [sys.executable, '-m', 'Trade_Bot_Training.precision_workflow', 'all']
    print('Isolated training process:', ' '.join(command), flush=True)
    print('CWD:', workspace, flush=True)
    subprocess.run(command, cwd=workspace, env=environment, check=True)
    refit_command = [sys.executable, '-m', 'Trade_Bot_Training.refit_precision_2025']
    print('Final refit through Dec 2025:', ' '.join(refit_command), flush=True)
    subprocess.run(refit_command, cwd=workspace, env=environment, check=True)
    records = {}
    for sym in SYMBOLS:
        base = workspace / 'artifacts' / 'models' / 'precision_m5' / sym
        meta = json.loads((base / 'metadata.json').read_text(encoding='utf8'))
        if meta['split']['test_end'][:10] > '2025-12-31':
            raise RuntimeError(f'Training violated cutoff: {sym} {meta["split"]["test_end"]}')
        if not meta.get('final_refit_on_all_pre2026'):
            raise RuntimeError(f'{sym} was not refitted on full pre-2026 history')
        records[sym] = {
            'final_refit_train_rows': meta['final_refit_train_rows'],
            'final_refit_last_decision_time': meta['final_refit_last_decision_time'],
            'model_gate_pre_refit_candidate_only': True,
            'model_sha256': sha256(base / 'model.json'),
            'metadata_sha256': sha256(base / 'metadata.json'),
            'pre2026_test_end': meta['split']['test_end'],
            'production_ready_by_historical_gate': meta['production_ready_by_historical_gate'],
            'historical_gate_failures': meta['production_gate_failures'],
        }
    result = {
        'trained_at_utc': datetime.now(timezone.utc).isoformat(),
        'classification': 'RETROSPECTIVE_2026_HOLDOUT_NOT_REALTIME_FORWARD',
        'train_exclusive_end': info['train_exclusive_end'],
        'symbols': records,
        'training_script_hashes': info['snapshotted_scripts'],
    }
    (workspace / 'training_complete.json').write_text(json.dumps(result, indent=2)+'\n',encoding='utf8')
    return result


def replay(workspace: Path, balance: float, spread_floor_pips: float) -> dict:
    workspace = _reject_live_workspace(workspace)
    info = _manifest(workspace)
    complete_file = workspace / 'training_complete.json'
    if not complete_file.exists():
        raise FileNotFoundError('Run train first; cannot replay without frozen candidate hashes')
    complete = json.loads(complete_file.read_text(encoding='utf8'))
    for sym in SYMBOLS:
        base = workspace / 'artifacts' / 'models' / 'precision_m5' / sym
        if sha256(base / 'model.json') != complete['symbols'][sym]['model_sha256']:
            raise ValueError(f'Candidate model changed after training: {sym}')
        if sha256(base / 'metadata.json') != complete['symbols'][sym]['metadata_sha256']:
            raise ValueError(f'Candidate metadata changed after training: {sym}')
    archive = workspace / 'replay_2026_jan_sep.zip'
    if sha256(archive) != info['replay_archive_sha256']:
        raise ValueError('2026 holdout archive changed after preparation')
    output = workspace / 'replay_results_2026'
    if output.exists():
        raise FileExistsError(f'Existing replay results retained: {output}. Use a NEW study to rerun.')
    # The prior built-in replay can be used unchanged. This is an EXPLORATORY flag
    # because the candidate was trained today, after the 2026 candles occurred;
    # it cannot truthfully claim to be a 2026 real-time forward test.
    from .precision_replay import run_precision_replay
    from .data import parse_time
    result = run_precision_replay(
        data_zip=archive, start=parse_time(REPLAY_START+'T00:00:00+00:00'),
        end=parse_time(REPLAY_END+'T00:00:00+00:00'), model_root=workspace / 'artifacts' / 'models' / 'precision_m5',
        output=output, balance=balance, spread_floor_pips=spread_floor_pips,
        exploratory=True,
    )
    result['evaluation_label'] = 'RETROSPECTIVE_2026_OUT_OF_TRAINING_HOLDOUT_NOT_REALTIME_FORWARD'
    result['train_data_exclusive_end'] = TRAIN_END
    result['test_data_start'] = REPLAY_START
    result['test_data_end_exclusive'] = REPLAY_END
    result['candidate_training_utc'] = complete['trained_at_utc']
    result['research_caveat'] = (
        'Models trained now using only candles before 2026. The 2026 replay is a temporal '
        'out-of-training-sample test, NOT an actual 2026 live forward test; do not tune after viewing the holdout.'
    )
    (output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    return result


def main():
    parser = argparse.ArgumentParser(description='Train Precision M5 with 2025-12-31 cutoff and retrospectively replay 2026; NO MT5 orders.')
    parser.add_argument('command', choices=('prepare', 'train', 'replay'))
    parser.add_argument('--workspace', type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument('--raw-root', type=Path, default=RAW)
    parser.add_argument('--broker-profile', type=Path, default=ROOT / 'artifacts' / 'reports' / 'broker_profile.json')
    parser.add_argument('--balance', type=float, default=150.)
    parser.add_argument('--spread-floor-pips', type=float, default=0.8)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.workspace, args.raw_root, args.broker_profile)
        print('PREPARED. Train excludes ALL 2026 price rows; 2026 reserve intact.')
        print(json.dumps({k:v for k,v in result.items() if k!='raw_files' and k!='snapshotted_scripts'}, indent=2))
        print('Workspace:', args.workspace.resolve())
    elif args.command == 'train':
        result = train(args.workspace)
        print(json.dumps(result, indent=2))
    else:
        result = replay(args.workspace, args.balance, args.spread_floor_pips)
        print(json.dumps(result, indent=2))
        print('Reports:', (args.workspace / 'replay_results_2026').resolve())


if __name__ == '__main__':
    main()
