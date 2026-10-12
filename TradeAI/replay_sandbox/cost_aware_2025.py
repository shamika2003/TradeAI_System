"""2025-only spread/session risk controls for frozen Precision M5 research replays.

Command line:
  python -m TradeAI.replay_sandbox.cost_aware_2025 prepare
  python -m TradeAI.replay_sandbox.cost_aware_2025 replay --balance 150 --spread-floor-pips 0.8

READ-ONLY to original market history, models, demo/live engine, NIRA, and MT5.
No future 2026 outcomes are used to set policy parameters. The 2026 study has
already been inspected, so ALL subsequent 2026 outcomes are EXPLORATORY.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path

import pandas as pd

from . import holdout_2025 as h
from .data import Candle, MT5Archive, parse_time
from .precision_replay import PrecisionPaperBroker, run_precision_replay

UTC = timezone.utc
POLICY_NAME = 'COST_AWARE_2025_SPREAD_SESSION_V1'
BLACKOUT_HOURS = (23, 0)  # Hours in original dataset timestamp clock, NOT user or broker-local clock.
FIXED_ABSOLUTE_SPREAD_CAP_PIPS = 10.0  # Risk budget chosen independently of 2026 outcomes.
NORMAL_HOUR_SPREAD_QUANTILE = 0.95
POLICY_FILE = 'cost_policy_2025.json'
OUTPUT_DIR = 'replay_results_2026_cost_aware'


def _study(workspace: Path) -> tuple[Path, dict, dict, Path]:
    workspace = h._reject_live_workspace(workspace)
    manifest = h._manifest(workspace)
    if manifest['train_exclusive_end'] != '2026-01-01' or manifest['replay_start_inclusive'] != '2026-01-01' or manifest['replay_end_exclusive'] != '2026-10-01':
        raise ValueError('Study is not the frozen 2025 cutoff / 2026 reserve.')
    train_path = workspace / 'training_complete.json'
    if not train_path.is_file():
        raise FileNotFoundError('Frozen research models not found: run holdout_2025 train first.')
    trained = json.loads(train_path.read_text(encoding='utf-8'))
    if trained['train_exclusive_end'] != '2026-01-01':
        raise ValueError('Training cutoff mismatch.')
    for sym in h.SYMBOLS:
        model = workspace / 'artifacts' / 'models' / 'precision_m5' / sym
        for file, field in [('model.json', 'model_sha256'), ('metadata.json', 'metadata_sha256')]:
            if h.sha256(model / file) != trained['symbols'][sym][field]:
                raise ValueError(f'Frozen model drift: {sym} {file}')
    archive = workspace / 'replay_2026_jan_sep.zip'
    if h.sha256(archive) != manifest['replay_archive_sha256']:
        raise ValueError('2026 reserve changed; cannot compare results.')
    return workspace, manifest, trained, archive


def _raw_2025_spread_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, compression='gzip', usecols=['time','spread'], dtype={'spread':'float64'})
    ts = pd.to_datetime(df['time'], utc=True, errors='raise')
    if ts.isna().any() or not ts.is_monotonic_increasing or ts.duplicated().any():
        raise ValueError(f'Corrupt time order: {path}')
    if len(df) < 10000 or (ts >= pd.Timestamp('2026-01-01', tz='UTC')).any():
        raise ValueError(f'2025 calibration source includes 2026 or is too short: {path}')
    selected = (ts >= pd.Timestamp('2025-01-01', tz='UTC')) & (ts < pd.Timestamp('2026-01-01',tz='UTC'))
    df = df.loc[selected].copy()
    if len(df) < 1000:
        raise ValueError(f'Insufficient 2025 data: {path}')
    df['hour'] = ts.loc[selected].dt.hour.to_numpy()
    if not (df['spread'] >= 0).all() or not df['spread'].notna().all():
        raise ValueError('Invalid 2025 spreads')
    return df


def make_policy(workspace: Path) -> dict:
    workspace, manifest, trained, archive_path = _study(workspace)
    policy_path = workspace / POLICY_FILE
    if policy_path.exists():
        raise FileExistsError(f'Policy already frozen; refusing overwrite: {policy_path}')
    profile = MT5Archive(archive_path).broker_profile
    policy = {
        'name': POLICY_NAME,
        'evaluation_classification': '2026_REPLAY_EXPLORATORY_HOLDOUT_ALREADY_INSPECTED',
        'development_start_inclusive': '2025-01-01',
        'development_end_exclusive': '2026-01-01',
        'blackout_hours_in_archived_timestamp_clock': list(BLACKOUT_HOURS),
        'blackout_entry_and_projected_30m_exit': True,
        'history_source': 'isolated pre-2026 raw M5 candles only',
        'spread_filter_observation': 'PREVIOUS_COMPLETED_M5_CANDLE_SPREAD_ONLY',
        'spread_quantile': NORMAL_HOUR_SPREAD_QUANTILE,
        'absolute_risk_budget_pips': FIXED_ABSOLUTE_SPREAD_CAP_PIPS,
        'model_hashes': {s: trained['symbols'][s]['model_sha256'] for s in h.SYMBOLS},
        'archive_sha256': manifest['replay_archive_sha256'],
        'pairs': {},
    }
    for sym in h.SYMBOLS:
        raw = workspace / 'artifacts' / 'datasets' / 'direction_m5_raw' / f'{sym}_M5.csv.gz'
        d = _raw_2025_spread_data(raw)
        normal = d.loc[~d['hour'].isin(BLACKOUT_HOURS), 'spread'].to_numpy()
        if len(normal) < 1000:
            raise ValueError(f'Too little nonblackout calibration data: {sym}')
        spec = profile['symbols'][sym]
        points_per_pip = 10 if int(spec['digits']) in (3,5) else 1
        normal_pips = normal / points_per_pip
        p95 = float(pd.Series(normal_pips).quantile(NORMAL_HOUR_SPREAD_QUANTILE))
        median = float(pd.Series(normal_pips).median())
        cap = min(p95, FIXED_ABSOLUTE_SPREAD_CAP_PIPS)
        if not all(isfinite(x) and x>=0 for x in (p95, median, cap)):
            raise ValueError(f'Invalid calibration: {sym}')
        policy['pairs'][sym] = {
            'nonblackout_2025_samples': len(normal),
            'median_nonblackout_spread_pips': round(median,6),
            'p95_nonblackout_spread_pips': round(p95,6),
            'max_last_closed_spread_pips': round(cap,6),
            'note': 'Risk ceiling, not a forecast of profit or fillable broker quote',
        }
    # If saved, forever immutable for this study. The 2026 replay must not tune it.
    policy_path.write_text(json.dumps(policy, indent=2)+'\n', encoding='utf-8', newline='\n')
    return policy


@dataclass(frozen=True)
class CostAwareEntryGate:
    policy: dict

    def __post_init__(self):
        if self.policy.get('name') != POLICY_NAME or self.policy.get('development_end_exclusive') != '2026-01-01':
            raise ValueError('Unrecognized/unlocked policy')
        if tuple(self.policy.get('blackout_hours_in_archived_timestamp_clock', [])) != BLACKOUT_HOURS:
            raise ValueError('Policy time guard changed unexpectedly')
        if self.policy.get('spread_filter_observation') != 'PREVIOUS_COMPLETED_M5_CANDLE_SPREAD_ONLY':
            raise ValueError('Forward-looking spread policy is forbidden')

    def __call__(self, candle: Candle, last_completed: Candle | None,
                 broker: PrecisionPaperBroker) -> tuple[bool, str]:
        if candle.time.tzinfo is None:
            return False, 'COST_POLICY_NAIVE_TIME'
        projected_exit = candle.time + timedelta(minutes=30)
        if candle.time.hour in BLACKOUT_HOURS or projected_exit.hour in BLACKOUT_HOURS:
            return False, 'COST_POLICY_ROLLOVER_WINDOW'
        # Reject stale snapshots after gaps or around DST jumps.
        if last_completed is None or last_completed.symbol != candle.symbol or last_completed.ends_at != candle.time:
            return False, 'COST_POLICY_MISSING_LAST_CLOSED_BAR'
        if candle.symbol not in self.policy['pairs']:
            return False, 'COST_POLICY_UNKNOWN_SYMBOL'
        last_spread_pips = (last_completed.spread_points * broker.point(candle.symbol)
                            * broker.spread_multiplier / broker.pip(candle.symbol))
        last_spread_pips = max(last_spread_pips, broker.spread_floor_pips)
        if not isfinite(last_spread_pips):
            return False, 'COST_POLICY_INVALID_SPREAD'
        allowed = float(self.policy['pairs'][candle.symbol]['max_last_closed_spread_pips'])
        if last_spread_pips > allowed + 1e-8:
            return False, 'COST_POLICY_SPREAD_TOO_WIDE'
        return True, ''


def run_policy_replay(workspace: Path, balance: float, spread_floor_pips: float) -> dict:
    workspace, manifest, trained, archive = _study(workspace)
    policy_path = workspace / POLICY_FILE
    if not policy_path.is_file():
        raise FileNotFoundError('First run: python -m TradeAI.replay_sandbox.cost_aware_2025 prepare')
    policy = json.loads(policy_path.read_text(encoding='utf-8'))
    gate = CostAwareEntryGate(policy)
    if policy['archive_sha256'] != manifest['replay_archive_sha256']:
        raise ValueError('Policy archive mismatch')
    for s in h.SYMBOLS:
        if policy['model_hashes'][s] != trained['symbols'][s]['model_sha256']:
            raise ValueError(f'Policy model drift: {s}')
    if balance <= 0 or spread_floor_pips < 0:
        raise ValueError('Invalid balance or spread floor')
    output = workspace / OUTPUT_DIR
    if output.exists():
        raise FileExistsError(f'Existing study replay results retained: {output}')
    result = run_precision_replay(
        data_zip=archive,
        start=parse_time('2026-01-01T00:00:00+00:00'),
        end=parse_time('2026-10-01T00:00:00+00:00'),
        model_root=workspace / 'artifacts' / 'models' / 'precision_m5',
        output=output,
        balance=balance,
        spread_floor_pips=spread_floor_pips,
        exploratory=True,
        entry_policy=gate,
    )
    result['strategy'] = 'FROZEN_PRECISION_M5_PLUS_2025_COST_RISK_FILTERS'
    result['evaluation_label'] = 'EXPLORATORY_2026_REPLAY_AFTER_BASELINE_EXAMINED_NOT_INDEPENDENT_TEST'
    result['cost_policy'] = policy
    result['research_caveat'] = (
        'No 2026 data used in constructing the fixed cost policy. However 2026 baseline results '
        'were already examined before proposing this policy, so this rerun is NOT a fresh blind holdout. '
        'Previous completed M5 bar spread is used for gating; candle-period bid/ask assumptions '
        'remain approximate and actual executable opening spreads are not verified ticks.'
    )
    (output / 'summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description='2025-only spread/session policy; research replay ONLY; NO MT5.')
    parser.add_argument('command', choices=('prepare', 'replay'))
    parser.add_argument('--workspace', type=Path, default=h.DEFAULT_WORKSPACE)
    parser.add_argument('--balance', type=float, default=150.)
    parser.add_argument('--spread-floor-pips', type=float, default=0.8)
    a = parser.parse_args()
    if a.command == 'prepare':
        result = make_policy(a.workspace)
        print('POLICY FROZEN: derived from pre-2026 data only; no model changed.')
        print(json.dumps(result, indent=2))
        print('Policy:', (a.workspace / POLICY_FILE).resolve())
    else:
        result = run_policy_replay(a.workspace, a.balance, a.spread_floor_pips)
        overview = {k:result[k] for k in ('status','evaluation_label','candles_replayed','qualified_signals',
                                         'cost_policy_rejected','filled_entries','trades_closed',
                                         'starting_balance_usd','ending_balance_usd','net_profit_usd',
                                         'return_pct','max_drawdown_pct','win_rate_pct','profit_factor')}
        print(json.dumps(overview, indent=2))
        print('Reports:', (a.workspace / OUTPUT_DIR).resolve())


if __name__ == '__main__':
    main()
