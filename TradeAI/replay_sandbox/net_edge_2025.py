"""Net-of-cost Precision M5 research, training exclusively on pre-2026 data.

This is not the existing direction classifier. BUY/SELL/HOLD labels reflect
30-minute NET trade outcome at minimum lot, using historical future entry/exit
prices and spreads ONLY FOR LABELS. Features are closed-candle causal inputs.

Commands:
  python -m TradeAI.replay_sandbox.net_edge_2025 prepare
  python -m TradeAI.replay_sandbox.net_edge_2025 train

No MT5 imports, no actual orders, no promotion to production/demo/live.
The 2026 holdout was previously inspected and is NOT a clean final test.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from . import holdout_2025 as h

SYMBOLS = ('EURUSD', 'GBPUSD', 'USDJPY')
BLACKOUT_HOURS = (23, 0)  # Historical timestamp clock, not verified UTC wall clock.
STUDY_NAME = 'NET_EDGE_COST_AWARE_M5_2025_V1'
OUTPUT_FOLDER = 'net_edge_2025'
TRAIN_START = '2025-01-01'
TRAIN_END = '2025-08-01'
VALIDATION_END = '2025-10-01'
TEST_END = '2026-01-01'
SPREAD_FLOOR_PIPS = 0.8
LABEL_MIN_NET_PIPS = 1.0
MIN_ACTION_CONFIDENCE = 0.55
# Predeclared validation-only thresholds; selection never looks at Oct-Dec test.
VALIDATION_THRESHOLD_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65)
MIN_VALIDATION_TRADES = 30
MIN_VALIDATION_NET_PIPS_PER_TRADE = 0.5
EXIT_HORIZON = 6


def _sha(path: Path) -> str:
    hash_ = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            hash_.update(block)
    return hash_.hexdigest()


def _workspace(path: Path):
    path = h._reject_live_workspace(path)
    manifest = h._manifest(path)
    if manifest['train_exclusive_end'] != TEST_END:
        raise RuntimeError('Not the isolated 2025-cutoff research workspace')
    policy_path = path / 'cost_policy_2025.json'
    if not policy_path.is_file():
        raise FileNotFoundError('Run cost_aware_2025 prepare before net_edge_2025')
    policy = json.loads(policy_path.read_text(encoding='utf8'))
    if (policy.get('development_end_exclusive') != TEST_END or
        tuple(policy.get('blackout_hours_in_archived_timestamp_clock', [])) != BLACKOUT_HOURS):
        raise ValueError('Frozen cost policy no longer matches the research design')
    # Enforce that policy was built using the same locked training study.
    if policy.get('archive_sha256') != manifest['replay_archive_sha256']:
        raise RuntimeError('Policy and isolated dataset mismatch')
    with ZipFile(path / 'replay_2026_jan_sep.zip') as z:
        spec = json.loads(z.read('broker_profile.json'))
    if spec.get('account_currency') != 'USD':
        raise ValueError('Only USD-denominated account supported')
    return path, manifest, policy, spec, _sha(policy_path)


def net_cost_labels(raw: pd.DataFrame, spec: dict, max_spread_pips: float) -> pd.DataFrame:
    """Return decision-aligned labels, never leak next-open/future spreads as features.

    `raw` should be _clean_m5 output with `time`, OHLC, and broker spread in points.
    The spread associated with the current completed candle is observable, and
    is used for the entry *eligibility* gate. Future spread is ONLY target cost.
    """
    point = float(spec['point'])
    digits = int(spec['digits'])
    pip = point * (10 if digits in (3, 5) else 1)
    lots = float(spec['volume_min'])
    commission = lots * float(spec['commission_per_lot'])
    contract = float(spec.get('contract_size') or 100000)
    slip = 2.0 * float(spec.get('slippage_pips', 0.0))
    if min(point,pip,lots,contract) <= 0 or commission < 0 or slip < 0:
        raise ValueError('Invalid broker costs or lot specification')

    t = raw['time']
    decision = t + pd.Timedelta(minutes=5)
    start = t.shift(-1)
    last = t.shift(-EXIT_HORIZON)
    exit_at = last + pd.Timedelta(minutes=5)
    contig = ((start - t == pd.Timedelta(minutes=5)) &
              (last - t == pd.Timedelta(minutes=30)))
    entry_bid = raw['open'].shift(-1)
    exit_bid = raw['close'].shift(-EXIT_HORIZON)
    entry_spread = np.maximum(raw['spread'].shift(-1) * point / pip, SPREAD_FLOOR_PIPS)
    exit_spread = np.maximum(raw['spread'].shift(-EXIT_HORIZON) * point / pip, SPREAD_FLOOR_PIPS)
    observed_spread = np.maximum(raw['spread'] * point / pip, SPREAD_FLOOR_PIPS)

    # Only EURUSD, GBPUSD (USD quote) and USDJPY (USD base) are supported.
    if spec['symbol'] in ('EURUSD', 'GBPUSD'):
        pip_value_usd = contract * lots * pip
    elif spec['symbol'] == 'USDJPY':
        pip_value_usd = contract * lots * pip / exit_bid
    else:
        raise ValueError('Unsupported symbol for the net-edge research')
    commission_pips = commission / pip_value_usd
    signed_move = (exit_bid - entry_bid) / pip
    buy_net_pips = signed_move - entry_spread - slip - commission_pips
    sell_net_pips = -signed_move - exit_spread - slip - commission_pips

    # Historical clock applies equally to both entries and projected exits.
    eligible = (
        contig &
        (decision >= pd.Timestamp(TRAIN_START, tz='UTC')) &
        (decision < pd.Timestamp(TEST_END, tz='UTC')) &
        (exit_at < pd.Timestamp(TEST_END, tz='UTC')) &
        ~decision.dt.hour.isin(BLACKOUT_HOURS) &
        ~exit_at.dt.hour.isin(BLACKOUT_HOURS) &
        observed_spread.notna() &
        (observed_spread <= max_spread_pips + 1e-9) &
        np.isfinite(buy_net_pips) & np.isfinite(sell_net_pips) &
        (entry_bid > 0) & (exit_bid > 0)
    )
    target = np.full(len(raw), 1, dtype=np.int8)  # 0 SELL, 1 HOLD, 2 BUY
    buy_good = (buy_net_pips >= LABEL_MIN_NET_PIPS) & (buy_net_pips > sell_net_pips)
    sell_good = (sell_net_pips >= LABEL_MIN_NET_PIPS) & (sell_net_pips > buy_net_pips)
    target[np.asarray(buy_good.fillna(False))] = 2
    target[np.asarray(sell_good.fillna(False))] = 0
    return pd.DataFrame({
        'decision_time': decision,
        'label_valid': eligible.fillna(False).astype(bool),
        'net_buy_pips': buy_net_pips,
        'net_sell_pips': sell_net_pips,
        'observed_spread_pips': observed_spread,
        'target_class': target,
        'entry_time': start,
        'exit_time': exit_at,
    })


def build_rows(workspace: Path, symbol: str, spec: dict, spread_cap: float) -> tuple[pd.DataFrame, list[str]]:
    # Load the EXACT previously frozen causal feature implementation, not a
    # possibly edited current project module. The original cutoff manifest
    # hashes this snapshot and _workspace verifies those hashes.
    source = workspace / 'Trade_Bot_Training' / 'precision_feature_engine.py'
    spec_source = importlib.util.spec_from_file_location('net_edge_frozen_2025_features', source)
    if spec_source is None or spec_source.loader is None:
        raise RuntimeError('Missing frozen feature snapshot')
    feature_module = importlib.util.module_from_spec(spec_source)
    spec_source.loader.exec_module(feature_module)
    _clean_m5 = feature_module._clean_m5
    build_precision_features = feature_module.build_precision_features
    precision_feature_columns = feature_module.precision_feature_columns
    p = workspace / 'artifacts' / 'datasets' / 'direction_m5_raw' / f'{symbol}_M5.csv.gz'
    candles = pd.read_csv(
        p, compression='gzip',
        usecols=['time','open','high','low','close','tick_volume','spread'],
    )
    candles['time'] = pd.to_datetime(candles['time'], utc=True, errors='raise')
    # Include 2024 warmup so all RSI/ATR/range and volume features are causal.
    candles = candles.loc[(candles['time'] >= '2024-12-01') &
                          (candles['time'] < TEST_END)].reset_index(drop=True)
    clean = _clean_m5(candles)
    feat = build_precision_features(clean)
    if not feat['time'].equals(clean['time']):
        raise ValueError(f'{symbol} feature rows not aligned to raw labels')
    labels = net_cost_labels(clean, spec, spread_cap)
    if not labels['decision_time'].equals(feat['decision_time']):
        raise RuntimeError(f'{symbol} decision and label timestamps do not match')
    cols = precision_feature_columns(feat) + ['observed_spread_pips']
    feat['observed_spread_pips'] = labels['observed_spread_pips'].astype('float32')
    keep = labels['label_valid'].to_numpy() & np.isfinite(feat[cols].to_numpy(dtype='float64')).all(axis=1)
    selected = feat.loc[keep, ['decision_time'] + cols].copy().reset_index(drop=True)
    for target_col in ('target_class','net_buy_pips','net_sell_pips','entry_time','exit_time'):
        selected[target_col] = labels.loc[keep, target_col].to_numpy()
    if len(selected) < 5000:
        raise RuntimeError(f'{symbol}: too few eligible 2025 observations ({len(selected)})')
    assert selected['decision_time'].max() < pd.Timestamp(TEST_END, tz='UTC')
    return selected, cols


def split_rows(rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    t = pd.to_datetime(rows['decision_time'], utc=True)
    e = pd.to_datetime(rows['exit_time'], utc=True)
    groups = {
        'train': rows.loc[(t < pd.Timestamp(TRAIN_END, tz='UTC')) &
                          (e <= pd.Timestamp(TRAIN_END, tz='UTC'))],
        'validation': rows.loc[(t >= pd.Timestamp(TRAIN_END, tz='UTC')) &
                               (t < pd.Timestamp(VALIDATION_END, tz='UTC')) &
                               (e <= pd.Timestamp(VALIDATION_END, tz='UTC'))],
        'test': rows.loc[(t >= pd.Timestamp(VALIDATION_END, tz='UTC')) &
                         (t < pd.Timestamp(TEST_END, tz='UTC')) &
                         (e < pd.Timestamp(TEST_END, tz='UTC'))],
    }
    for name, group in groups.items():
        if len(group) < 200:
            raise ValueError(f'{name}: too few observations ({len(group)})')
    return groups


def count_classes(rows: pd.DataFrame) -> dict:
    c = rows['target_class'].value_counts()
    return {name: int(c.get(i,0)) for i,name in ((0,'SELL'),(1,'HOLD'),(2,'BUY'))}


def prepare(workspace: Path) -> dict:
    workspace, manifest, policy, profile, policy_hash = _workspace(workspace)
    output = workspace / OUTPUT_FOLDER
    if output.exists():
        raise FileExistsError(f'Research output exists; refusing to overwrite: {output}')
    data = {
        'study': STUDY_NAME,
        'status': 'PREPARED_NO_NEW_MODEL_TRAINED',
        'account_currency': 'USD',
        'models_unchanged': True,
        'classification': '2025_DEVELOPMENT_DATA_ONLY__2026_ALREADY_INSPECTED',
        'source_manifest_sha256': _sha(workspace/'cutoff_manifest.json'),
        'source_cost_policy_sha256': policy_hash,
        'blackout_hours_historical_clock': list(BLACKOUT_HOURS),
        'split_boundaries': {'train_end_exclusive':TRAIN_END, 'validation_end_exclusive':VALIDATION_END,
                             'test_end_exclusive':TEST_END},
        'minimum_label_net_pips': LABEL_MIN_NET_PIPS,
        'action_thresholds_frozen_grid': list(VALIDATION_THRESHOLD_GRID),
        'threshold_selection':'lowest threshold whose validation-only nonoverlapping trades >=30 and mean realized net pips >= +0.5, otherwise abstain',
        'feature_contract': 'Same causal completed-bar precision M5 features plus completed-bar spread; no future prices or spreads as features',
        'labels': 'Fixed 30-minute BUY/SELL actual future fill-equivalent net pips vs 1-pip floor, else HOLD',
        'pairs': {},
    }
    for symbol in SYMBOLS:
        spec = dict(profile['symbols'][symbol], symbol=symbol)
        cap = float(policy['pairs'][symbol]['max_last_closed_spread_pips'])
        rows, cols = build_rows(workspace, symbol, spec, cap)
        groups = split_rows(rows)
        stats = {'spread_cap_pips':cap, 'features':len(cols), 'feature_sha256':hashlib.sha256('\n'.join(cols).encode()).hexdigest(),
                 'samples':{}, 'label_distribution':{}}
        for name, group in groups.items():
            stats['samples'][name] = len(group)
            stats['label_distribution'][name] = count_classes(group)
            if min(count_classes(group).values()) < 30:
                raise RuntimeError(f'{symbol} {name}: insufficient samples of one class; review labels')
        data['pairs'][symbol] = stats
        print(symbol, 'rows', sum(stats['samples'].values()),
              '| train', stats['samples']['train'], '| validation',stats['samples']['validation'],
              '| test',stats['samples']['test'], '| classes',stats['label_distribution']['train'], flush=True)
    output.mkdir(parents=True, exist_ok=False)
    (output/'prepared.json').write_text(json.dumps(data, indent=2)+'\n',encoding='utf-8')
    return data


def select_nonoverlap(preds: pd.DataFrame, threshold: float = MIN_ACTION_CONFIDENCE) -> pd.DataFrame:
    """Single pair: enforce one active 30m trade, then admit next candidate."""
    selected = []
    next_entry = pd.Timestamp('1900-01-01', tz='UTC')
    for i, row in preds.iterrows():
        if row['predicted_class'] == 1 or row['confidence'] < threshold:
            continue
        if row['entry_time'] < next_entry:
            continue
        selected.append(i)
        next_entry = row['exit_time']
    return preds.loc[selected].copy()


def net_pips_of(selected: pd.DataFrame) -> np.ndarray:
    return np.where(selected['predicted_class'] == 2,selected['net_buy_pips'],selected['net_sell_pips'])


def validation_threshold(validation: pd.DataFrame, probs: np.ndarray) -> tuple[float | None, list[dict]]:
    """Select a *research* threshold using validation only, or abstain.

    Thresholds and acceptance rules were frozen before looking at the Oct-Dec
    test. No automatic override when validation cannot establish net edge.
    """
    scored = validation[['decision_time','entry_time','exit_time','net_buy_pips','net_sell_pips']].copy().reset_index(drop=True)
    scored['predicted_class'] = probs.argmax(axis=1)
    scored['confidence'] = probs.max(axis=1)
    candidates = []
    for threshold in VALIDATION_THRESHOLD_GRID:
        accepted = select_nonoverlap(scored, threshold)
        realized = net_pips_of(accepted)
        trades = len(accepted)
        mean_net = float(realized.mean()) if trades else None
        total_net = float(realized.sum()) if trades else None
        passed = trades >= MIN_VALIDATION_TRADES and mean_net is not None and mean_net >= MIN_VALIDATION_NET_PIPS_PER_TRADE
        candidates.append({'threshold':threshold,'trades':trades,'mean_net_pips':mean_net,
                           'total_net_pips':total_net,'passed':passed})
    viable = [c for c in candidates if c['passed']]
    if not viable:
        return None, candidates
    # Choose the simplest sufficient confidence requirement; no test-set fitting.
    return float(min(c['threshold'] for c in viable)), candidates


def train(workspace: Path) -> dict:
    from xgboost import XGBClassifier
    workspace, manifest, policy, profile, policy_hash = _workspace(workspace)
    output = workspace / OUTPUT_FOLDER
    manifest_file = output / 'prepared.json'
    if not manifest_file.is_file():
        raise FileNotFoundError('First run: net_edge_2025 prepare')
    prepared = json.loads(manifest_file.read_text(encoding='utf-8'))
    if (prepared['study'] != STUDY_NAME or prepared['source_cost_policy_sha256'] != policy_hash or
        prepared['source_manifest_sha256'] != _sha(workspace/'cutoff_manifest.json')):
        raise RuntimeError('Source study or cost policy changed; refusing train')
    result_path = output/'training_report.json'
    if result_path.exists() or (output/'models').exists():
        raise FileExistsError('Already trained/partially trained. Refusing to overwrite model results.')
    reports = {'study':STUDY_NAME,'classification':'PRE2026_CHRONOLOGICAL_RESEARCH_NOT_LIVE_APPROVED',
               'production_promoted':False,'symbols':{}}
    for symbol in SYMBOLS:
        spec = dict(profile['symbols'][symbol],symbol=symbol)
        rows, cols = build_rows(workspace,symbol,spec,float(policy['pairs'][symbol]['max_last_closed_spread_pips']))
        groups = split_rows(rows)
        if hashlib.sha256('\n'.join(cols).encode()).hexdigest() != prepared['pairs'][symbol]['feature_sha256']:
            raise RuntimeError('Feature drift since prepared')
        for stage, group in groups.items():
            if len(group) != prepared['pairs'][symbol]['samples'][stage] or count_classes(group) != prepared['pairs'][symbol]['label_distribution'][stage]:
                raise RuntimeError(f'{symbol}: label/source drift since prepare')
        params = dict(n_estimators=700, max_depth=4, min_child_weight=25,
                      learning_rate=0.04, early_stopping_rounds=60,
                      subsample=0.88, colsample_bytree=0.82,
                      reg_alpha=0.20, reg_lambda=12.0, gamma=0.02,
                      objective='multi:softprob', num_class=3, eval_metric='mlogloss',
                      tree_method='hist', max_bin=128, random_state=42, n_jobs=4)
        model = XGBClassifier(**params)
        tr, val, test = (groups[k] for k in ('train','validation','test'))
        model.fit(tr[cols].astype('float32'),tr['target_class'].to_numpy(),
                  eval_set=[(val[cols].astype('float32'),val['target_class'].to_numpy())], verbose=False)
        validation_probs = model.predict_proba(val[cols].astype('float32'))
        locked_threshold, search = validation_threshold(val, validation_probs)
        probs = model.predict_proba(test[cols].astype('float32'))
        predicted = np.asarray(probs).argmax(axis=1).astype('int8')
        confidence = np.asarray(probs).max(axis=1)
        report = test[['decision_time','entry_time','exit_time','target_class','net_buy_pips','net_sell_pips']].copy().reset_index(drop=True)
        report['predicted_class'] = predicted
        report['confidence'] = confidence
        threshold = locked_threshold if locked_threshold is not None else 1.01
        report['selected'] = ((predicted != 1) & (confidence >= threshold))
        accepted = select_nonoverlap(report, threshold)
        observed = net_pips_of(accepted)
        accept_wins = int((observed > 0).sum())
        month_results = {}
        if len(accepted):
            accepted = accepted.copy()
            accepted['net_pips'] = observed
            for month,part in accepted.groupby(accepted['decision_time'].dt.strftime('%Y-%m')):
                month_results[str(month)] = {'trades':len(part), 'net_pips':round(float(part['net_pips'].sum()),3)}
        stats = {
            '2025_test_window':'2025-10-01 THROUGH 2025-12-31',
            'test_rows':len(test),
            'locked_validation_threshold':locked_threshold,
            'validation_gate_passed':locked_threshold is not None,
            'validation_threshold_trials':search,
            'raw_action_candidates':int(report['selected'].sum()),
            'non_overlapping_test_trades':len(accepted),
            'net_winning_trades':accept_wins,
            'net_win_rate_pct':round(100*accept_wins/len(accepted),3) if len(accepted) else None,
            'mean_net_pips_per_trade':round(float(np.mean(observed)),4) if len(accepted) else None,
            'total_net_pips':round(float(np.sum(observed)),3) if len(accepted) else None,
            'monthly':month_results,
            'selected_trees':int(model.best_iteration)+1,
            'model_not_promoted':True,
            'caveat':'Per-symbol non-overlap approximate research; no account-level margin, gap or multi-pair execution. NOT a real broker replay.'
        }
        model_dir = output/'models'/symbol
        model_dir.mkdir(parents=True, exist_ok=False)
        model.save_model(model_dir/'model.json')
        metadata = {'symbol':symbol, 'study':STUDY_NAME, 'features':cols,'feature_sha256':prepared['pairs'][symbol]['feature_sha256'],
                    'class_map':{'SELL':0,'HOLD':1,'BUY':2},'confidence_threshold_selected_on_validation':locked_threshold,
                    'model_sha256':_sha(model_dir/'model.json'),'train_rows':len(tr),'validation_rows':len(val),
                    'test_rows':len(test),
            'locked_validation_threshold':locked_threshold,
            'validation_gate_passed':locked_threshold is not None,
            'validation_threshold_trials':search,'development_labels':'FUTURE fill-equivalent net pips used for targets only',
                    'independent_2026_test':False}
        (model_dir/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf8')
        report.to_csv(output/f'{symbol}_test_predictions.csv.gz',compression='gzip',index=False)
        reports['symbols'][symbol] = stats
        print(symbol, ':',json.dumps(stats,indent=2),flush=True)
    result_path.write_text(json.dumps(reports,indent=2)+'\n',encoding='utf8')
    return reports


def main():
    p = argparse.ArgumentParser(description='2025-only net-edge labels and research model; NO MT5 orders')
    p.add_argument('command',choices=('prepare','train'))
    p.add_argument('--workspace',type=Path,default=h.DEFAULT_WORKSPACE)
    a = p.parse_args()
    if a.command=='prepare':
        x = prepare(a.workspace)
        print(json.dumps({'study':x['study'],'status':x['status'], 'pairs':x['pairs']},indent=2))
        print('Prepared research:',(a.workspace/OUTPUT_FOLDER).resolve())
    else:
        x=train(a.workspace)
        print('FINISHED: 2025 research training only; NO live promotion.')
        print('Report:',(a.workspace/OUTPUT_FOLDER/'training_report.json').resolve())

if __name__=='__main__':
    main()
