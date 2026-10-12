"""Final Precision M5 refit on ALL pre-2026 labelled decisions.

This file is copied into the isolated research workspace as
Trade_Bot_Training/refit_precision_2025.py, then executed ONLY in that workspace.
Production models are never touched. DO NOT use any 2026 data or metrics here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import shutil

from xgboost import XGBClassifier

from Trade_Bot_Training.precision_config import MODEL_PARAMS, SYMBOLS, model_dir, report_dir
from Trade_Bot_Training.precision_dataset_builder import build_symbol_frame
from Trade_Bot_Training.precision_feature_engine import precision_feature_columns

CUTOFF = '2026-01-01'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def refit_symbol(symbol: str) -> dict:
    base = model_dir(symbol)
    meta_path = base / 'metadata.json'
    model_path = base / 'model.json'
    if not meta_path.exists() or not model_path.exists():
        raise FileNotFoundError(f'Missing initial internal-test candidate for {symbol}')
    meta = json.loads(meta_path.read_text(encoding='utf8'))
    if meta.get('final_refit_on_all_pre2026'):
        raise RuntimeError(f'{symbol}: final refit already performed')
    selected_trees = int(meta['selected_trees'])
    if not 1 <= selected_trees <= int(MODEL_PARAMS['n_estimators']):
        raise ValueError(f'{symbol}: invalid validation-selected tree count {selected_trees}')

    # Input comes ONLY from the isolated truncated M5 raw files. Existing
    # label generator removes last 6 rows if the 30-minute target isn't present.
    frame, data_summary = build_symbol_frame(symbol, verbose=False)
    if frame.empty or str(frame['decision_time'].max())[:10] >= CUTOFF:
        raise RuntimeError(f'{symbol}: pre-2026 cutoff violated or training set empty')
    columns = precision_feature_columns(frame)
    if list(meta['feature_columns']) != list(columns):
        raise RuntimeError(f'{symbol}: feature schema differs from initial candidate')
    X = frame[columns].astype('float32')
    y = frame['target_class'].astype('int8')
    if set(y.unique().tolist()) != {0, 1, 2}:
        raise RuntimeError(f'{symbol}: not all three target classes available')

    # Back up the exact internally evaluated candidate before the full-history
    # refit. The earlier pre-2026 test gate is evidence for that *older* fit,
    # not independent evaluation evidence of the final refitted model.
    internal_dir = report_dir(symbol) / 'internal_pre_refit_candidate'
    internal_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(model_path, internal_dir / 'model.json')
    shutil.copy2(meta_path, internal_dir / 'metadata.json')
    original_model_hash = sha256(internal_dir / 'model.json')

    print(f'FINAL FULL PRE-2026 REFIT {symbol}: {len(frame):,} rows, '
          f'ending {frame["decision_time"].iloc[-1]}, {selected_trees} trees', flush=True)
    params = dict(MODEL_PARAMS)
    params['n_estimators'] = selected_trees
    params.pop('early_stopping_rounds', None)
    final_model = XGBClassifier(**params)
    final_model.fit(X, y, verbose=False)
    final_model.save_model(model_path)

    meta['final_refit_on_all_pre2026'] = True
    meta['final_refit_created_utc'] = datetime.now(timezone.utc).isoformat()
    meta['final_refit_train_rows'] = len(frame)
    meta['final_refit_last_decision_time'] = str(frame['decision_time'].iloc[-1])
    meta['final_refit_first_decision_time'] = str(frame['decision_time'].iloc[0])
    meta['final_refit_tree_count_selected_without_2026'] = selected_trees
    meta['initial_candidate_sha256'] = original_model_hash
    meta['internal_pre2026_test_metrics_apply_to_initial_candidate_only'] = True
    meta['production_gate_is_pre_refit_internal_candidate_gate'] = True
    meta['post_refit_validation'] = '2026 RETROSPECTIVE HOLDOUT REPLAY ONLY; NOT REALTIME FORWARD'
    meta['important'] = (
        'Full pre-2026 refit uses all available 2025 data including former internal test. '
        'Stored final_test / historical gate refer ONLY to the earlier initial candidate. '
        '2026 holdout remains unseen by fit and hyperparameter selection; do not tune on it.'
    )
    meta_path.write_text(json.dumps(meta, indent=2)+'\n', encoding='utf8')
    result = {
        'symbol': symbol,
        'rows': len(frame),
        'first_decision': str(frame['decision_time'].iloc[0]),
        'last_decision': str(frame['decision_time'].iloc[-1]),
        'fitted_trees': selected_trees,
        'baseline_model_sha256': original_model_hash,
        'refitted_model_sha256': sha256(model_path),
        'original_gate_passed': bool(meta['production_ready_by_historical_gate']),
    }
    (report_dir(symbol) / 'full_refit_manifest.json').write_text(
        json.dumps(result, indent=2)+'\n', encoding='utf8')
    return result


def main() -> None:
    report = {sym: refit_symbol(sym) for sym in SYMBOLS}
    out = report_dir(SYMBOLS[0]).parent / 'all_pre2026_refit.json'
    out.write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print('ALL FINAL RE-FITS COMPLETE:', out, flush=True)


if __name__ == '__main__':
    main()
