"""Shared final-evaluation gate and read-only access to the saved team result.

Imports intentionally use only the standard library. Viewing results never loads
the model, opens the raw test table, generates predictions, or bootstraps scores.
"""
import ast
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selected_run(root, selection):
    rel = Path(selection['selected_run'])
    run = (root / rel).resolve()
    if rel.is_absolute() or not run.is_relative_to((root / 'artifacts').resolve()):
        raise ValueError('Selected run must be a repository-relative path within artifacts/.')
    return run


def assert_evaluation_allowed(root, run_dir):
    """Validate the team decision before any test labels or pipeline are loaded."""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    selection = _read(root / 'artifacts/shared/final_model_selection.json')
    selected = _selected_run(root, selection)
    if run_dir != selected:
        raise ValueError(f'This is not the final selected run. The team selected {selection["selected_run"]}.')
    marker = root / 'artifacts/shared/heldout_evaluation.json'
    if marker.exists():
        raise RuntimeError('The shared held-out evaluation has already started or completed. Read the saved results; do not evaluate again.')
    for family in ['boosting', 'logistic_random_forest']:
        for name in ['test_metrics.json', 'test_predictions.csv']:
            if next((root / 'artifacts' / family).glob(f'*/{name}'), None) is not None:
                raise RuntimeError('Existing held-out outputs found. Resolve the shared evaluation record; do not evaluate again.')
    if selection.get('status') != 'approved_before_test':
        raise ValueError('The final team selection has not been approved.')
    meta = _read(selected / 'selected_metadata.json')
    if meta.get('quick_run') or not meta.get('threshold_rationale', '').strip():
        raise ValueError('Require a full run with a recorded threshold rationale.')
    for key in ['model', 'strategy', 'threshold', 'threshold_rule', 'threshold_rationale']:
        if meta.get(key) != selection.get(key):
            raise ValueError(f'{key} differs from the approved final decision.')
    if meta.get('fingerprints_sha256') != selection.get('fingerprints_sha256'):
        raise ValueError('Run fingerprints differ from the approved final decision.')
    # Check both current inputs and the frozen copies used by the archived evaluator.
    files = {
        'train': (root / 'notebooks' / meta['train_path']).resolve(),
        'test': (root / 'notebooks' / meta['test_path']).resolve(),
        'pipeline': selected / 'selected_pipeline.joblib',
        'folds': selected / 'cv_folds.csv',
        'feature_sets': selected / 'feature_sets.json',
    }
    for key, path in files.items():
        if _sha(path) != selection['fingerprints_sha256'][key]:
            raise ValueError(f'{key} changed after selection.')
    source = root / 'scripts/provenance/06b_section11_executed.py'
    if _sha(source) != selection['test_code_sha256']:
        raise ValueError('Archived evaluation code differs from the executed source fingerprint.')
    return selection, meta


def load_saved_results(root):
    """Read the selected run's results, or its committed evidence snapshot.

The snapshot allows a fresh checkout to display results without local training
artifacts or raw data. Both copies are verified when both are available.
"""
    root = Path(root).resolve()
    selection = _read(root / 'artifacts/shared/final_model_selection.json')
    completion = _read(root / 'artifacts/shared/heldout_evaluation.json')
    if completion.get('status') != 'completed':
        raise ValueError('No completed shared evaluation is recorded.')
    if completion.get('selected_run') != selection['selected_run']:
        raise ValueError('Selection and completion records refer to different runs.')
    run = _selected_run(root, selection)
    snapshot = root / 'reports/final_evaluation'
    manifest = _read(snapshot / 'provenance.json')
    for name, expected in manifest['snapshot_sha256'].items():
        if _sha(snapshot / name) != expected:
            raise ValueError(f'Saved evidence snapshot changed: {name}')
    for directory in [run, snapshot]:
        metrics_path = directory / 'test_metrics.json'
        if metrics_path.exists() and _sha(metrics_path) != completion['test_metrics_sha256']:
            raise ValueError(f'Saved test metrics changed: {metrics_path}')
    metrics = _read(run / 'test_metrics.json' if (run / 'test_metrics.json').exists()
                    else snapshot / 'test_metrics.json')
    if metrics['point']['threshold'] != selection['threshold']:
        raise ValueError('Saved test threshold differs from the final decision.')
    return {'selection': selection, 'completion': completion, 'test': metrics}


def run_selected_test(root, run_dir):
    """One permitted evaluation; never used to refresh already saved results.

This preserves the exact original test computation while enforcing a shared
cross-notebook selection check and an exclusive start/completion record.
"""
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    selection, _ = assert_evaluation_allowed(root, run_dir)
    source = root / 'scripts/provenance/06b_section11_executed.py'
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {'N_BOOT': 2000, 'BOOT_SEED': 4012}
    exec(compile(tree, str(source), 'exec'), namespace)
    marker = root / 'artifacts/shared/heldout_evaluation.json'
    now = lambda: datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).isoformat()
    record = {'status': 'in_progress', 'selected_run': selection['selected_run'],
              'started_at': now(), 'selection_record': 'artifacts/shared/final_model_selection.json'}
    # An exclusive create also prevents two notebook processes starting together.
    with marker.open('x') as stream:
        json.dump(record, stream, indent=2)
    previous_cwd = Path.cwd()
    try:
        os.chdir(root / 'notebooks')
        result = namespace['run_final_test'](run_dir)
        record.update(status='completed', completed_at=now(),
                      test_metrics_sha256=_sha(run_dir / 'test_metrics.json'),
                      test_predictions_sha256=_sha(run_dir / 'test_predictions.csv'))
    except Exception as exc:
        record.update(status='failed', error=type(exc).__name__ + ': ' + str(exc))
        raise
    finally:
        os.chdir(previous_cwd)
        marker.write_text(json.dumps(record, indent=2) + '\n')
    return result
