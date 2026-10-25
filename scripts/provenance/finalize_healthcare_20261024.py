"""Audit saved validation runs, then optionally execute the existing 06b test function once."""
import argparse
import ast
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path('/Users/vl/bt4012-group14-healthcare-fraud-detection')
RUNS = {
    '06b': ROOT / 'artifacts/boosting/20261011_232303_full',
    '06c': ROOT / 'artifacts/logistic_random_forest/20261024_144141_full',
}
RATIONALE = (
    'Pei confirmed team agreement in this chat on 24 October 2026, before final '
    'held-out evaluation. Use the existing joint mean-CV-AP selection policy: '
    'candidates within one standard error of the highest mean AP qualify; prefer '
    'original, then class_weight, then smote, then higher mean AP. '
    'The operational objective is at least 80% recall on development OOF predictions. '
    'Among thresholds meeting this objective, choose the highest precision. '
    'This prioritises identifying potentially fraudulent providers while accepting '
    'the observed validation review workload of 701/4328 providers. '
    'This is a project assumption, not an empirically established business cost '
    'optimum or a guarantee of 80% recall on unseen data.'
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def audit():
    metas, summaries, predictions, fold_tables = {}, [], {}, {}
    ref_ids = None
    for label, run in RUNS.items():
        m = json.loads((run / 'selected_metadata.json').read_text())
        assert not m['quick_run'], f'{label}: quick run'
        for key in ['train', 'test']:
            p = (ROOT / 'notebooks' / m[f'{key}_path']).resolve()
            assert sha(p) == m['fingerprints_sha256'][key], f'{label}: {key} changed'
        for key, name in [('feature_sets', 'feature_sets.json'), ('folds', 'cv_folds.csv'),
                          ('pipeline', 'selected_pipeline.joblib')]:
            assert sha(run / name) == m['fingerprints_sha256'][key], f'{label}: {key} changed'
        if metas:
            ref = metas['06b']
            for key in ['train', 'test', 'feature_sets', 'folds']:
                assert m['fingerprints_sha256'][key] == ref['fingerprints_sha256'][key]
            assert m['feature_order'] == ref['feature_order']
            assert m['threshold_rule'] == ref['threshold_rule']
            for key in ['tolerance_se', 'strategy_preference']:
                assert m['selection_rule'][key] == ref['selection_rule'][key]
        metas[label] = m
        f = pd.read_csv(run / 'fold_metrics.csv')
        o = pd.read_csv(run / 'oof_predictions.csv', float_precision='round_trip')
        s = pd.read_csv(run / 'cv_summary.csv')
        search = pd.read_csv(run / 'cv_search_results.csv')
        assert len(search) == (120 if label == '06b' else 240)
        folds = pd.read_csv(run / 'cv_folds.csv').set_index('Provider').sort_index()
        for row in s.itertuples():
            g = o[(o.model == row.model) & (o.strategy == row.strategy)].set_index('Provider').sort_index()
            assert len(g) == m['n_train'] and g.index.is_unique
            assert g.oof_prob.notna().all() and g.oof_prob.between(0, 1).all()
            identity = g[['fold', 'y_true']]
            if ref_ids is None:
                ref_ids = identity
            pd.testing.assert_frame_equal(identity, ref_ids)
            assert g.index.equals(folds.index)
            assert np.array_equal(g['fold'], folds['fold'])
            assert np.array_equal(g.y_true, folds.PotentialFraud)
            fs = f[(f.model == row.model) & (f.strategy == row.strategy)].sort_values('fold')
            assert len(fs) == 5 and fs.fold.nunique() == 5
            recalculated = []
            for k in range(5):
                z = g[g.fold == k]
                ap = average_precision_score(z.y_true, z.oof_prob)
                assert np.isclose(ap, fs.loc[fs.fold == k, 'ap'].iloc[0])
                recalculated.append(ap)
            assert np.isclose(np.mean(recalculated), row.ap_mean)
            assert np.isclose(np.std(recalculated, ddof=1), row.ap_std)
            assert np.isclose(average_precision_score(g.y_true, g.oof_prob), row.pooled_oof_ap)
            if row.model != 'dummy':
                predictions[(label, row.model, row.strategy)] = g
                fold_tables[(label, row.model, row.strategy)] = fs
        summaries.append(s[s.model != 'dummy'].assign(notebook=label, source_run=run.name))
    candidates = pd.concat(summaries, ignore_index=True).sort_values('ap_mean', ascending=False)
    best = candidates.iloc[0]
    rule = metas['06b']['selection_rule']
    se = float(best.ap_std / np.sqrt(5))
    cutoff = float(best.ap_mean - rule['tolerance_se'] * se)
    candidates['within_tolerance'] = candidates.ap_mean >= cutoff
    order = {s: i for i, s in enumerate(rule['strategy_preference'])}
    candidates['strategy_rank'] = candidates.strategy.map(order)
    selected = candidates[candidates.within_tolerance].sort_values(
        ['strategy_rank', 'ap_mean'], ascending=[True, False]).iloc[0]
    label = selected.notebook
    m = metas[label]
    assert (selected.model, selected.strategy) == (m['model'], m['strategy']), 'Winner needs a different frozen pipeline'
    p = predictions[(label, selected.model, selected.strategy)]
    from sklearn.metrics import precision_recall_curve, confusion_matrix
    prec, rec, thr = precision_recall_curve(p.y_true, p.oof_prob)
    idx = int(np.argmax(np.where(rec[:-1] >= 0.8, prec[:-1], -1)))
    assert np.isclose(thr[idx], m['threshold'])
    cm = confusion_matrix(p.y_true, p.oof_prob >= m['threshold'], labels=[0, 1]).ravel()
    assert cm.tolist() == [m['oof_metrics'][k] for k in ['tn', 'fp', 'fn', 'tp']]
    comparisons = []
    baseline = fold_tables[(label, selected.model, selected.strategy)].set_index('fold').ap
    for key, fs in fold_tables.items():
        delta = baseline - fs.set_index('fold').ap
        comparisons.append({'comparison': f'{selected.model}/{selected.strategy} minus {key[1]}/{key[2]}',
                            'notebook': key[0], **{f'fold_{k}': float(delta[k]) for k in range(5)},
                            'mean_ap_difference': float(delta.mean())})
    decision = {
        'status': 'approved_before_test', 'approval_source': 'User confirmed team agreement in this chat',
        'approval_date': '2026-10-24', 'model': selected.model, 'strategy': selected.strategy,
        'source_notebook': label, 'selected_run': str(RUNS[label].relative_to(ROOT)),
        'source_runs': {k: str(v.relative_to(ROOT)) for k, v in RUNS.items()},
        'selection_rule': {'primary_metric': 'mean_fold_average_precision',
                           'tolerance_se': rule['tolerance_se'], 'standard_error_of_best': se,
                           'cutoff': cutoff, 'strategy_preference': rule['strategy_preference'],
                           'within_strategy_tiebreak': 'higher_mean_AP',
                           'note': 'Selection heuristic, not statistical equivalence.'},
        'threshold': m['threshold'], 'threshold_rule': m['threshold_rule'],
        'threshold_rationale': RATIONALE, 'oof_metrics': m['oof_metrics'],
        'fingerprints_sha256': m['fingerprints_sha256'],
    }
    return candidates, pd.DataFrame(comparisons), decision, m


def report(candidates, decision, test):
    pt = test['point']
    lines = ['# Final model selection and held-out evaluation', '',
             'Decision confirmed and evaluation performed on 24 October 2026 (Asia/Kuala_Lumpur).', '',
             '**Selected model: LightGBM, original class distribution.** The selection used only saved validation results from 06b and 06c. The model and its threshold were frozen before the held-out evaluation.', '',
             '## Validation comparison', '',
             '| Candidate | Strategy | Mean fold AP | Pooled OOF AP | Within tolerance |',
             '|---|---|---:|---:|---|']
    for r in candidates.itertuples():
        lines.append(f'| {r.model} | {r.strategy} | {r.ap_mean:.5f} | {r.pooled_oof_ap:.5f} | {"Yes" if r.within_tolerance else "No"} |')
    lines += ['', f'Tolerance cutoff: {decision["selection_rule"]["cutoff"]:.6f}. Eligible candidates are ranked by imbalance strategy (original, class weighting, SMOTE), then mean AP. This rule does not prefer logistic regression over tree models. It is not an equivalence test.', '',
              'All 18 non-dummy candidate combinations were checked against their saved OOF predictions. Data, feature-list and shared-fold fingerprints match between the two runs. The current files and selected pipelines match their recorded fingerprints.', '',
              '## Frozen threshold rationale', '', RATIONALE, '',
              f'Threshold: **{decision["threshold"]:.12f}**. No model fitting or threshold tuning was performed on the test set.', '',
              '## Held-out results', '',
              f'Test providers: {test["n_test"]}; positive labels: {test["n_test_fraud"]}.', '',
              '| Metric | Development OOF | Held-out test | Test 95% bootstrap interval |',
              '|---|---:|---:|---|']
    for k in ['ap', 'roc_auc', 'precision', 'recall', 'f1']:
        ci = test['bootstrap_95ci'][k]
        lines.append(f'| {k} | {decision["oof_metrics"][k]:.4f} | {pt[k]:.4f} | {ci["ci_low"]:.4f}–{ci["ci_high"]:.4f} |')
    lines += ['', f'True positives: **{pt["tp"]}**; false positives: **{pt["fp"]}**; false negatives: **{pt["fn"]}**; true negatives: **{pt["tn"]}**.', '',
              f'Providers flagged: **{pt["tp"] + pt["fp"]}/{test["n_test"]} ({pt["flagged_share"]:.2%})**.', '',
              'Intervals use the existing notebook protocol: 2,000 label-stratified provider bootstrap samples, seed 4012, with fixed predictions and threshold. They do not include retraining/model-selection uncertainty or uncertainty in prevalence, and assume providers can be resampled independently within class.', '',
              '## Interpretation and limits', '',
              'The 80% recall objective was selected on validation predictions; test recall is an observation, not a constraint to retune against. Do not change the selected model or threshold based on these results. Upstream full-dataset EDA/feature development and network-feature dependencies remain limitations, so this is not a completely isolated end-to-end evaluation. PotentialFraud is a proxy label, not proof of misconduct.', '',
              '## Handover', '',
              f'- Frozen run: `{decision["selected_run"]}`.',
              '- Use `selected_pipeline.joblib` with `selected_metadata.json` for the dashboard; preserve feature order and restore the three missing-value columns before prediction. Label the output a risk score.',
              '- `test_predictions.csv` and `test_metrics.json` are saved alongside that pipeline.',
              '- `artifacts/shared/final_model_selection.json` records the pre-test choice; `artifacts/shared/heldout_evaluation.json` records completion.',
              '- Paired fold AP differences and joint selection candidates are saved in `reports/06b_06c_paired_ap.csv` and `reports/06b_06c_final_candidates.csv`.',
              '- The original training notebooks were not rerun or altered. Section 11 of 06c should remain disabled. Swei can use the frozen model and results for SHAP/evaluation; Pei can build the dashboard.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', action='store_true')
    args = parser.parse_args()
    candidates, paired, decision, metadata = audit()
    print(candidates[['model', 'strategy', 'ap_mean', 'within_tolerance']].to_string(index=False))
    print('AUDIT PASSED; selected', decision['model'], decision['strategy'], decision['selected_run'])
    if not args.finalize:
        return
    shared = ROOT / 'artifacts/shared'
    selection_file = shared / 'final_model_selection.json'
    marker = shared / 'heldout_evaluation.json'
    for family in ['boosting', 'logistic_random_forest']:
        assert not list((ROOT / 'artifacts' / family).glob('*/test_metrics.json')), 'A held-out evaluation already exists; inspect before proceeding.'
    assert not marker.exists() and not selection_file.exists(), 'Shared final evaluation record already exists.'
    run = ROOT / decision['selected_run']
    notebook = ROOT / 'notebooks/06b_boosting_models.ipynb'
    n = json.loads(notebook.read_text())
    cell = next(''.join(c['source']) for c in n['cells'] if 'def run_final_test(' in ''.join(c.get('source', [])))
    tree = ast.parse(cell)
    # Reuse the reviewed, existing section-11 implementation; omit its UI configuration/dispatch.
    tree.body = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {'N_BOOT': 2000, 'BOOT_SEED': 4012}
    exec(compile(tree, str(notebook) + ':section11', 'exec'), namespace)
    decision['test_code_sha256'] = hashlib.sha256(cell.encode()).hexdigest()
    decision['recorded_at'] = datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).isoformat()
    dump(selection_file, decision)
    metadata['threshold_rationale'] = RATIONALE
    metadata['final_selection_record'] = str(selection_file.relative_to(ROOT))
    dump(run / 'selected_metadata.json', metadata)
    marker_data = {'status': 'in_progress', 'selected_run': decision['selected_run'],
                   'started_at': decision['recorded_at'], 'selection_record': str(selection_file.relative_to(ROOT))}
    with marker.open('x') as f:
        json.dump(marker_data, f, indent=2)
    os.chdir(ROOT / 'notebooks')
    print('Executing the existing 06b final-test function once.', flush=True)
    results = namespace['run_final_test'](run)
    test = json.loads((run / 'test_metrics.json').read_text())
    marker_data.update(status='completed', completed_at=datetime.now(ZoneInfo('Asia/Kuala_Lumpur')).isoformat(),
                       test_metrics_sha256=sha(run / 'test_metrics.json'),
                       test_predictions_sha256=sha(run / 'test_predictions.csv'))
    dump(marker, marker_data)
    candidates.to_csv(ROOT / 'reports/06b_06c_final_candidates.csv', index=False)
    paired.to_csv(ROOT / 'reports/06b_06c_paired_ap.csv', index=False)
    (ROOT / 'reports/final_model_evaluation.md').write_text(report(candidates, decision, test))
    print(results.to_string())
    print('Saved final evaluation report and shared records. No refitting or threshold adjustment performed.')


if __name__ == '__main__':
    main()
