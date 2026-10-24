"""Safety tests use fabricated temporary files, never the real held-out data."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.final_evaluation import assert_evaluation_allowed, load_saved_results, run_selected_test


class FinalEvaluationGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'artifacts/boosting/example_full'
        self.run.mkdir(parents=True)
        self.shared = self.root / 'artifacts/shared'
        self.shared.mkdir()
        (self.root / 'notebooks').mkdir()
        (self.root / 'data/processed').mkdir(parents=True)
        (self.root / 'scripts/provenance').mkdir(parents=True)
        source = self.root / 'scripts/provenance/06b_section11_executed.py'
        source.write_text('# fabricated source; never evaluated\n')
        paths = {'train': self.root / 'data/processed/train.csv',
                 'test': self.root / 'data/processed/test.csv',
                 'pipeline': self.run / 'selected_pipeline.joblib',
                 'folds': self.run / 'cv_folds.csv',
                 'feature_sets': self.run / 'feature_sets.json'}
        for key, path in paths.items():
            path.write_text('fabricated-' + key)
        fingerprints = {key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in paths.items()}
        self.selection = {'selected_run': str(self.run.relative_to(self.root)),
                          'status': 'approved_before_test', 'model': 'lgbm', 'strategy': 'original',
                          'threshold': .1, 'threshold_rule': {'rule': 'min_recall', 'value': .8},
                          'threshold_rationale': 'Fixture rationale.', 'fingerprints_sha256': fingerprints,
                          'test_code_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        self.meta = {**self.selection, 'quick_run': False,
                     'train_path': '../data/processed/train.csv',
                     'test_path': '../data/processed/test.csv'}
        self.write(self.shared / 'final_model_selection.json', self.selection)
        self.write(self.run / 'selected_metadata.json', self.meta)

    @staticmethod
    def write(path, data):
        path.write_text(json.dumps(data))

    def test_valid_preflight(self):
        selection, _ = assert_evaluation_allowed(self.root, self.run)
        self.assertEqual(selection['model'], 'lgbm')

    def test_same_run_id_in_other_family_rejected(self):
        other = self.root / 'artifacts/logistic_random_forest/example_full'
        with self.assertRaisesRegex(ValueError, 'not the final selected run'):
            assert_evaluation_allowed(self.root, other)

    def test_started_completed_or_failed_blocks_execution_before_imports(self):
        for status in ['in_progress', 'completed', 'failed']:
            with self.subTest(status=status):
                self.write(self.shared / 'heldout_evaluation.json', {'status': status})
                with patch('src.final_evaluation.ast.parse', side_effect=AssertionError('Must not load evaluator')):
                    with self.assertRaisesRegex(RuntimeError, 'already started or completed'):
                        run_selected_test(self.root, self.run)

    def test_existing_other_family_result_blocks(self):
        other = self.root / 'artifacts/logistic_random_forest/other_full'
        other.mkdir(parents=True)
        self.write(other / 'test_metrics.json', {})
        with self.assertRaisesRegex(RuntimeError, 'Existing held-out outputs'):
            assert_evaluation_allowed(self.root, self.run)

    def test_changed_threshold_rejected(self):
        self.meta['threshold'] = .2
        self.write(self.run / 'selected_metadata.json', self.meta)
        with self.assertRaisesRegex(ValueError, 'threshold differs'):
            assert_evaluation_allowed(self.root, self.run)

    def test_changed_model_rejected(self):
        (self.run / 'selected_pipeline.joblib').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'pipeline changed'):
            assert_evaluation_allowed(self.root, self.run)

    def test_changed_fingerprint_record_rejected(self):
        self.meta['fingerprints_sha256'] = dict(self.meta['fingerprints_sha256'], pipeline='wrong')
        self.write(self.run / 'selected_metadata.json', self.meta)
        with self.assertRaisesRegex(ValueError, 'fingerprints differ'):
            assert_evaluation_allowed(self.root, self.run)

    def test_snapshot_view_without_model_or_test_inputs(self):
        snapshot = self.root / 'reports/final_evaluation'
        snapshot.mkdir(parents=True)
        metrics = {'point': {'threshold': .1, 'recall': .85}}
        self.write(snapshot / 'test_metrics.json', metrics)
        digest = hashlib.sha256((snapshot / 'test_metrics.json').read_bytes()).hexdigest()
        self.write(snapshot / 'provenance.json', {'snapshot_sha256': {'test_metrics.json': digest}})
        self.write(self.shared / 'heldout_evaluation.json',
                   {'status': 'completed', 'selected_run': self.selection['selected_run'],
                    'test_metrics_sha256': digest})
        (self.run / 'selected_pipeline.joblib').unlink()
        (self.root / 'data/processed/test.csv').unlink()
        with patch('src.final_evaluation.run_selected_test', side_effect=AssertionError('Read only')):
            result = load_saved_results(self.root)
        self.assertEqual(result['test'], metrics)
        # Corrupted local results must not be silently replaced by the snapshot.
        self.write(self.run / 'test_metrics.json', {'changed': True})
        with self.assertRaisesRegex(ValueError, 'Saved test metrics changed'):
            load_saved_results(self.root)


if __name__ == '__main__':
    unittest.main()
