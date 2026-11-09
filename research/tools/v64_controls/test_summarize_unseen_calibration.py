"""In-memory regression checks for source-checkpoint provenance in reports."""
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.v64_controls import summarize_unseen_calibration as report


class SourceCheckpointTests(unittest.TestCase):
    def collect(self, selected, result_hash='a' * 64):
        metrics = {space: {mode: {'top1': 0.1, 'top5': 0.2}
                           for mode in ('single_trial', 'subject_aggregated')}
                   for space in ('clean', 'blur')}
        metrics['candidate_count'] = 200

        def read_json(path, issues):
            if path.name == 'selected_source.json':
                return selected
            if path.name in ('worker_status.json', 'pipeline_status.json'):
                return {'state': 'complete'}
            if path.name == 'result.json':
                pool, seed = path.parent.name.split('_seed')
                return dict(state='complete', pool=pool, seed=int(seed),
                            checkpoint={'sha256': result_hash},
                            tests={test: {'zero': metrics, 'calibrated': metrics}
                                   for test in report.POOLS[pool]},
                            best_epoch=1, trained_epochs=1, fit_images=10, fit_trials=10,
                            dev_images=2, trainable_parameters=67136,
                            train_subjects=['source'], development_selection='development only')
            return None

        with patch.object(report, 'read_json', side_effect=read_json):
            return report.collect(Path('/readonly_fixture'), [33])

    def test_current_and_legacy_source_schemas(self):
        for selected in ({'checkpoint_sha256': 'a' * 64}, {'sha256': 'a' * 64},
                         {'checkpoint_sha256': 'a' * 64, 'sha256': 'a' * 64}):
            with self.subTest(selected=selected):
                rows, _, issues = self.collect(selected)
                self.assertEqual(len(rows), 8)
                self.assertTrue(all(row['status'] == 'complete' for row in rows))
                self.assertEqual(issues, [])

    def test_wrong_result_hash_rejected(self):
        rows, _, issues = self.collect({'checkpoint_sha256': 'a' * 64}, 'b' * 64)
        self.assertTrue(all(row['status'] == 'invalid' for row in rows))
        self.assertTrue(all('differs' in issue['error'] for issue in issues))

    def test_missing_hash_rejected(self):
        for selected in (None, {}, {'checkpoint_sha256': None}, {'checkpoint_sha256': ''}):
            with self.subTest(selected=selected):
                rows, _, issues = self.collect(selected)
                self.assertTrue(all(row['status'] == 'invalid' for row in rows))
                self.assertTrue(all('Missing checkpoint hash' in issue['error'] for issue in issues))
        rows, _, _ = self.collect({'checkpoint_sha256': 'a' * 64}, None)
        self.assertTrue(all(row['status'] == 'invalid' for row in rows))

    def test_conflicting_source_hashes_rejected(self):
        rows, _, issues = self.collect({'checkpoint_sha256': 'a' * 64, 'sha256': 'b' * 64})
        self.assertTrue(all(row['status'] == 'invalid' for row in rows))
        self.assertTrue(all('Conflicting checkpoint hashes' in issue['error'] for issue in issues))


if __name__ == '__main__':
    unittest.main()
