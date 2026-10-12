"""Local-only candidate adaptation checks. No GPU or remote operations."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import build_release as b
import deploy as d

HERE = Path(__file__).resolve().parent

class CandidateTests(unittest.TestCase):
    def test_candidate_manifest_exact_closure(self):
        self.assertTrue(hasattr(b, 'candidate_payload'), 'missing candidate validator')
        manifest, payload = b.candidate_payload(HERE.parent / 'candidate')
        self.assertEqual(len(payload), 103)
        self.assertEqual(set(payload), set(manifest['source_files']))
        self.assertIn('src/transgenic/training/mvp_profiles.py', payload)
        self.assertNotIn('commit', manifest)

    def test_wrong_archive_pin_rejected(self):
        self.assertTrue(hasattr(b, 'candidate_payload'), 'missing candidate validator')
        with patch.object(d, 'CANDIDATE_ARCHIVE_SHA', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                b.candidate_payload(HERE.parent / 'candidate')

    def test_wrong_manifest_pin_rejected(self):
        self.assertTrue(hasattr(b, 'candidate_payload'), 'missing candidate validator')
        with patch.object(d, 'CANDIDATE_MANIFEST_SHA', '0' * 64):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                b.candidate_payload(HERE.parent / 'candidate')

    def test_no_overwrite(self):
        self.assertTrue(hasattr(b, 'build'), 'missing exclusive builder')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'release'; out.mkdir()
            marker = out / 'sentinel'; marker.write_text('retain')
            with self.assertRaises(FileExistsError):
                b.build(HERE.parent / 'candidate', out)
            self.assertEqual(marker.read_text(), 'retain')

    def test_prepare_profile_seed(self):
        tree = ast.parse((HERE / 'deploy.py').read_text())
        stage = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'container_stage')
        argv = next(n for n in ast.walk(stage) if isinstance(n, ast.List) and
                    any(isinstance(x, ast.Constant) and x.value == 'prepare' for x in n.elts))
        values = [n.value if isinstance(n, ast.Constant) else None for n in argv.elts]
        self.assertIn('--profile', values)
        self.assertEqual(values[values.index('--profile') + 1], 'development-single')
        self.assertEqual(values[values.index('--seed') + 1], 123)

    def test_old_prepared_receipt_rejected(self):
        self.assertTrue(hasattr(d, 'validate_prepared'), 'missing prepared profile guard')
        bindings = {'profile': 'development-single', 'seed': 123, 'run_role': 'mvp_finetune'}
        good = {'bindings': bindings, 'bindings_sha256': d.binding_digest(bindings)}
        d.validate_prepared(good)
        for key, value in (('profile', 'legacy-three-seed'), ('seed', 1), ('seed', '123'),
                           ('run_role', 'b5_baseline')):
            bad = copy.deepcopy(good); bad['bindings'][key] = value
            bad['bindings_sha256'] = d.binding_digest(bad['bindings'])
            with self.assertRaises(ValueError): d.validate_prepared(bad)
        bad = copy.deepcopy(good); bad['bindings_sha256'] = '0' * 64
        with self.assertRaises(ValueError): d.validate_prepared(bad)

    def test_controller_gate_rejects_unreviewed_stale_overbudget_old_receipts(self):
        self.assertTrue(hasattr(d, 'validate_controller_gate'), 'missing conditional execution gate')
        gate = dict(scope='S0_P1_DIAGNOSTIC_ONLY', profile='development-single', seed=123,
                    release_json_sha256='a' * 64, measured_at_unix=1000,
                    full_accounting_reviewed=True, no_overlap=True, new_package_independent_review=True,
                    old_owned_pending_zero_confirmed_cancel=True, staging_ownership_mapping_reviewed=True,
                    b5_exclusions_reviewed=True, prior_diagnostic_node_hours=0,
                    other_pending_reserved_node_hours=0, training_authorized=False,
                    accounting_record_reference='controller:accounting', review_record_reference='controller:review')
        # Legacy attestation is never an admissible durable authorization.
        with self.assertRaises(ValueError): d.validate_controller_gate(gate, 'a' * 64, now=1001)
        for key, value in (('profile', 'historical-three-seed'), ('seed', 1),
            ('new_package_independent_review', False), ('no_overlap', False),
            ('old_owned_pending_zero_confirmed_cancel', False), ('measured_at_unix', 0),
            ('prior_diagnostic_node_hours', 4.6), ('other_pending_reserved_node_hours', 5),
            ('prior_diagnostic_node_hours', float('nan')), ('training_authorized', True),
            ('release_json_sha256', 'b' * 64)):
            bad = dict(gate); bad[key] = value
            with self.assertRaises(ValueError): d.validate_controller_gate(bad, 'a' * 64, now=1001)

    def test_new_namespace_and_preserved_exclusions(self):
        text = (HERE / 'p1.sbatch').read_text()
        self.assertIn('staging/dev123_p1_release_v5_20260917', text)
        self.assertIn('#SBATCH --exclude=gh062,gh093\n', text)
        self.assertIn('--kill-after=20s 1700s', text)
        self.assertIn('#SBATCH --time=00:30:00', text)
        self.assertIn("'dev123-p1-v5-job-'", (HERE / 'deploy.py').read_text())

if __name__ == '__main__': unittest.main()
