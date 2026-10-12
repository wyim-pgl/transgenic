"""Local regressions over copies of captured job 6165521, never a tool rerun."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

BASE = Path(__file__).resolve().parents[1]
HELPER = Path(__file__).with_name('recover.py')
ORIGIN = '/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp_gffcompare_smoke_20260916/run'


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob('*') if p.is_file()}


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'copied_run'
        shutil.copytree(BASE / 'real_run', self.root)

    def run_analysis(self, origin=ORIGIN):
        return subprocess.run([sys.executable, '-I', '-B', str(HELPER), '--root', str(self.root),
                               '--origin', origin], capture_output=True, text=True)

    def test_actual_all_cases_read_only_and_exact_paths(self):
        before = snapshot(self.root)
        result = self.run_analysis()
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['status'], 'SEMANTIC_CHECKS_PASSED')
        self.assertEqual(len(report['observations']), 16)
        self.assertEqual(len(report['inventory']), 237)
        self.assertEqual(len([p for p in report['inventory'] if p.endswith(('.tmap', '.refmap'))]), 34)
        expected_counts = {
            'ordinary': [1, 0, 0], 'terminal_short': [1, 0, 0], 'terminal_extend': [1, 0, 0],
            'junction_change': [0, 1, 1], 'opposite_strand': [0, 1, 1], 'unmatched': [0, 1, 1],
            'mono_exact': [1, 0, 0], 'mono_utr_variant': [1, 0, 0], 'mono_cds_change': [0, 1, 1],
            'mono_opposite': [0, 1, 1], 'mono_no_cds': [0, 1, 1], 'empty_query': [0, 0, 1],
            'multiisoform_unique': [2, 0, 0], 'duplicate_query_terminal': None,
            'duplicate_reference_terminal': None, 'duplicate_mono_cds': None,
        }
        self.assertEqual({n: o['producer_counts'] for n, o in report['observations'].items()}, expected_counts)
        self.assertEqual(report['observations']['junction_change']['raw']['tmap_rows'][0]['class_code'], 'j')
        self.assertEqual(report['observations']['opposite_strand']['raw']['tmap_rows'][0]['class_code'], 's')
        self.assertEqual(report['observations']['unmatched']['raw']['tmap_rows'][0]['class_code'], 'u')
        self.assertEqual(report['observations']['empty_query']['raw']['tmap_rows'], [])
        self.assertEqual(report['observations']['empty_query']['raw']['exit'], 0)
        self.assertEqual(report['observations']['multiisoform_unique']['producer_counts'], [2, 0, 0])
        self.assertEqual([r['qry_id'] for r in report['observations']['duplicate_query_terminal']['raw']['tmap_rows']], ['q', 'q_terminal'])
        self.assertEqual(report['observations']['duplicate_reference_terminal']['producer_error']['error_type'], 'ValueError')
        self.assertIsNone(report['decision'])
        self.assertEqual(snapshot(self.root), before)

    def test_old_raw_glob_misses_actual_query_sibling_maps(self):
        # The deployed smoke.py:160 lookup; no substitute tmap is generated.
        self.assertEqual(list((self.root / 'ordinary/raw').glob('*.tmap')), [])
        self.assertTrue((self.root / 'ordinary/compare.predicted.gtf.tmap').is_file())
        result = self.run_analysis()
        self.assertEqual(result.returncode, 0, result.stderr)
        raw = json.loads(result.stdout)['observations']['ordinary']['raw']
        self.assertEqual(raw['tmap_path'], 'ordinary/compare.predicted.gtf.tmap')

    def test_missing_maps_rejected(self):
        for name in ('ordinary/compare.predicted.gtf.tmap', 'ordinary/compare.predicted.gtf.refmap',
                     'ordinary/producer/compare.predicted.gtf.tmap', 'ordinary/producer/compare.predicted.gtf.refmap'):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.unlink()
                result = self.run_analysis()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('inventory mismatch', result.stderr)
                path.write_bytes(original)

    def test_extra_map_rejected_no_wildcard_selection(self):
        for name in ('ordinary/raw/extra.tmap', 'ordinary/extra.refmap', 'ordinary/producer/extra.tmap'):
            with self.subTest(name=name):
                path = self.root / name
                shutil.copyfile(self.root / 'ordinary/compare.predicted.gtf.tmap', path)
                result = self.run_analysis()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('inventory mismatch', result.stderr)
                path.unlink()

    def test_mutated_maps_rejected(self):
        for name in ('ordinary/compare.predicted.gtf.tmap', 'ordinary/compare.predicted.gtf.refmap',
                     'ordinary/producer/compare.predicted.gtf.tmap', 'ordinary/producer/compare.predicted.gtf.refmap'):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original.replace(b'g\tr', b'h\tr', 1))  # same length
                result = self.run_analysis()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('raw hash mismatch', result.stderr)
                path.write_bytes(original)

    def test_relocated_map_is_not_searched_for(self):
        (self.root / 'ordinary/compare.predicted.gtf.tmap').rename(self.root / 'ordinary/raw/compare.predicted.gtf.tmap')
        result = self.run_analysis()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('inventory mismatch', result.stderr)

    def test_rewritten_manifest_cannot_authorize_mutation(self):
        path = self.root / 'ordinary/compare.predicted.gtf.tmap'
        path.write_bytes(path.read_bytes().replace(b'g\tr', b'h\tr', 1))
        manifest = self.root / 'hashes.json'
        hashes = json.loads(manifest.read_text())
        hashes['ordinary/compare.predicted.gtf.tmap'] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(hashes))
        result = self.run_analysis()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('remote manifest pin mismatch', result.stderr)

    def test_wrong_or_relative_origin_rejected(self):
        for origin in ('run', '/arbitrary/old/run', str(self.root), ORIGIN + '/..'):
            with self.subTest(origin=origin):
                result = self.run_analysis(origin)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('logged absolute origin mismatch', result.stderr)

    def test_changed_result_rejected_not_replaced(self):
        (self.root / 'ordinary/result.json').write_text('{"counts": {"f1_tp": 0}}')
        result = self.run_analysis()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('raw hash mismatch', result.stderr)

    def test_symlink_map_rejected_even_with_same_bytes(self):
        path = self.root / 'ordinary/compare.predicted.gtf.tmap'
        path.unlink()
        path.symlink_to(BASE / 'real_run/ordinary/compare.predicted.gtf.tmap')
        result = self.run_analysis()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('symlink', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
