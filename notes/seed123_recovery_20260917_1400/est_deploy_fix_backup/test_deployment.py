"""Synthetic closure only: no real EST data, samtools or cluster access."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).parent
NAMES = ('est_real_sample', 'est_hsp_contract', 'est_hsp_expanded', 'est_phase_accept', 'est_phase_project')


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WORKER = '''import hashlib, json
from pathlib import Path
def main(argv):
    args = dict(zip(argv[::2], argv[1::2]))
    out = Path(args['--out']); out.mkdir()
    (out / 'diagnostics.jsonl').write_bytes(b'fixture only\\n')
    report = dict(status='COMPLETE', diagnostics_sha256=hashlib.sha256((out / 'diagnostics.jsonl').read_bytes()).hexdigest(), manifest_file={'sha256':args['--sha']}, training_eligible=False, full_c0=False, a11_status='unassessable', route_a_complete=False, molecule_counts_complete=False, raster=False)
    (out / 'report.json').write_text(json.dumps(report))
    (out / 'COMPLETE').write_text(hashlib.sha256((out / 'report.json').read_bytes()).hexdigest()+'\\n')
    EXTRA
    return EXIT
'''


class Deployment(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in list(sys.modules):
            if name == 'evidence' or name.startswith('evidence.'):
                del sys.modules[name]
        self.addCleanup(self.clean_modules)

    def clean_modules(self):
        for name in list(sys.modules):
            if name == 'evidence' or name.startswith('evidence.'):
                del sys.modules[name]

    def build(self, extra='pass', exitcode=0):
        self.assertTrue((HERE / 'build_release.py').exists(), 'builder is not implemented')
        self.builder = load('build_release')
        self.runner = load('run_compute')
        source = self.root / 'source' / 'revision/scripts/evidence'
        source.mkdir(parents=True)
        pins = {}
        for name in NAMES:
            path = source / (name + '.py')
            data = WORKER.replace('EXTRA', extra).replace('EXIT', str(exitcode)) if name == NAMES[0] else '# fixture\n'
            path.write_text(data)
            pins[str(path.relative_to(self.root / 'source'))] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.manifest = self.root / 'manifest.json'
        self.manifest.write_text('{"species":"Sitalica"}\n')
        self.release = self.root / 'release'
        self.sha = self.builder.build(self.root / 'source', self.release, pins, self.manifest,
                                      hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        return self.release

    def run_fixture(self):
        # Only runtime identity checks are substituted: closure/inventory/import/
        # report verification and worker CLI run for real on tiny fixture files.
        with patch.object(self.runner, 'runtime_identity', return_value={'fixture': True}):
            return self.runner.run(self.release, self.sha, self.root / 'output')

    def test_build_inventory_and_no_overwrite(self):
        self.build()
        self.assertEqual((self.release / 'code/evidence/__init__.py').read_bytes(), b'')
        self.assertEqual((self.release / 'manifest.json').read_bytes(), self.manifest.read_bytes())
        self.runner.verify_release(self.release, self.sha)
        with self.assertRaises(FileExistsError):
            self.builder.build(self.root / 'source', self.release, {}, self.manifest, '0'*64)

    def test_wrong_ledger(self):
        self.build()
        with self.assertRaises(ValueError):
            self.runner.verify_release(self.release, '0'*64)

    def test_tamper_extra_missing_and_symlink(self):
        for mode in ('tamper', 'extra', 'missing', 'symlink'):
            with self.subTest(mode=mode):
                if hasattr(self, 'release'):
                    self.tearDown(); self.clean_modules(); self.temp.cleanup(); self.setUp()
                self.build()
                target = self.release / 'code/evidence/est_hsp_contract.py'
                if mode == 'tamper': target.write_text('# changed')
                if mode == 'extra': (self.release / 'extra').write_text('x')
                if mode == 'missing': (self.release / 'code/evidence/__init__.py').unlink()
                if mode == 'symlink':
                    target.unlink(); target.symlink_to(self.manifest)
                with self.assertRaises(ValueError): self.runner.verify_release(self.release, self.sha)

    def test_shadow_preloaded_package_rejected(self):
        self.build()
        import types
        sys.modules['evidence'] = types.ModuleType('evidence')
        self.assertNotEqual(self.run_fixture(), 0)
        self.assertFalse((self.root / 'output/control/verification.json').exists())

    def test_complete_receipt_and_output_collision(self):
        self.build()
        self.assertEqual(self.run_fixture(), 0)
        receipt = json.loads((self.root / 'output/control/verification.json').read_text())
        self.assertEqual(receipt['status'], 'VERIFIED_DIAGNOSTIC_COMPLETE')
        self.assertFalse(receipt['biological_acceptance'])
        with self.assertRaises(FileExistsError): self.run_fixture()

    def test_postcheck_blocks_publication(self):
        self.build("Path(__file__).write_text('# changed after import')")
        self.assertNotEqual(self.run_fixture(), 0)
        self.assertTrue((self.root / 'output/sample/COMPLETE').exists())
        self.assertFalse((self.root / 'output/control/verification.json').exists())
        result = json.loads((self.root / 'output/control/result.json').read_text())
        self.assertTrue(result['postcheck_attempted'])

    def test_nonzero_preserves_worker_output(self):
        self.build(exitcode=7)
        self.assertEqual(self.run_fixture(), 7)
        self.assertTrue((self.root / 'output/sample/COMPLETE').exists())
        self.assertFalse((self.root / 'output/control/verification.json').exists())

    def test_missing_marker_refuses_receipt(self):
        self.build("(out / 'COMPLETE').unlink()")
        self.assertNotEqual(self.run_fixture(), 0)
        self.assertFalse((self.root / 'output/control/verification.json').exists())

    def test_tampered_diagnostics_refuse_receipt(self):
        self.build("(out / 'diagnostics.jsonl').write_bytes(b'tampered')")
        self.assertNotEqual(self.run_fixture(), 0)
        self.assertFalse((self.root / 'output/control/verification.json').exists())

    def test_exception_still_postchecks(self):
        self.build("raise RuntimeError('fixture failure')")
        self.assertNotEqual(self.run_fixture(), 0)
        result = json.loads((self.root / 'output/control/result.json').read_text())
        self.assertTrue(result['postcheck_attempted'])
        self.assertIn('RuntimeError: fixture failure', result['errors'])
        self.assertTrue((self.root / 'output/sample/report.json').exists())

    def test_working_directory_shadow_does_not_load(self):
        self.build()
        shadow = self.root / 'shadow/evidence'
        shadow.mkdir(parents=True)
        (shadow / '__init__.py').write_text("raise RuntimeError('shadow imported')")
        sys.path.insert(0, str(shadow.parent))
        try:
            self.assertEqual(self.run_fixture(), 0)
        finally:
            sys.path.remove(str(shadow.parent))

    def test_runtime_failure_prevents_worker(self):
        self.build()
        with patch.object(self.runner, 'runtime_identity', side_effect=ValueError('bad runtime')) as check:
            self.assertNotEqual(self.runner.run(self.release, self.sha, self.root / 'output'), 0)
        self.assertEqual(check.call_count, 2)
        self.assertFalse((self.root / 'output/sample').exists())

    def test_runtime_postcheck_blocks_receipt(self):
        self.build()
        with patch.object(self.runner, 'runtime_identity', side_effect=[{'fixture': True}, ValueError('changed runtime')]):
            self.assertNotEqual(self.runner.run(self.release, self.sha, self.root / 'output'), 0)
        self.assertTrue((self.root / 'output/sample/COMPLETE').exists())
        self.assertFalse((self.root / 'output/control/verification.json').exists())

    def test_source_pin_wrong_and_source_link_refused(self):
        self.build()
        pins = json.loads((self.release / 'ledger.json').read_text())['source_hashes']
        name = next(iter(pins))
        pins[name] = '0' * 64
        with self.assertRaises(ValueError):
            self.builder.build(self.root / 'source', self.root / 'wrong-pin', pins, self.manifest,
                               hashlib.sha256(self.manifest.read_bytes()).hexdigest())
        path = self.root / 'source' / name
        data = path.read_bytes()
        path.unlink(); path.symlink_to(self.manifest)
        pins[name] = hashlib.sha256(data).hexdigest()
        with self.assertRaises(ValueError):
            self.builder.build(self.root / 'source', self.root / 'linked-source', pins, self.manifest,
                               hashlib.sha256(self.manifest.read_bytes()).hexdigest())

    def test_environment_context_does_not_capture_secrets(self):
        self.build()
        with patch.dict('os.environ', {'SECRET_TOKEN': 'must-not-retain'}):
            self.assertEqual(self.run_fixture(), 0)
        result = (self.root / 'output/control/result.json').read_text()
        self.assertNotIn('must-not-retain', result)

    def test_builder_rejects_bad_source_pin(self):
        self.build()
        with self.assertRaises(ValueError):
            self.builder.build(self.root / 'source', self.root / 'bad-release', {}, self.manifest, '0'*64)
        self.assertFalse((self.root / 'bad-release').exists())


if __name__ == '__main__':
    unittest.main()
