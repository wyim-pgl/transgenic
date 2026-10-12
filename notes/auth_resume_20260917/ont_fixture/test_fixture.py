"""Local preparation checks only. Never counts as real-tool F4 evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
import os

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixture', HERE/'fixture.py')
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)

class FixtureTests(unittest.TestCase):
    def test_deterministic_inputs_and_fixed_oracle(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            a, b = Path(tmp)/'a', Path(tmp)/'b'
            f.generate(a); f.generate(b)
            self.assertEqual(f.inventory(a), f.inventory(b))
            expected = json.loads((a/'expected.json').read_text())
            self.assertEqual(expected['rows'][0]['n_ONT'], 3)
            self.assertTrue(expected['rows'][0]['supported'])
            self.assertEqual(expected['rows'][1]['n_ONT'], 2)
            self.assertFalse(expected['rows'][1]['supported'])
            self.assertLess(sum(p.stat().st_size for p in a.rglob('*') if p.is_file()), 150000)
            with self.assertRaises(FileExistsError):
                f.generate(a)

    def test_wrong_junctions_rejected(self):
        with self.assertRaises(ValueError):
            f.require_rows([], [{'n_ONT': 3}])

    def test_closure_and_pins(self):
        release = HERE/'package'
        f.verify_release(release)

    def test_real_frozen_cli_preflight_and_panel_seam(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)/'synthetic'; f.generate(root)
            argv = [sys.executable, '-I', '-B', str(HERE/'fixture.py'), 'entry', str(HERE/'package'), str(root), 'driver', '--',
                '--root', str(root), '--run', f.RUN, '--reads', str(root/'evidence'/f.DATASET/(f.RUN+'.fa.gz')),
                '--ref', str(root/'genomes/fixture.fa'), '--gff', str(root/'genomes/fixture.gff3'),
                '--panels', str(root/'panel.tsv'), '--ref-md5', f.digest(root/'genomes/fixture.fa', 'md5'),
                '--gff-md5', f.digest(root/'genomes/fixture.gff3', 'md5'), '--threads', '4']
            result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report['status'], 'PREPARATORY_BYTE_CHECKS_ONLY')
            self.assertFalse(report['scientific_input_validation'])
            self.assertFalse((root/'evidence/mvp_scoring_20260916').exists())
            # Original frozen default still rejects our synthetic panel in a fresh process.
            raw = [sys.executable, '-I', '-B', str(HERE/'package/frozen/revision/scripts/mvp/mvp_align_panel_longreads.py')]+argv[9:]
            rejected = subprocess.run(raw, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn('panel TSV hash mismatch', rejected.stderr)

    def test_stub_pipeline_and_real_postflight_NOT_real_tool_evidence(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            control = Path(tmp)
            root = control/'synthetic'; f.generate(root)
            binary = control/'bin'; binary.mkdir()
            for name in ('minimap2', 'samtools', 'seqkit'):
                path = binary/name
                path.write_text('#!'+sys.executable+'\n'+(HERE/'stub_tool.py').read_text())
                path.chmod(0o700)
            argv = [sys.executable, '-I', '-B', str(HERE/'fixture.py'), 'entry', str(HERE/'package'), str(root), 'driver', '--',
                '--root', str(root), '--run', f.RUN, '--reads', str(root/'evidence'/f.DATASET/(f.RUN+'.fa.gz')),
                '--ref', str(root/'genomes/fixture.fa'), '--gff', str(root/'genomes/fixture.gff3'),
                '--panels', str(root/'panel.tsv'), '--ref-md5', f.digest(root/'genomes/fixture.fa', 'md5'),
                '--gff-md5', f.digest(root/'genomes/fixture.gff3', 'md5'), '--threads', '4', '--execute']
            env = dict(os.environ, PATH=str(binary), PYTHONDONTWRITEBYTECODE='1')
            wrapper = f.verify_release(HERE/'package')
            receipt = {}
            wrapper.child(argv, control, env, 20, 1, receipt)
            self.assertEqual(receipt['exit_code'], 0, (control/'driver.stderr').read_text())
            gate = f.module(HERE/'package/gate.py', 'stub_gate')
            scored = root/'evidence/mvp_scoring_20260916'/f.RUN
            receipt['outputs'] = gate.inventory(scored)
            result = f.postflight(HERE/'package', root, receipt)
            self.assertEqual(result['fixture_oracle']['rows'][0]['n_ONT'], 3)
            # Mutated stdout cannot be accepted merely because exit was zero.
            (control/'driver.stdout').write_text('{}')
            with self.assertRaises(ValueError):
                f.postflight(HERE/'package', root, receipt)

    def test_junction_cli_help_bootstrap(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            root = Path(tmp)/'synthetic'; f.generate(root)
            result = subprocess.run([sys.executable, '-I', '-B', str(HERE/'fixture.py'), 'entry',
                str(HERE/'package'), str(root), 'junction', '--', '--help'], capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('--alignment', result.stdout)

if __name__ == '__main__':
    unittest.main()
