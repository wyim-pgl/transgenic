import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest

HERE = Path(__file__).resolve().parent

class Boundary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('wrapper', HERE/'wrapper.py')
        cls.w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.w)

    def test_bad_pin(self):
        with self.assertRaisesRegex(ValueError, 'pin'):
            self.w.pinned(HERE/'test_wrapper.py', '0'*64)

    def test_no_reuse(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                self.w.fresh(Path(d))

    def test_receipt_scope_and_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'stdout'; p.write_text('{}')
            r = dict(scope='full_alignment_descriptive', structural_status='UNASSESSABLE',
                     training_eligible=False, decision=None, exit_code=0,
                     stdout=self.w.identity(p), stderr=self.w.identity(p), outputs={})
            self.w.check_boundary(r)
            r['scope']='orientation_sample'
            with self.assertRaises(ValueError): self.w.check_boundary(r)
            r['scope']='full_alignment_descriptive'; p.write_text('changed')
            with self.assertRaises(ValueError): self.w.check_boundary(r)

    def test_timeout_retains_logs_and_status(self):
        with tempfile.TemporaryDirectory() as d:
            r={}
            with self.assertRaises(TimeoutError):
                self.w.child([sys.executable,'-c','import time; print("started",flush=True); time.sleep(20)'],
                             Path(d),dict(os.environ),0.1,0.1,r)
            self.assertIsNotNone(r['exit_code'])
            self.assertEqual(r['status'],'TIMED_OUT')
            self.assertTrue((Path(d)/'driver.stderr').exists())
            self.assertIn('started',(Path(d)/'driver.stdout').read_text())

    def test_mutated_recursive_output_rejected_by_existing_gate(self):
        path = HERE.parents[1]/'wait_work_20260916/longread/gate.py'
        self.w.pinned(path, self.w.GATE_SHA)
        spec = importlib.util.spec_from_file_location('inventory_gate', path)
        gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
        with tempfile.TemporaryDirectory() as d:
            out = Path(d); (out/'orientation').mkdir()
            p = out/'orientation/audit.json'; p.write_text('{}')
            expected = gate.inventory(out)
            gate.verify_inventory(out, expected)
            p.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'inventory'):
                gate.verify_inventory(out, expected)

    def test_nonzero_child_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            r=dict(scope='full_alignment_descriptive', structural_status='UNASSESSABLE',
                   training_eligible=False, decision=None)
            self.w.child([sys.executable,'-c','raise SystemExit(7)'],Path(d),dict(os.environ),3,0.1,r)
            with self.assertRaisesRegex(ValueError, 'nonzero'):
                self.w.check_boundary(r)
            self.assertFalse((Path(d)/'COMPLETE').exists())

    def test_slurm_environment_alone_insufficient(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                self.w.slurm({'budget':dict(cpus=4,memory_mib=1024,wall_seconds=60,cleanup_seconds=2)},Path(d))

    def test_success_real_child(self):
        with tempfile.TemporaryDirectory() as d:
            r={}
            self.w.child([sys.executable,'-c','print("{}")'],Path(d),dict(os.environ),3,0.1,r)
            self.assertEqual(r['exit_code'],0)
            self.assertEqual(json.loads((Path(d)/'driver.stdout').read_text()),{})


class SyntheticExecute(unittest.TestCase):
    """Real wrapper orchestration, synthetic scheduler/data/science/workload only.

    Based on the reviewed harness, without altering its vulnerability proofs.
    The copied gate and nine frozen files retain their original hashes.
    """
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ont_fixes_')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.code = self.base/'code/wrapper.py'
        self.code.parent.mkdir()
        shutil.copyfile(HERE/'wrapper.py', self.code)
        spec = importlib.util.spec_from_file_location('synthetic_wrapper', self.code)
        self.w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.w)
        old = HERE.parents[1]/'wait_work_20260916/longread'
        self.release = self.base/'release'; self.release.mkdir()
        for name in ('gate.py', 'source_pins.json'):
            shutil.copyfile(old/name, self.release/name)
        self.pins = json.loads((old/'source_pins.json').read_text())
        self.source = self.release/'frozen'
        for name in self.pins['files']:
            dst = self.source/name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old/'frozen'/name, dst)
        self.control = self.base/'control'
        self.root = self.base/'data'; self.root.mkdir()
        self.w.ROOT = self.root
        self.run = 'SRR28341943'
        self.reads = self.root/'evidence/ont/Athaliana_FLIC_PRJNA1087576'/f'{self.run}.fa.gz'
        self.roles = self.root/'evidence/DATASET_ROLES.longread_v1.tsv'
        for p in (self.reads, self.roles, self.root/'genomes/Athaliana_167_TAIR10.fa',
                  self.root/'genomes/Athaliana_167_TAIR10.gene.clean.gff3', self.release/'authority.tsv'):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text('synthetic\n')
        binary = self.base/'bin'; binary.mkdir()
        tools = {}
        for name in ('minimap2', 'samtools', 'seqkit', 'scontrol'):
            p = binary/name; p.write_text('#!/bin/sh\nexit 0\n'); p.chmod(0o700)
            tools[name] = dict(path=str(p), sha256=self.sha(p))
        self.python = str(Path(sys.executable).resolve())
        self.request = dict(scope=self.w.SCOPE, runtime_reviewed=True, release=str(self.release),
                            control=str(self.control), run=self.run,
                            budget=dict(cpus=4, memory_mib=1024, wall_seconds=8, cleanup_seconds=1),
                            scontrol=tools.pop('scontrol'), tools=tools)
        tools['python'] = dict(path=self.python, sha256=self.sha(self.python))
        self.reqpath = self.base/'request.json'
        self.out = self.root/'output'
        self.imported = False

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def invoke(self, mode='success'):
        self.reqpath.write_text(json.dumps(self.request))
        original_spec = importlib.util.spec_from_file_location
        original_child = self.w.child
        driver = SimpleNamespace(output_directory=lambda root, run: self.out,
                                 w=SimpleNamespace(load_panel=lambda p: list(range(50))))
        def spec(name, path, *a, **kw):
            s = original_spec(name, path, *a, **kw)
            if name == 'ont_pinned_gate':
                self.imported = True
                real_exec = s.loader.exec_module
                def exec_module(g):
                    real_exec(g)
                    if mode != 'real_driver':
                        g.load_driver = lambda source: driver
                    g.read_binding = lambda *a: {'roles': {'path': str(self.roles)}}
                    g.reference_binding = lambda *a: {'files': {'fasta': {'md5': '0'*32}, 'gff': {'md5': '0'*32}}}
                    def postflight(args, receipt):
                        if mode in ('stderr_mutation', 'stdout_mutation'):
                            (self.control/('driver.'+mode.split('_')[0])).write_text('mutated')
                        if mode == 'input_mutation': self.reads.write_text('mutated')
                        if mode == 'output_mutation': (self.out/'tiny.txt').write_text('mutated')
                        if mode == 'closure_mutation': (self.source/'revision/scripts/json.py').write_text('# added')
                        return {'synthetic_science_substitution': True}
                    g.verify_alignment = postflight
                s.loader.exec_module = exec_module
            return s
        def child(argv, control, env, seconds, grace, receipt):
            receipt['requested_driver_argv'] = argv
            code = f'from pathlib import Path; print("{{}}",flush=True); Path({str(self.out)!r}).mkdir(); Path({str(self.out/"tiny.txt")!r}).write_text("tiny"); '
            if mode == 'nonzero': code += 'raise SystemExit(7)'
            elif mode == 'timeout': code += 'import time; time.sleep(20)'
            elif mode == 'nested_shadow':
                sample = self.source/'revision/scripts/68_sample_fasta.py'
                code += f'import subprocess,sys; subprocess.run([sys.executable,{str(sample)!r},"--help"],check=True)'
            original_child([self.python, '-I', '-B', '-c', code], control, env,
                           .1 if mode == 'timeout' else seconds, .1, receipt)
        with patch.object(self.w, 'slurm', return_value={'synthetic_scheduler_substitution': True}), \
             patch.object(self.w.importlib.util, 'spec_from_file_location', side_effect=spec), \
             patch.object(self.w, 'child', side_effect=child):
            return self.w.execute(str(self.reqpath), self.sha(self.reqpath), self.sha(self.code))

    def assert_incomplete(self):
        self.assertFalse((self.control/'COMPLETE').exists())
        self.assertTrue((self.control/'FAILED.json').exists())
        self.assertEqual(json.loads((self.control/'receipt.json').read_text())['status'], 'INCOMPLETE')

    def test_execute_synthetic_happy_path(self):
        r = self.invoke()
        self.assertEqual(r['status'], 'VERIFIED_DESCRIPTIVE')
        self.assertEqual((self.control/'COMPLETE').read_text(), self.sha(self.control/'receipt.json')+'\n')
        self.assertEqual(r['scope'], 'full_alignment_descriptive')
        self.assertIsNone(r['decision']); self.assertFalse(r['training_eligible'])
        self.assertEqual(r['structural_status'], 'UNASSESSABLE')
        for stream in ('stdout', 'stderr'):
            self.assertEqual(r[stream], r['diagnostics']['driver.'+stream])
        self.assertEqual(r['requested_driver_argv'][:4], [self.python, '-I', '-B', str(self.source/'revision/scripts/mvp/mvp_align_panel_longreads.py')])

    def test_execute_failure_and_postflight_mutations(self):
        for mode in ('nonzero', 'timeout', 'input_mutation', 'output_mutation',
                     'stdout_mutation', 'stderr_mutation', 'closure_mutation'):
            with self.subTest(mode=mode):
                case = SyntheticExecute(); case.setUp()
                try:
                    with self.assertRaises((ValueError, TimeoutError)): case.invoke(mode)
                    case.assert_incomplete()
                finally: case.doCleanups()

    def test_unlisted_imports_rejected_before_gate_import_or_marker(self):
        for name, mode in (('src/transgenic.py', 'real_driver'),
                           ('revision/scripts/json.py', 'nested_shadow')):
            with self.subTest(name=name):
                case = SyntheticExecute(); case.setUp()
                try:
                    marker = case.base/'executed'
                    (case.source/name).write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\n')
                    error = None
                    try: case.invoke(mode)
                    except Exception as caught: error = caught
                    self.assertFalse(marker.exists(), 'unlisted code executed before rejection')
                    self.assertFalse(case.imported, 'gate import preceded closure validation')
                    self.assertIsInstance(error, ValueError)
                    self.assertIn('closure', str(error))
                    case.assert_incomplete()
                finally: case.doCleanups()

    def test_closure_rejects_extras_missing_files_and_redirects(self):
        for kind in ('pyc', 'init', 'directory', 'symlink', 'directory_symlink', 'fifo', 'missing'):
            with self.subTest(kind=kind):
                case = SyntheticExecute(); case.setUp()
                try:
                    extra = case.source/'src/extra'
                    if kind == 'pyc': extra.with_suffix('.pyc').write_bytes(b'fake')
                    elif kind == 'init': (case.source/'src/transgenic/__init__.py').write_text('')
                    elif kind == 'directory': extra.mkdir()
                    elif kind == 'symlink': extra.symlink_to(case.source/'revision/protocols/b6_panels_v1.tsv')
                    elif kind == 'directory_symlink':
                        p = case.source/'src/transgenic/evidence'
                        p.rename(case.base/'evidence'); p.symlink_to(case.base/'evidence', target_is_directory=True)
                    elif kind == 'fifo': os.mkfifo(extra)
                    else: (case.source/'revision/scripts/68_sample_fasta.py').unlink()
                    with self.assertRaises((ValueError, FileNotFoundError)): case.invoke()
                    self.assertFalse(case.imported)
                    case.assert_incomplete()
                finally: case.doCleanups()

    def test_unsupported_interpreter_rejected_before_any_path_access(self):
        with patch.object(self.w.sys, 'version_info', (3, 8, 20)):
            with self.assertRaisesRegex(ValueError, r'Python >=3\.9 required'):
                self.w.execute('/nonexistent/request.json', '0'*64, '0'*64)
        self.assertFalse(self.control.exists())

    def test_actual_nested_helper_import_layout_clean_environment(self):
        # Actual frozen helpers, their unchanged non-isolated invocations, no workload.
        env = dict(PATH=str(self.base/'bin'), HOME=str(self.base), TMPDIR=str(self.base),
                   LANG='C', LC_ALL='C', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
        for name in ('68_sample_fasta.py', '67_orientation_audit.py', 'mvp/mvp_junction_table.py'):
            script = self.source/'revision/scripts'/name
            code = f'import subprocess,sys; raise SystemExit(subprocess.run([sys.executable,{str(script)!r},"--help"]).returncode)'
            result = subprocess.run([self.python, '-I', '-B', '-c', code], env=env,
                                    cwd=self.base, capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('usage:', result.stdout)


if __name__=='__main__': unittest.main()

