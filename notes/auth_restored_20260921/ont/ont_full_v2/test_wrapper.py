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


def _old_release():
    """Locate the historical longread release regardless of this file's depth."""
    for base in HERE.parents:
        candidate = base/'wait_work_20260916/longread'
        if (candidate/'gate.py').is_file():
            return candidate
    raise FileNotFoundError('historical longread release not found above '+str(HERE))


OLD_RELEASE = _old_release()

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
        path = OLD_RELEASE/'gate.py'
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
        old = OLD_RELEASE
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
        # Declared synthetic seam: a fake cgroup v1 memory controller, since a test
        # host has no Slurm job cgroup. Real values only ever come from a real allocation.
        self.cgroup_root = self.base/'cgroup/memory'
        leaf = self.cgroup_root/'slurm/job_synthetic'; leaf.mkdir(parents=True)
        for name, value in (('memory.limit_in_bytes', '1073741824'),
                            ('memory.memsw.limit_in_bytes', '1073741824'),
                            ('memory.use_hierarchy', '1'), ('memory.failcnt', '0'),
                            ('memory.memsw.failcnt', '0'),
                            ('memory.oom_control', 'oom_kill_disable 0\nunder_oom 0\noom_kill 0')):
            (leaf/name).write_text(value+'\n')
        self.proc_cgroup = self.base/'proc_cgroup'
        self.proc_cgroup.write_text('9:memory:/slurm/job_synthetic\n')
        binary = self.base/'bin'; binary.mkdir()
        tools = {}
        for name in ('minimap2', 'samtools', 'seqkit', 'scontrol'):
            p = binary/name; p.write_text('#!/bin/sh\nexit 0\n'); p.chmod(0o700)
            tools[name] = dict(path=str(p), sha256=self.sha(p))
        self.python = str(Path(sys.executable).resolve())
        self.request = dict(scope=self.w.SCOPE, runtime_reviewed=True, release=str(self.release),
                            control=str(self.control), run=self.run,
                            budget=dict(cpus=4, memory_mib=1024, wall_seconds=8, cleanup_seconds=1,
                                        output_mib=4, scratch_mib=64),
                            cgroup=dict(required=True, root=str(self.cgroup_root)),
                            seams=['synthetic scheduler/data/science/workload substitutions'],
                            substitutions=None,
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
             patch.object(self.w, 'PROC_CGROUP', self.proc_cgroup), \
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


class RuntimeBoundary(unittest.TestCase):
    """Allocation-side observations the real-tool job 6165965 could not supply.

    Job 6165965 exercised only child/identity/save. slurm(), the affinity
    list, the output-size ceiling, the cgroup view and the scratch ceiling had
    no coverage from that run at all.
    """
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='ont_runtime_')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        spec = importlib.util.spec_from_file_location('runtime_wrapper', HERE/'wrapper.py')
        self.w = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.w)
        self.control = self.base/'control'; self.control.mkdir()
        self.cpus = len(os.sched_getaffinity(0))
        self.budget = dict(cpus=self.cpus, memory_mib=1024, wall_seconds=60, cleanup_seconds=5,
                           output_mib=4, scratch_mib=64)

    def scontrol(self, fields):
        path = self.base/'scontrol'
        path.write_text('#!/bin/sh\ncat <<\'EOF\'\n'+fields+'\nEOF\n')
        path.chmod(0o700)
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def request(self, **over):
        host = __import__('socket').gethostname()
        fields = ('JobId=4242 JobState=RUNNING UserId=u({uid}) NumNodes=1 NumCPUs={cpus} NumTasks=1 '
                  'NodeList={host} AllocTRES=cpu={cpus},mem=1024M RunTime=00:00:10 TimeLimit=00:10:00'
                  ).format(uid=os.getuid(), cpus=self.budget['cpus'], host=host)
        req = dict(budget=dict(self.budget), scontrol=self.scontrol(fields))
        req.update(over)
        return req

    def test_slurm_records_the_exact_allocated_affinity_list(self):
        req = self.request()
        env = dict(SLURM_JOB_ID='4242', SLURM_CPUS_PER_TASK=str(self.budget['cpus']))
        with patch.dict(os.environ, env, clear=False):
            record = self.w.slurm(req, self.control)
        self.assertEqual(record['affinity'], sorted(os.sched_getaffinity(0)))
        self.assertTrue(all(type(cpu) is int for cpu in record['affinity']))
        self.assertEqual(len(record['affinity']), self.budget['cpus'])

    def test_budget_requires_explicit_output_and_scratch_ceilings(self):
        for missing in ('output_mib', 'scratch_mib'):
            with self.subTest(missing=missing):
                req = self.request()
                del req['budget'][missing]
                env = dict(SLURM_JOB_ID='4242', SLURM_CPUS_PER_TASK=str(self.budget['cpus']))
                with patch.dict(os.environ, env, clear=False):
                    with self.assertRaises((ValueError, KeyError)):
                        self.w.slurm(req, self.control)

    def test_enforce_output_limit_sets_both_soft_and_hard(self):
        code = ('import json,resource,sys,importlib.util;'
                's=importlib.util.spec_from_file_location("w",{wrapper!r});'
                'w=importlib.util.module_from_spec(s);s.loader.exec_module(w);'
                'print(json.dumps(w.enforce_output_limit(4)))').format(wrapper=str(HERE/'wrapper.py'))
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        observed = json.loads(result.stdout)
        self.assertEqual(observed['after'], [4194304, 4194304])
        self.assertEqual(len(observed['before']), 2)

    def test_output_limit_actually_stops_an_oversized_write(self):
        target = self.base/'oversized.bin'
        code = ('import importlib.util,sys;'
                's=importlib.util.spec_from_file_location("w",{wrapper!r});'
                'w=importlib.util.module_from_spec(s);s.loader.exec_module(w);'
                'w.enforce_output_limit(1);'
                'open({target!r},"wb").write(b"x"*(4*1024*1024))').format(
                    wrapper=str(HERE/'wrapper.py'), target=str(target))
        result = subprocess.run([sys.executable, '-I', '-B', '-c', code],
                                capture_output=True, text=True, timeout=60)
        self.assertNotEqual(result.returncode, 0)
        self.assertLessEqual(target.stat().st_size, 1024*1024)

    def test_cgroup_observation_reads_declared_controller(self):
        root = self.base/'cgroup/memory'; (root/'slurm/job_4242').mkdir(parents=True)
        leaf = root/'slurm/job_4242'
        (leaf/'memory.limit_in_bytes').write_text('4294967296\n')
        (leaf/'memory.memsw.limit_in_bytes').write_text('4294967296\n')
        (leaf/'memory.use_hierarchy').write_text('1\n')
        (leaf/'memory.failcnt').write_text('0\n')
        (leaf/'memory.oom_control').write_text('oom_kill_disable 0\nunder_oom 0\noom_kill 0\n')
        proc = self.base/'proc_cgroup'
        proc.write_text('9:memory:/slurm/job_4242\n1:name=systemd:/user.slice\n')
        with patch.object(self.w, 'PROC_CGROUP', proc):
            observed = self.w.cgroup_observation(dict(required=True, root=str(root)))
        self.assertEqual(observed['path'], str(leaf))
        self.assertEqual(observed['values']['memory.limit_in_bytes'], '4294967296')
        self.assertEqual(observed['values']['memory.memsw.limit_in_bytes'], '4294967296')
        self.assertEqual(observed['values']['memory.failcnt'], '0')
        self.assertEqual(observed['values']['memory.oom_control'].splitlines()[1], 'under_oom 0')

    def test_required_cgroup_absent_is_rejected_not_silently_skipped(self):
        proc = self.base/'proc_cgroup'
        proc.write_text('9:memory:/slurm/job_4242\n')
        with patch.object(self.w, 'PROC_CGROUP', proc):
            with self.assertRaises(ValueError):
                self.w.cgroup_observation(dict(required=True, root=str(self.base/'absent')))

    def test_scratch_usage_rejects_an_over_budget_namespace(self):
        (self.control/'big.bin').write_text('x'*4096)
        self.w.check_scratch([self.control], 1)
        with self.assertRaises(ValueError):
            self.w.check_scratch([self.control], 0.001)


class ExecuteDrivenFixture(unittest.TestCase):
    """The fixture must drive wrapper.execute(), not reimplement the orchestration.

    Every substitution is declared in the request and recorded in the receipt.
    slurm() runs for real here against a stand-in scontrol; the alignment
    workload and the gate's data bindings are the declared seams.
    """
    def setUp(self):
        self.cpus = len(os.sched_getaffinity(0))
        if self.cpus < 4:
            self.skipTest('wrapper requires at least four allocated CPUs')
        self.tmp = tempfile.TemporaryDirectory(prefix='ont_execdriven_')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.code = self.base/'code/wrapper.py'; self.code.parent.mkdir()
        shutil.copyfile(HERE/'wrapper.py', self.code)
        spec = importlib.util.spec_from_file_location('execdriven_wrapper', self.code)
        self.w = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.w)
        driver_spec = importlib.util.spec_from_file_location('fixture_execute', HERE/'fixture_execute.py')
        self.fx = importlib.util.module_from_spec(driver_spec); driver_spec.loader.exec_module(self.fx)

        self.release = self.base/'release'; self.release.mkdir()
        for name in ('gate.py', 'source_pins.json'):
            shutil.copyfile(OLD_RELEASE/name, self.release/name)
        # The real package carries the wrapper inside the release, beside gate.py.
        shutil.copyfile(HERE/'wrapper.py', self.release/'wrapper.py')
        self.pins = json.loads((OLD_RELEASE/'source_pins.json').read_text())
        self.source = self.release/'frozen'
        for name in self.pins['files']:
            dst = self.source/name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(OLD_RELEASE/'frozen'/name, dst)

        self.root = self.base/'data'; self.root.mkdir(); self.w.ROOT = self.root
        self.run = 'SRR28341943'
        self.out = self.root/'output'
        self.roles = self.root/'evidence/DATASET_ROLES.longread_v1.tsv'
        self.reads = self.root/'evidence/ont/Athaliana_FLIC_PRJNA1087576'/(self.run+'.fa.gz')
        for q in (self.reads, self.roles, self.root/'genomes/Athaliana_167_TAIR10.fa',
                  self.root/'genomes/Athaliana_167_TAIR10.gene.clean.gff3', self.release/'authority.tsv'):
            q.parent.mkdir(parents=True, exist_ok=True); q.write_text('declared fixture stand-in\n')

        binary = self.base/'bin'; binary.mkdir()
        self.tools = {}
        for name in ('minimap2', 'samtools', 'seqkit'):
            t = binary/name; t.write_text('#!/bin/sh\nexit 0\n'); t.chmod(0o700)
            self.tools[name] = dict(path=str(t), sha256=self.sha(t))
        self.python = str(Path(sys.executable).resolve())
        self.tools['python'] = dict(path=self.python, sha256=self.sha(self.python))

        host = __import__('socket').gethostname()
        fields = ('JobId=4242 JobState=RUNNING UserId=u({uid}) NumNodes=1 NumCPUs={cpus} NumTasks=1 '
                  'NodeList={host} AllocTRES=cpu={cpus},mem=1024M RunTime=00:00:05 TimeLimit=00:30:00'
                  ).format(uid=os.getuid(), cpus=self.cpus, host=host)
        sc = binary/'scontrol'
        sc.write_text('#!/bin/sh\ncat <<\'EOF\'\n'+fields+'\nEOF\n'); sc.chmod(0o700)
        self.scontrol = dict(path=str(sc), sha256=self.sha(sc))

        self.cgroup_root = self.base/'cgroup/memory'
        leaf = self.cgroup_root/'slurm/job_4242'; leaf.mkdir(parents=True)
        for name, value in (('memory.limit_in_bytes', '1073741824'),
                            ('memory.memsw.limit_in_bytes', '1073741824'),
                            ('memory.use_hierarchy', '1'), ('memory.failcnt', '0'),
                            ('memory.memsw.failcnt', '0'),
                            ('memory.oom_control', 'oom_kill_disable 0\nunder_oom 0\noom_kill 0')):
            (leaf/name).write_text(value+'\n')
        self.proc_cgroup = self.base/'proc_cgroup'
        self.proc_cgroup.write_text('9:memory:/slurm/job_4242\n')

        self.subs = self.base/'fixture_substitutions.py'
        shutil.copyfile(HERE/'fixture_substitutions.py', self.subs)
        self.control = self.base/'control'
        self.reqpath = self.base/'request.json'

    @staticmethod
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def request(self, **over):
        req = self.fx.build_request(
            self.w, self.release, self.control, self.run,
            budget=dict(cpus=self.cpus, memory_mib=1024, wall_seconds=120, cleanup_seconds=5,
                        output_mib=4, scratch_mib=64),
            tools=self.tools, scontrol=self.scontrol,
            cgroup=dict(required=True, root=str(self.cgroup_root)),
            seams=['declared stand-in alignment workload, no approved runtime tooling on this host',
                   'declared stand-in scontrol, not a real allocation',
                   'declared synthetic cgroup v1 controller, not a real job cgroup',
                   'gate data bindings replaced by the pinned substitution module'],
            substitutions=dict(path=str(self.subs), sha256=self.sha(self.subs)),
            fixture=dict(output_directory=str(self.out), roles=str(self.roles)))
        req.update(over)
        return req

    def invoke(self, request=None, trace=None):
        request = self.request() if request is None else request
        original_child = self.w.child
        def child(argv, control, env, seconds, grace, receipt):
            # Declared seam: the real frozen alignment workload cannot run without the
            # approved runtime tooling. Real child/stop_group handling still executes.
            receipt['requested_driver_argv'] = argv
            code = ('from pathlib import Path; print("declared stand-in workload", flush=True); '
                    'Path({!r}).mkdir(parents=True); Path({!r}).write_text("tiny")').format(
                        str(self.out), str(self.out/'tiny.txt'))
            return original_child([self.python, '-I', '-B', '-c', code], control, env, seconds, grace, receipt)
        env = dict(SLURM_JOB_ID='4242', SLURM_CPUS_PER_TASK=str(self.cpus))
        with patch.dict(os.environ, env, clear=False), \
             patch.object(self.w, 'PROC_CGROUP', self.proc_cgroup), \
             patch.object(self.w, 'child', side_effect=child):
            if trace is None:
                return self.fx.run(self.w, request, self.reqpath)
            sys.settrace(trace)
            try:
                return self.fx.run(self.w, request, self.reqpath)
            finally:
                sys.settrace(None)

    def test_fixture_drives_execute_to_publication(self):
        receipt = self.invoke()
        self.assertEqual(receipt['status'], 'VERIFIED_DESCRIPTIVE')
        self.assertEqual((self.control/'COMPLETE').read_text(), self.sha(self.control/'receipt.json')+'\n')
        self.assertIsNone(receipt['decision'])
        self.assertFalse(receipt['training_eligible'])
        self.assertEqual(receipt['structural_status'], 'UNASSESSABLE')

    def test_slurm_really_ran_inside_execute(self):
        receipt = self.invoke()
        self.assertEqual(receipt['scheduler']['job'], '4242')
        self.assertEqual(receipt['scheduler']['affinity'], sorted(os.sched_getaffinity(0)))
        self.assertEqual(receipt['scheduler']['fields']['JobState'], 'RUNNING')
        self.assertTrue((self.control/'slurm.stdout').exists())

    def test_receipt_records_declared_seams_and_substitution_identity(self):
        receipt = self.invoke()
        self.assertEqual(receipt['seams'], self.request()['seams'])
        self.assertEqual(receipt['substitutions']['identity']['sha256'], self.sha(self.subs))
        self.assertIn('reference_binding', receipt['substitutions']['applied']['replaced'])

    def test_substituted_binding_is_detected_not_self_reported(self):
        receipt = self.invoke()
        bindings = receipt['bindings']
        self.assertTrue(bindings['slurm']['matches_expected'])
        self.assertFalse(bindings['gate.reference_binding']['matches_expected'])
        self.assertFalse(bindings['gate.verify_alignment']['matches_expected'])

    def test_request_without_declared_seams_is_rejected(self):
        req = self.request(); del req['seams']
        with self.assertRaises(ValueError) as caught:
            self.invoke(req)
        self.assertIn('seam', str(caught.exception))

    def test_substitution_module_hash_mismatch_is_rejected(self):
        req = self.request()
        req['substitutions'] = dict(path=str(self.subs), sha256='0'*64)
        with self.assertRaises(ValueError) as caught:
            self.invoke(req)
        self.assertIn('pin mismatch', str(caught.exception))
        self.assertIn('fixture_substitutions.py', str(caught.exception))

    def test_driver_rejects_a_release_pin_mismatch_before_import(self):
        with self.assertRaises(ValueError):
            self.fx.load_wrapper(self.release, '0'*64, self.sha(self.release/'gate.py'),
                                 self.sha(self.release/'source_pins.json'))

    def test_every_wrapper_function_is_exercised_through_execute(self):
        names = {n.name for n in __import__('ast').parse(self.code.read_text()).body
                 if isinstance(n, __import__('ast').FunctionDef)}
        seen = set()
        target = str(self.code)
        def tracer(frame, event, arg):
            if event == 'call' and frame.f_code.co_filename == target:
                seen.add(frame.f_code.co_name)
            return None
        self.invoke(trace=tracer)
        self.assertEqual(names - seen, set(), 'never reached through execute(): '+str(sorted(names-seen)))


if __name__=='__main__': unittest.main()

