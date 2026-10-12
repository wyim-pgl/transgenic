#!/usr/bin/env python3
"""Review-only tiny synthetic checks. NOT real tool/scientific/runtime acceptance."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
WRAPPER = HERE.parent/'wrapper.py'
OLD = HERE.parents[2]/'wait_work_20260916/longread'
PYTHON = str(Path(sys.executable).resolve())
sys.dont_write_bytecode = True

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m

class Review(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='fixture_', dir=HERE)
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.wrapper_copy = self.base/'code/wrapper.py'
        self.wrapper_copy.parent.mkdir()
        shutil.copyfile(WRAPPER, self.wrapper_copy)
        self.w = load(self.wrapper_copy, 'review_wrapper')
        self.release = self.base/'release'; self.release.mkdir()
        for name in ('gate.py', 'source_pins.json'):
            shutil.copyfile(OLD/name, self.release/name)
        self.pins = json.loads((OLD/'source_pins.json').read_text())
        for name in self.pins['files']:
            dst = self.release/'frozen'/name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(OLD/'frozen'/name, dst)
        self.source = self.release/'frozen'
        self.control = self.base/'control'
        self.root = self.base/'data'; self.root.mkdir()
        self.w.ROOT = self.root
        self.run = 'SRR28341943'
        self.reads = self.root/'evidence/ont/Athaliana_FLIC_PRJNA1087576'/f'{self.run}.fa.gz'
        self.roles = self.root/'evidence/DATASET_ROLES.longread_v1.tsv'
        self.ref = self.root/'genomes/Athaliana_167_TAIR10.fa'
        self.gff = self.root/'genomes/Athaliana_167_TAIR10.gene.clean.gff3'
        for p in (self.reads, self.roles, self.ref, self.gff, self.release/'authority.tsv'):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text('synthetic fixture\n')
        binary = self.base/'bin'; binary.mkdir()
        tools = {}
        for name in ('minimap2', 'samtools', 'seqkit', 'scontrol'):
            p = binary/name; p.write_text('#!/bin/sh\nexit 0\n'); p.chmod(0o700)
            tools[name] = dict(path=str(p), sha256=sha(p))
        self.request = dict(scope=self.w.SCOPE, runtime_reviewed=True, release=str(self.release),
                            control=str(self.control), run=self.run,
                            budget=dict(cpus=4, memory_mib=1024, wall_seconds=6, cleanup_seconds=1),
                            scontrol=tools.pop('scontrol'), tools=tools)
        tools['python'] = dict(path=PYTHON, sha256=sha(PYTHON))
        self.reqpath = self.base/'request.json'
        self.out = self.root/'output'

    def invoke(self, mode='success'):
        self.reqpath.write_text(json.dumps(self.request))
        original_spec = importlib.util.spec_from_file_location
        original_child = self.w.child
        driver = SimpleNamespace(output_directory=lambda root,run:self.out,
                                 w=SimpleNamespace(load_panel=lambda p: list(range(50))))
        def spec(name, path, *a, **kw):
            s = original_spec(name, path, *a, **kw)
            if name == 'ont_pinned_gate':
                real_exec = s.loader.exec_module
                def exec_module(g):
                    real_exec(g)
                    g.load_driver = lambda source: driver
                    g.read_binding = lambda *a: {'roles': {'path':str(self.roles)}}
                    g.reference_binding = lambda *a: {'files': {'fasta':{'md5':'0'*32},'gff':{'md5':'0'*32}}}
                    def postflight(args, receipt):
                        if mode == 'input_mutation': self.reads.write_text('mutated')
                        if mode == 'stderr_mutation': (self.control/'driver.stderr').write_text('mutated')
                        if mode == 'stdout_mutation': (self.control/'driver.stdout').write_text('mutated')
                        if mode == 'output_mutation': (self.out/'tiny.txt').write_text('mutated')
                        return {'synthetic':True}
                    g.verify_alignment = postflight
                s.loader.exec_module = exec_module
            return s
        def child(argv, control, env, seconds, grace, receipt):
            receipt['requested_driver_argv'] = argv
            code = 'from pathlib import Path; import sys,time; print("{}",flush=True); '
            code += f'Path({str(self.out)!r}).mkdir(); Path({str(self.out/"tiny.txt")!r}).write_text("tiny"); '
            if mode == 'nonzero': code += 'sys.exit(7)'
            elif mode in ('timeout','term'): code += 'time.sleep(20)'
            else: code += 'sys.exit(0)'
            timer = None
            if mode == 'term':
                timer = threading.Timer(.25, lambda: os.kill(os.getpid(), signal.SIGTERM)); timer.start()
            try:
                original_child([PYTHON,'-I','-B','-c',code],control,env,.15 if mode=='timeout' else seconds,.1,receipt)
            finally:
                if timer: timer.cancel(); timer.join()
        with patch.object(self.w,'slurm',return_value={'synthetic':True}), patch.object(self.w.importlib.util,'spec_from_file_location',side_effect=spec), patch.object(self.w,'child',side_effect=child):
            return self.w.execute(str(self.reqpath),sha(self.reqpath),sha(self.wrapper_copy))

    def test_execute_synthetic_success(self):
        r = self.invoke()
        self.assertEqual(r['status'],'VERIFIED_DESCRIPTIVE')
        self.assertEqual((self.control/'COMPLETE').read_text(),sha(self.control/'receipt.json')+'\n')
        self.assertFalse(r['training_eligible']); self.assertEqual(r['structural_status'],'UNASSESSABLE')
        argv = r['requested_driver_argv']
        self.assertEqual(argv[:4],[PYTHON,'-I','-B',str(self.source/'revision/scripts/mvp/mvp_align_panel_longreads.py')])
        self.assertIn('--ref-md5',argv); self.assertNotIn('--fasta',argv)

    def test_execute_failures_do_not_publish(self):
        # Separate fixture per failure, preserving exclusive namespace semantics.
        for mode in ('nonzero','timeout','term','input_mutation','stdout_mutation','output_mutation'):
            with self.subTest(mode=mode):
                case = Review(); case.setUp()
                try:
                    with self.assertRaises((ValueError,TimeoutError,InterruptedError)): case.invoke(mode)
                    self.assertFalse((case.control/'COMPLETE').exists())
                    self.assertTrue((case.control/'FAILED.json').exists())
                    self.assertEqual(json.loads((case.control/'receipt.json').read_text())['status'],'INCOMPLETE')
                    self.assertTrue((case.control/'driver.stdout').exists())
                finally: case.doCleanups()

    def test_execute_stderr_mutation_still_publishes(self):
        r = self.invoke('stderr_mutation')
        self.assertTrue((self.control/'COMPLETE').exists())
        self.assertNotEqual(r['stderr'],r['diagnostics']['driver.stderr'])

    def test_actual_isolated_driver_cli(self):
        p = subprocess.run([PYTHON,'-I','-B',str(self.source/'revision/scripts/mvp/mvp_align_panel_longreads.py'),'--help'],capture_output=True,text=True,timeout=5)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertIn('--ref-md5',p.stdout)

    def test_unlisted_source_module_executes_before_gate_rejects(self):
        marker = self.base/'unlisted_executed'
        (self.source/'src/transgenic.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\n')
        self.assertFalse(list(self.source.rglob('__init__.py')))
        self.assertFalse(list(self.source.rglob('*.pyc')))
        for name, expected in self.pins['files'].items(): self.assertEqual(sha(self.source/name),expected)
        code = f'import runpy; g=runpy.run_path({str(self.release/"gate.py")!r}); g["load_driver"]({str(self.source)!r})'
        p = subprocess.run([PYTHON,'-I','-B','-c',code],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(p.returncode,0)
        self.assertTrue(marker.exists(),p.stderr)

    def test_isolation_not_inherited_by_nested_sample(self):
        marker = self.base/'shadow_executed'
        (self.source/'revision/scripts/json.py').write_text(f'from pathlib import Path\nPath({str(marker)!r}).write_text("executed")\n')
        sample = self.source/'revision/scripts/68_sample_fasta.py'
        code = f'import subprocess,sys; raise SystemExit(subprocess.run([sys.executable,{str(sample)!r},"--help"]).returncode)'
        env = dict(PATH=str(self.base/'bin'),HOME=str(self.base),TMPDIR=str(self.base),LANG='C',LC_ALL='C',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
        p = subprocess.run([PYTHON,'-I','-B','-c',code],capture_output=True,text=True,env=env,timeout=5)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertTrue(marker.exists())

    def test_scheduler_valid_and_fail_closed(self):
        base = f'JobId=123 JobState=RUNNING UserId=test({os.getuid()}) NumNodes=1 NumCPUs=4 NumTasks=1 NodeList={socket.gethostname()} AllocTRES=cpu=4,mem=1G RunTime=00:00:01 TimeLimit=00:01:00'
        self.control.mkdir()
        variants = [('valid',base,True),('gpu',base.replace('mem=1G','mem=1G,gres/gpu=1'),False),('short',base.replace('00:01:00','00:00:02'),False),('unlimited',base.replace('00:01:00','UNLIMITED'),False),('bad_memory',base.replace('mem=1G','mem=2G'),False)]
        for name,text,ok in variants:
            d = self.control/name; d.mkdir()
            with patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_CPUS_PER_TASK':'4'}), patch.object(os,'sched_getaffinity',return_value=set(range(4))), patch.object(subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout=text,stderr='')):
                if ok: self.assertEqual(self.w.slurm(self.request,d)['job'],'123')
                else:
                    with self.assertRaises(ValueError): self.w.slurm(self.request,d)

    def test_same_group_descendant_is_stopped_after_driver_exit(self):
        pidfile = self.base/'pid'
        code = f'import subprocess; p=subprocess.Popen([{PYTHON!r},"-c","import time; time.sleep(20)"]); open({str(pidfile)!r},"w").write(str(p.pid))'
        self.control.mkdir(); receipt={}
        self.w.child([PYTHON,'-I','-B','-c',code],self.control,dict(os.environ),3,.1,receipt)
        pid = int(pidfile.read_text())
        proc = Path('/proc')/str(pid)/'stat'
        self.assertTrue(not proc.exists() or proc.read_text().split()[2]=='Z')

    def test_escaped_session_descendant_survives_group_cleanup(self):
        pidfile = self.base/'escaped_pid'
        code = f'import subprocess; p=subprocess.Popen([{PYTHON!r},"-c","import time; time.sleep(20)"],start_new_session=True); open({str(pidfile)!r},"w").write(str(p.pid))'
        self.control.mkdir(); receipt={}
        try:
            self.w.child([PYTHON,'-I','-B','-c',code],self.control,dict(os.environ),3,.1,receipt)
            pid = int(pidfile.read_text())
            self.assertNotEqual((Path('/proc')/str(pid)/'stat').read_text().split()[2],'Z')
        finally:
            if pidfile.exists():
                try: os.killpg(int(pidfile.read_text()), signal.SIGKILL)
                except ProcessLookupError: pass

if __name__ == '__main__': unittest.main(verbosity=2)
