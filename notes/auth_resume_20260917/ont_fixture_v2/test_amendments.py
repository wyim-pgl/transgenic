"""Local operational tests, never runtime/controller approval."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
import shutil
from types import SimpleNamespace
from contextlib import ExitStack
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fixture_v2', HERE/'fixture.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)

class Amendments(unittest.TestCase):
    def test_policy_rejects_hard_quota_and_wrong_budget(self):
        good = dict(output_policy='finite_fixture_soft_aggregate', aggregate_review_limit_bytes=268435456,
                    aggregate_hard_limit_bytes=None, per_file_hard_limit_bytes=4194304,
                    cpus=4, memory_mib=4096, wall_seconds=360)
        self.assertTrue(hasattr(f, 'check_policy'), 'missing executable policy gate')
        f.check_policy(good)
        for key, value in [('aggregate_hard_limit_bytes', 268435456), ('aggregate_review_limit_bytes', 1),
                           ('output_policy', 'hard_quota'), ('per_file_hard_limit_bytes', 8388608), ('cpus', 8)]:
            with self.assertRaises(ValueError): f.check_policy(dict(good, **{key:value}))

    def test_start_filesystem_metadata(self):
        self.assertTrue(hasattr(f, 'filesystem_record'), 'missing start filesystem record')
        record = f.filesystem_record(HERE)
        for key in ('resolved_path', 'filesystem_type', 'mount', 'available_bytes', 'available_inodes', 'uid', 'mode'):
            self.assertIn(key, record)
        self.assertEqual(record['resolved_path'], str(HERE))

    def test_publication_threshold_counts_own_files(self):
        self.assertTrue(hasattr(f, 'check_output_threshold'), 'missing final logical byte check')
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            out = Path(tmp)
            (out/'receipt.json').write_bytes(b'1234'); (out/'COMPLETE').write_bytes(b'5678')
            with self.assertRaisesRegex(ValueError, 'threshold'): f.check_output_threshold(out, 7)
            self.assertEqual(f.check_output_threshold(out, 9), 8)

    def test_term_and_int_raise_and_restore(self):
        self.assertTrue(hasattr(f, 'driver_signals'), 'missing signal cleanup bridge')
        for sig in (signal.SIGTERM, signal.SIGINT):
            old = signal.getsignal(sig)
            with self.assertRaises(InterruptedError):
                with f.driver_signals(): os.kill(os.getpid(), sig)
            self.assertEqual(signal.getsignal(sig), old)

    def test_signal_bridge_executes_real_wrapper_child_cleanup(self):
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            out=Path(tmp); receipt={}; wrapper=f.verify_release(HERE/'package')
            timer=threading.Timer(0.5, lambda: os.kill(os.getpid(), signal.SIGTERM))
            timer.start()
            try:
                with self.assertRaises(InterruptedError):
                    with f.driver_signals():
                        wrapper.child([sys.executable,'-I','-B','-c','import time; print("started",flush=True); time.sleep(30)'],out,dict(os.environ),10,0.2,receipt)
            finally: timer.join()
            self.assertLess(receipt['exit_code'],0)
            self.assertIn('stdout',receipt)
            with self.assertRaises(ProcessLookupError): os.killpg(receipt['pid'],0)

    def test_startup_limit_exact(self):
        script = (HERE/'fixture.sbatch').read_text()
        self.assertIn('ulimit -f 4096', script)
        result = subprocess.run(['/bin/bash', '-c', 'ulimit -f 4096; exec '+sys.executable+' -I -B -c "import resource; print(resource.getrlimit(resource.RLIMIT_FSIZE))"'], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), '(4194304, 4194304)')
        for name in ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'LD_AUDIT', 'BASH_ENV', 'PYTHONSTARTUP'):
            self.assertIn(name, script)

    def test_clean_launch_environment_preserves_slurm_and_ignores_bash_env(self):
        source=(HERE/'fixture.sbatch').read_text()
        sanitize=source[source.index('set -euo pipefail'):source.index(': "${RELEASE:')]
        launch=source[source.index('launch_env=('):source.index('/usr/bin/env -i "${launch_env[@]}"')]
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            marker=Path(tmp)/'injected'; startup=Path(tmp)/'bash_env'
            startup.write_text('touch '+str(marker)+'\n')
            command=sanitize+launch+'\n/usr/bin/env -i "${launch_env[@]}" '+sys.executable+' -I -B -c '+__import__('shlex').quote('import os,json; print(json.dumps(dict(os.environ)))')
            environment=dict(os.environ,LD_PRELOAD='',LD_LIBRARY_PATH='/unreviewed',LD_AUDIT='',
                BASH_ENV=str(startup),PYTHONPATH='/unreviewed',PYTHONSTARTUP='/unreviewed',
                SLURM_JOB_ID='123',SLURM_TEST_PRESERVE='yes',UNRELATED_SECRET='not-exported',
                ONT_FIXTURE_APPROVAL='/test-only',ONT_FIXTURE_APPROVAL_SHA256='test-only')
            result=subprocess.run(['/bin/bash','-p','-c',command],env=environment,capture_output=True,text=True,timeout=10)
            self.assertEqual(result.returncode,0,result.stderr)
            observed=json.loads(result.stdout)
            self.assertEqual(observed['SLURM_TEST_PRESERVE'],'yes')
            for key in ('LD_PRELOAD','LD_LIBRARY_PATH','LD_AUDIT','BASH_ENV','PYTHONPATH','PYTHONSTARTUP','UNRELATED_SECRET'):
                self.assertNotIn(key,observed)
            self.assertFalse(marker.exists())

    def test_runtime_inventory_and_cgroup_fail_closed(self):
        self.assertTrue((HERE/'runtime_closure.py').exists(), 'missing bounded runtime collector')
        c = f.module(HERE/'runtime_closure.py', 'closure_test')
        record = c.identity(Path(sys.executable))
        self.assertEqual(record['sha256'], f.digest(Path(sys.executable).resolve()))
        with self.assertRaises((ValueError, KeyError)):
            c.check_cgroup({})
        result = subprocess.run([sys.executable, '-I', '-B', '-c', c.PYTHON_INVENTORY], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        inventory = json.loads(result.stdout)
        self.assertIn('gzip', inventory['modules'])
        self.assertIn('site', inventory['modules'])
        self.assertIn('pth', inventory)

    def test_v1_cgroup_requires_job_hierarchy_memory_and_swap(self):
        c=f.module(HERE/'runtime_closure.py','closure_v1_test')
        current='/sys/fs/cgroup/memory/slurm/uid_1/job_123/step_0'
        job=str(Path(current).parent)
        files={'/proc/self/cgroup':'5:memory:/slurm/uid_1/job_123/step_0\n',
               current+'/tasks':str(os.getpid()), job+'/memory.use_hierarchy':'1',
               job+'/memory.limit_in_bytes':'4294967296',job+'/memory.memsw.limit_in_bytes':'4294967296'}
        with patch.dict(os.environ,{'SLURM_JOB_ID':'123'}),patch.object(Path,'exists',lambda p:str(p) in files),patch.object(Path,'read_text',lambda p:files[str(p)]):
            self.assertEqual(c.check_cgroup({'scheduler_cgroup_reviewed':True})['version'],1)
            files[job+'/memory.use_hierarchy']='0'
            with self.assertRaises(ValueError):c.check_cgroup({'scheduler_cgroup_reviewed':True})
            files[job+'/memory.use_hierarchy']='1';files[job+'/memory.memsw.limit_in_bytes']='8589934592'
            with self.assertRaises(ValueError):c.check_cgroup({'scheduler_cgroup_reviewed':True})

    def test_run_guard_and_failure_after_publication_keep_driver_outputs(self):
        # Simulate only external allocation/approval/runtime metadata and resource limits.
        # Real run(), frozen CLI, wrapper.child, stubs and postflight execute locally.
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            base = Path(tmp); launcher = base/'launcher'; launcher.mkdir()
            for name in ('fixture.py', 'fixture.sbatch', 'runtime_closure.py'):
                shutil.copyfile(HERE/name, launcher/name)
            (launcher/'MANIFEST.sha256').write_text('TEST ONLY')
            (launcher/'runtime_evidence').mkdir()
            (launcher/'runtime_evidence/expected_closure.json').write_text('{}')
            binary=base/'bin'; binary.mkdir(); tools={}
            for name in ('minimap2', 'samtools', 'seqkit'):
                path=binary/name
                path.write_text('#!'+sys.executable+'\nimport sys\nif sys.argv[1:] == ["--version"]:\n print("STUB"); sys.exit(0)\n'+(HERE/'stub_tool.py').read_text())
                path.chmod(0o700)
                tools[name]=dict(path=str(path),sha256=f.digest(path),version='STUB',version_args=['--version'])
            py=Path(sys.executable).resolve()
            tools['python']=dict(path=str(py),sha256=f.digest(py),version='Python '+sys.version.split()[0],version_args=['--version'])
            (launcher/'runtime.json').write_text(json.dumps(dict(tools=tools)))
            out=base/'out'
            approval=dict(scope='TEST_ONLY_REAL_TOOL_FIXTURE', runtime_closure_approved=True, output=str(out),
                output_policy='finite_fixture_soft_aggregate', aggregate_review_limit_bytes=268435456,
                aggregate_hard_limit_bytes=None, per_file_hard_limit_bytes=4194304,
                cpus=4,memory_mib=4096,wall_seconds=360, scheduler_logs=[str(base/'stdout'),str(base/'stderr')],
                manifest_sha256=f.digest(launcher/'MANIFEST.sha256'), runtime_closure_sha256=f.digest(launcher/'runtime_evidence/expected_closure.json'))
            ap=base/'approval.json'; ap.write_text(json.dumps(approval))
            realmodule=f.module
            def modules(path,name):
                if Path(path).name=='runtime_closure.py': return SimpleNamespace(collect=lambda *a:{},check_cgroup=lambda *a:{})
                return realmodule(path,name)
            original_stat=Path.stat
            def private_stat(path, *args, **kwargs):
                st=original_stat(path,*args,**kwargs)
                if path == out:
                    values=list(st); values[0]=(st.st_mode & ~0o777) | 0o700
                    return os.stat_result(values)  # Local drvfs cannot enforce Unix private modes.
                return st
            with ExitStack() as stack:
                for p in [patch.object(Path,'stat',private_stat),patch.object(f,'HERE',launcher),patch.object(f,'module',side_effect=modules),
                    patch.dict(os.environ,{'SLURM_JOB_ID':'123','SLURM_CPUS_PER_TASK':'4','ONT_FIXTURE_APPROVAL':str(ap),'ONT_FIXTURE_APPROVAL_SHA256':f.digest(ap)},clear=True),
                    patch.object(f.os,'sched_getaffinity',return_value={0,1,2,3}),
                    patch.object(f.resource,'getrlimit',return_value=(4194304,4194304)),patch.object(f.resource,'setrlimit'),
                    patch('builtins.print',side_effect=BrokenPipeError('final stdout failure'))]: stack.enter_context(p)
                rejected=base/'policy_rejected'
                ap.write_text(json.dumps(dict(approval,output=str(rejected),aggregate_review_limit_bytes=1)))
                os.environ['ONT_FIXTURE_APPROVAL_SHA256']=f.digest(ap)
                with self.assertRaisesRegex(ValueError,'policy mismatch'): f.run(HERE/'package',rejected)
                self.assertFalse((rejected/'synthetic').exists())
                ap.write_text(json.dumps(approval));os.environ['ONT_FIXTURE_APPROVAL_SHA256']=f.digest(ap)
                with self.assertRaises(BrokenPipeError): f.run(HERE/'package',out)
            self.assertFalse((out/'COMPLETE').exists())
            self.assertTrue((out/'FAILED.json').exists())
            self.assertTrue((out/'driver.stdout').exists())
            self.assertTrue((out/'synthetic/evidence/mvp_scoring_20260916'/f.RUN/'junctions.tsv').exists())
            self.assertEqual(json.loads((out/'receipt.json').read_text())['status'],'INCOMPLETE')

    def test_python39_refused_before_allocation(self):
        with patch.object(f.sys, 'version_info', (3,9)):
            with self.assertRaisesRegex(ValueError, 'Python >=3.10'):
                f.run(HERE/'package', HERE/'never_created')

if __name__ == '__main__': unittest.main()
