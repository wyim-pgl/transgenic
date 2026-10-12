"""Independent targeted probes, no real runtime or source inputs."""
import hashlib, importlib.util, json, os, signal, sys, time, unittest
from pathlib import Path
from unittest.mock import patch
E = Path(__file__).resolve().parent
D = E.parent.parent / 'seed123_execution_20260917/est_deployment'
spec = importlib.util.spec_from_file_location('fixtures', D/'test_deployment.py')
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
records = []

class Independent(unittest.TestCase):
    def setUp(self):
        self.case = f.Deployment(); self.case.setUp(); self.addCleanup(self.case.doCleanups)
        self.case.build(); self.r = self.case.runner
        self.out = self.case.root/'output'

    def check_signal(self, phase, sig):
        original_write = self.r.write_json
        original_mask = signal.pthread_sigmask
        original_binding = self.r.report_binding
        fired = []
        def fire():
            if not fired:
                fired.append(True); os.kill(os.getpid(), sig)
        def write(path, value):
            original_write(path, value)
            if path.name == phase:
                fire()
        def mask(how, values):
            old = original_mask(how, values)
            if phase == 'pending' and how == signal.SIG_BLOCK:
                fire()
            return old
        def binding(*args):
            if phase == 'binding': fire()
            return original_binding(*args)
        with patch.object(self.r, 'runtime_identity', return_value={'fixture':True}), \
             patch.object(self.r, 'write_json', side_effect=write), \
             patch.object(self.r, 'report_binding', side_effect=binding), \
             patch.object(signal, 'pthread_sigmask', side_effect=mask):
            rc = self.r.run(self.case.release, self.case.sha, self.out)
        result = json.loads((self.out/'control/result.json').read_text())
        self.assertTrue(fired); self.assertNotEqual(rc, 0)
        self.assertEqual(result['status'], 'INCOMPLETE'); self.assertTrue(result['interrupted'])
        self.assertFalse((self.out/'control/verification.json').exists())
        if phase != 'start.json':
            self.assertEqual((self.out/'sample/diagnostics.jsonl').read_bytes(), b'fixture only\n')
        if phase == 'verification.json':
            self.assertTrue((self.out/'control/verification.interrupted.json').exists())
        records.append(dict(phase=phase, signal=int(sig), rc=rc, result=result))

    def test_start_term(self): self.check_signal('start.json', signal.SIGTERM)
    def test_binding_term(self): self.check_signal('binding', signal.SIGTERM)
    def test_pending_term(self): self.check_signal('pending', signal.SIGTERM)
    def test_pending_usr1(self): self.check_signal('pending', signal.SIGUSR1)
    def test_result_usr1(self): self.check_signal('result.json', signal.SIGUSR1)
    def test_verification_usr1(self): self.check_signal('verification.json', signal.SIGUSR1)
    def test_binding_deadline(self):
        original = self.r.report_binding
        def slow(*args): time.sleep(5); return original(*args)
        with patch.object(self.r, 'runtime_identity', return_value={'fixture':True}), \
             patch.object(self.r, 'POSTCHECK_SECONDS', 2), \
             patch.object(self.r, 'report_binding', side_effect=slow):
            started = time.monotonic(); rc = self.r.run(self.case.release, self.case.sha, self.out)
        self.assertLess(time.monotonic()-started, 5); self.assertNotEqual(rc, 0)
        result = json.loads((self.out/'control/result.json').read_text())
        self.assertEqual(result['postchecks']['binding'], 'FAIL'); self.assertTrue(result['deadline_expired'])
        self.assertFalse((self.out/'control/verification.json').exists())
        self.assertEqual((self.out/'sample/diagnostics.jsonl').read_bytes(), b'fixture only\n')
        records.append(dict(phase='binding-deadline', rc=rc, result=result))

    def test_helper_link_and_namespace_guards(self):
        helper = f.load('prepare_controller')
        pins = json.loads((self.case.release/'ledger.json').read_text())['files']
        ns = self.case.root/'ns'
        args = [self.case.release, self.case.sha, pins['run_compute.py'], pins['sample.sbatch']]
        target = self.case.release/'run_compute.py'
        alias = self.case.root/'alias'; os.link(target, alias)
        with self.assertRaises(ValueError): helper.prepare(*args, ns, ns/'stdout', ns/'stderr')
        self.assertFalse(ns.exists()); alias.unlink()
        alias.symlink_to(self.case.release, target_is_directory=True)
        with self.assertRaises(ValueError): helper.prepare(alias, *args[1:], ns, ns/'stdout', ns/'stderr')
        self.assertFalse(ns.exists())
        traversed = self.case.root/'missing'/'..'/'ns'
        with self.assertRaises(ValueError): helper.prepare(*args, traversed, traversed/'stdout', traversed/'stderr')
        self.assertFalse(ns.exists())
        receipt = helper.prepare(*args, ns, ns/'stdout', ns/'stderr')
        modes = {n: oct((ns/n).stat().st_mode & 0o777) for n in ('stdout','stderr','preparation.json')}
        modes['namespace'] = oct(ns.stat().st_mode & 0o777)
        records.append(dict(phase='helper-effective-modes', receipt=receipt, modes=modes))
        # Record the mode defect separately, then finish independent retention checks.
        self.assertEqual(receipt['status'], 'PREPARED_NOT_SUBMITTED')
        (ns/'stdout').write_bytes(b'KEEP_STDOUT\n'); (ns/'stderr').write_bytes(b'KEEP_STDERR\n')
        with self.assertRaises(FileExistsError): helper.prepare(*args, ns, ns/'stdout', ns/'stderr')
        self.assertEqual((ns/'stdout').read_bytes(), b'KEEP_STDOUT\n')
        self.assertEqual((ns/'stderr').read_bytes(), b'KEEP_STDERR\n')
        self.assertFalse(Path(receipt['worker_output']).exists())
        records.append(dict(phase='helper-extra-guards', receipt=receipt))

if __name__ == '__main__':
    try: unittest.main()
    finally: (E/'adversarial-results.json').write_text(json.dumps(records, indent=2)+'\n')
