import importlib.util
import json
import os
from pathlib import Path
import sys
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
                     stdout=self.w.identity(p), outputs={})
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

if __name__=='__main__': unittest.main()
