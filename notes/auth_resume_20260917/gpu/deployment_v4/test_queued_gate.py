"""Synthetic fixtures only. Never an authorization or a live admission."""
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import deploy as d

class QueuedGateTests(unittest.TestCase):
    def test_queued_authority_has_no_observation_age(self):
        self.assertTrue(hasattr(d, 'admission'), 'durable/live split missing')
        a = d.admission.example()
        a.update(approved=True, issued_at_unix=1000, controller_reference='fixture',
                 review_reference='fixture', terminal_recovery_reference='fixture',
                 staging_reference='fixture', reservation_identity='fixture')
        a['release_json_sha256'] = 'a'*64
        a['transport_sha256'] = 'b'*64
        a['ledger_sha256'] = 'c'*64
        before = d.binding_digest(a)
        d.admission.validate_authorization(a, 'a'*64, 'b'*64, 'c'*64, now=87400)
        self.assertEqual(before, d.binding_digest(a))
        for key, value in [('approved', False), ('seed', True), ('issued_at_unix', 87401),
                           ('training_authorized', True), ('release_json_sha256', 'd'*64),
                           ('serialized_through_terminal_reconciliation', False)]:
            bad = copy.deepcopy(a); bad[key] = value
            with self.assertRaises(ValueError):
                d.admission.validate_authorization(bad, 'a'*64, 'b'*64, 'c'*64, now=87400)

    def test_live_observation_freshness(self):
        self.assertTrue(hasattr(d, 'admission'), 'bounded observation checker missing')
        r = dict(start_monotonic=100, end_monotonic=101, start_unix=1000, end_unix=1001,
                 exit_code=0, stderr='', stdout='complete')
        d.admission.fresh([r], 400, 1300)
        for key, value in [('start_monotonic', 99), ('end_monotonic', 401),
                           ('end_unix', 1301), ('exit_code', 1), ('stderr', 'warning'),
                           ('stdout', ''), ('start_unix', float('nan'))]:
            bad = dict(r); bad[key] = value
            with self.assertRaises(ValueError): d.admission.fresh([bad], 400, 1300)

    def test_caps_independent_and_self_once(self):
        self.assertTrue(hasattr(d, 'admission'), 'accounting reconciliation missing')
        self.assertEqual(d.admission.budget([(0.2, 0.8)], [(1, 4)]), (1.7, 6.8))
        for spent in [[(4.6, 0)], [(0, 18.1)], [(float('nan'), 0)]]:
            with self.assertRaises(ValueError): d.admission.budget(spent, [])

    def test_finance_requires_clear_units_and_full_b5_reserve(self):
        self.assertTrue(hasattr(d, 'admission'), 'finance gate missing')
        text = "Project Summary for User 'wcyim':\n\nAccount                        Balance(Hours)   Deposited(Hours)  Project\n----------------------------  ----------------  ----------------  ----------------------\nbilv-dtai-gh                              1154              1499  example\n"
        self.assertEqual(d.admission.balance(text), 1154)
        for bad in [text.replace('Balance(Hours)', 'Balance'), '', text+text,
                    text.replace('1154', 'NaN')]:
            with self.assertRaises(ValueError): d.admission.balance(bad)
        self.assertEqual(d.admission.reserve(1154, [44]*8), 392)
        with self.assertRaises(ValueError): d.admission.reserve(391, [44]*8)

    def test_launcher_failure_retains_exclusive_claim_and_refuses_second_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'deploy.py').write_text("from pathlib import Path\nPath('invoked').open('x').write('once')\nraise SystemExit(37)\n")
            env = dict(os.environ, SLURM_JOB_ID='4000000', SLURM_JOB_NUM_NODES='1', SLURM_GPUS_ON_NODE='4',
                DEPLOY_CONTROLLER_GATE_SHA256='a'*64, DEPLOY_RELEASE_TRANSPORT_SHA256='b'*64,
                DEPLOY_RELEASE_LEDGER_SHA256='c'*64)
            script = Path(__file__).with_name('p1.sbatch')
            first = subprocess.run(['/bin/bash',str(script)],cwd=root,env=env,capture_output=True,timeout=10)
            self.assertEqual(first.returncode,37)
            self.assertEqual((root/'S0_P1_STARTED').read_text(),'4000000\n')
            second = subprocess.run(['/bin/bash',str(script)],cwd=root,env=env,capture_output=True,timeout=10)
            self.assertNotEqual(second.returncode,0)
            self.assertEqual((root/'invoked').read_text(),'once')
            self.assertEqual((root/'S0_P1_STARTED').read_text(),'4000000\n')

    def test_timeout_is_once_and_persistent_evidence(self):
        self.assertTrue(hasattr(d, 'admission'), 'bounded query runner missing')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); claim = root/'S0_P1_STARTED'; claim.write_text('fixture')
            with self.assertRaises(ValueError):
                d.admission.query(root, 'timeout', ['/usr/bin/sleep', '2'], timeout=0.02)
            record = json.loads((root/'timeout.json').read_text())
            self.assertTrue(record['timed_out'])
            self.assertEqual(claim.read_text(), 'fixture')
            self.assertEqual(len(list(root.glob('timeout*.json'))), 1)

if __name__ == '__main__': unittest.main()
