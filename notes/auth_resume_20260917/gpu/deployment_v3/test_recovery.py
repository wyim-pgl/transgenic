"""Local synthetic regression fixtures, never real authorization."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import admission as a
import deploy as d
from test_admission_cases import fixture, NOW

class RecoveryTests(unittest.TestCase):
    def test_known_resume_directories_only(self):
        self.assertTrue(hasattr(d, 'checkpoint_inventory'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ('model.safetensors', 'meta.json', 'eval.json'):
                (root/name).write_text(name)
            self.assertEqual(d.checkpoint_inventory(root), [])
            for name in ('accelerate_state', 'pending_grads'):
                (root/name).mkdir(); (root/name/'private').write_text('do not read or copy')
            self.assertEqual(d.checkpoint_inventory(root), ['accelerate_state', 'pending_grads'])
            (root/'unknown').mkdir()
            with self.assertRaises(ValueError): d.checkpoint_inventory(root)
            (root/'unknown').rmdir()
            (root/'pending_grads/private').unlink(); (root/'pending_grads').rmdir()
            (root/'pending_grads').symlink_to(root/'accelerate_state')
            with self.assertRaises(ValueError): d.checkpoint_inventory(root)
            (root/'pending_grads').unlink(); (root/'pending_grads').write_text('not a directory')
            with self.assertRaises(ValueError): d.checkpoint_inventory(root)

    def test_recovery_copy_preserves_source_and_refuses_overwrite(self):
        self.assertTrue(hasattr(d, 'recover_evidence'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'old.db'; source.write_bytes(b'fixture'); source.chmod(0o444)
            with patch.object(d, 'EVIDENCE_SOURCE', source), patch.object(d, 'EVIDENCE_BYTES', 7), \
                 patch.object(a, 'UID', os.getuid()), patch.dict(d.PINS, evidence=d.sha(source)):
                before = source.stat()
                d.recover_evidence(root/'new.db')
                self.assertEqual((root/'new.db').read_bytes(), b'fixture')
                self.assertEqual(source.stat().st_ino, before.st_ino)
                self.assertEqual(source.stat().st_mtime_ns, before.st_mtime_ns)
                self.assertEqual(source.stat().st_ctime_ns, before.st_ctime_ns)
                with self.assertRaises(FileExistsError): d.recover_evidence(root/'new.db')
                for mode in (0o644, 0o664):
                    source.chmod(mode)
                    with self.assertRaises(ValueError): d.recover_evidence(root/'bad.db')
                source.chmod(0o444)
                (root/'old.db-wal').write_text('wal')
                with self.assertRaises(ValueError): d.recover_evidence(root/'bad.db')
                (root/'old.db-wal').unlink()
                with patch.dict(d.PINS, evidence='0'*64):
                    with self.assertRaises(ValueError): d.recover_evidence(root/'bad.db')
                self.assertFalse((root/'bad.db').exists())

    def test_recovery_rejects_symlink_owner_and_copy_time_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'; source.write_bytes(b'fixture'); source.chmod(0o444)
            with patch.object(d,'EVIDENCE_SOURCE',source), patch.object(d,'EVIDENCE_BYTES',7), \
                 patch.object(a,'UID',os.getuid()), patch.dict(d.PINS,evidence=d.sha(source)):
                with patch.object(a,'UID',os.getuid()+1):
                    with self.assertRaises(ValueError): d.recover_evidence(root/'wrong-owner')
                alias=root/'alias'; alias.symlink_to(source)
                with patch.object(d,'EVIDENCE_SOURCE',alias):
                    with self.assertRaises(ValueError): d.recover_evidence(root/'alias-dest')
                real_copy=d.shutil.copyfileobj
                def corrupt_destination(stream,out,length):
                    real_copy(stream,out,length);out.write(b'bad')
                with patch.object(d.shutil,'copyfileobj',side_effect=corrupt_destination):
                    with self.assertRaises(ValueError): d.recover_evidence(root/'bad-destination')
                def replace_source(stream,out,length):
                    real_copy(stream,out,length)
                    replacement=root/'replacement'; replacement.write_bytes(b'fixture');replacement.chmod(0o444)
                    replacement.replace(source)
                with patch.object(d.shutil,'copyfileobj',side_effect=replace_source):
                    with self.assertRaisesRegex(ValueError,'source changed'): d.recover_evidence(root/'source-race')
                self.assertEqual(source.read_bytes(),b'fixture')

    def test_true_failed_history_and_exclusive_cpu_capacity(self):
        auth, current, jobs, rows = fixture()
        jobs[current].update(NumCPUs='288', CPUsPerTask='64', NumTasks='1', AllocTRES='billing=4000,cpu=288,gres/gpu=4,node=1')
        rows[current].update(cpus='288', alloc=jobs[current]['AllocTRES'])
        rows['3163986'].update(state='FAILED', elapsed='51', nodes='1', cpus='288',
            alloc=jobs[current]['AllocTRES'], exit='1:0', start='2026-09-17T18:47:46',
            end='2026-09-17T18:48:37', placement='gh077')
        rows['3163545']['exit']='0:0'
        self.assertTrue(hasattr(a, 'node_capacity'))
        node = a.node_capacity('NodeName=gh018 CPUTot=288', 'gh018')
        result = a.reconcile(auth, current, jobs, rows, NOW, node)
        self.assertAlmostEqual(result['node_hours'], 1.5+51/3600)
        self.assertAlmostEqual(result['gpu_hours'], 6+204/3600)
        for key, value in [('state','CANCELLED'), ('elapsed','0'), ('exit','0:0'), ('nodes','2'), ('cpus','64')]:
            bad=copy.deepcopy(rows); bad['3163986'][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.reconcile(auth,current,jobs,bad,NOW,node)
        for key, value in [('NumCPUs','128'), ('CPUsPerTask','288'), ('NumTasks','2'), ('ReqTRES','billing=4000,cpu=288,gres/gpu=4,node=1')]:
            bad=copy.deepcopy(jobs);bad[current][key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.reconcile(auth,current,bad,rows,NOW,node)
        with self.assertRaises(ValueError): a.reconcile(auth,current,jobs,rows,NOW)
        with self.assertRaises(ValueError): a.node_capacity('NodeName=gh019 CPUTot=288','gh018')

    def test_host_python36_api(self):
        real = a.subprocess.Popen
        def python36(*args, **kwargs):
            self.assertNotIn('text', kwargs)
            return real(*args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, patch.object(a.subprocess, 'Popen', side_effect=python36):
            a.query(Path(tmp), 'compat', ['/bin/printf', 'ok'])
        for name in ('admission.py','deploy.py'):
            ast.parse(Path(__file__).with_name(name).read_text(), feature_version=(3,6))

    def test_recovery_authority_is_explicit_and_quota_fits(self):
        auth=a.example()
        self.assertIn('terminal_recovery_reference',auth)
        self.assertNotIn('cancellation_history_reference',auth)
        self.assertEqual(auth['evidence_recovery']['source'], str(d.EVIDENCE_SOURCE))
        self.assertEqual(auth['evidence_recovery']['sha256'], d.PINS['evidence'])
        self.assertFalse(auth['approved'])
        self.assertFalse(auth['training_authorized'])
        auth.update(approved=True,issued_at_unix=1000,controller_reference='fixture',review_reference='fixture',
            terminal_recovery_reference='fixture',staging_reference='fixture',reservation_identity='fixture',
            release_json_sha256='a'*64,transport_sha256='b'*64,ledger_sha256='c'*64)
        for key in ('source','sha256','bytes','owner','read_only','preserve_source'):
            bad=copy.deepcopy(auth);bad['evidence_recovery'][key]=None
            with self.subTest(key=key), self.assertRaises(ValueError):
                a.validate_authorization(bad,'a'*64,'b'*64,'c'*64,now=87400)
        bad=copy.deepcopy(auth);bad['terminal_history']['3163986']['elapsed_seconds']=0
        with self.assertRaises(ValueError): a.validate_authorization(bad,'a'*64,'b'*64,'c'*64,now=87400)
        self.assertLess(1068773376+2200000000+2_000_000_000, 5*2**30)

if __name__ == '__main__': unittest.main()
