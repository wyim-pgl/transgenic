import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import shutil
import os
import tarfile
import sys
import zlib
import authority
import run_panel_cpu as job

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RELEASE = HERE/'final_release'

class AuthorityTests(unittest.TestCase):
    def test_exact_original_objects_and_bytes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            root = Path(td)/'root'; root.mkdir()
            receipt = authority.install(REPO, root)
            expected = authority.git(REPO, 'show', authority.SPEC)
            self.assertEqual(authority.verify(root), receipt)
            self.assertEqual(authority.git(root, 'show', authority.SPEC), expected)
            self.assertEqual(len(receipt['objects']), 5)
            for row in receipt['objects']:
                self.assertEqual(authority.git(root,'cat-file',row['type'],row['oid']), authority.git(REPO,'cat-file',row['type'],row['oid']))
            self.assertEqual(authority.git(root,'for-each-ref'), b'')
            with self.assertRaises(FileExistsError): authority.install(REPO,root)
    def test_missing_or_tampered_objects_fail(self):
        for mutation in ['delete','corrupt','valid_object_wrong_oid']:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(dir=HERE) as td:
                root=Path(td)/'root'; root.mkdir(); receipt=authority.install(REPO,root)
                oid=receipt['objects'][-1]['oid']; p=root/'.git/objects'/oid[:2]/oid[2:]
                if mutation=='delete': p.unlink()
                else:
                    p.chmod(0o644)
                    p.write_bytes(b'corrupt' if mutation=='corrupt' else zlib.compress(b'blob 4\x00evil'))
                with self.assertRaises((ValueError,subprocess.CalledProcessError)): authority.verify(root)
    def test_parent_git_and_receipt_substitution_rejected(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            root=Path(td)/'root'; root.mkdir()
            with self.assertRaises(ValueError): authority.verify(root)
            authority.install(REPO,root)
            p=root/'reference_authority.json'; data=json.loads(p.read_text()); data['commit']='0'*40; p.write_text(json.dumps(data))
            with self.assertRaises(ValueError): authority.verify(root)

class JobTests(unittest.TestCase):
    def test_export_transport_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            with tarfile.open(RELEASE/'source.tar') as tar: tar.extractall(td)
            root=Path(td)/'source'
            self.assertEqual(job.verify_source(root)['component_sha256'],job.SOURCE_PIN)
            self.assertEqual(authority.git(root,'show',authority.SPEC),authority.git(REPO,'show',authority.SPEC))
            # Valid root and original objects, now alter source: must not reach input scanning.
            (root/'revision/scripts/mvp/mvp_prepare_panel_inputs.py').write_text('tampered')
            with self.assertRaises(ValueError): job.verify_source(root)
    def test_authority_only_transport_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            with tarfile.open(RELEASE/'reference_authority.tar') as tar: tar.extractall(td)
            self.assertEqual(authority.verify(td)['commit'],authority.COMMIT)
    def test_packaged_helpers_match_reviewed_files(self):
        for name in ('authority.py','run_panel_cpu.py','panel_cpu.sbatch','cpu_dependency.json'):
            self.assertEqual((RELEASE/name).read_bytes(),(HERE/name).read_bytes())
        receipt=json.loads((RELEASE/'transport.json').read_text())
        for name,digest in receipt['files'].items(): self.assertEqual(job.sha(RELEASE/name),digest)
    def test_pins_receipt_cannot_authorize_replacement(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            root=Path(td); (root/'source_pins.json').write_text(json.dumps({'commit':job.SOURCE_COMMIT,'source_files':{}}))
            with self.assertRaises(ValueError): job.verify_source(root)
    def test_login_invocation_refused_before_output_or_data(self):
        with tempfile.TemporaryDirectory(dir=HERE) as td:
            out=Path(td)/'new'
            env={k:v for k,v in os.environ.items() if k!='SLURM_JOB_ID'}
            proc=subprocess.run([sys.executable,str(HERE/'run_panel_cpu.py'),'--root',str(RELEASE/'source'),'--out',str(out)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            self.assertNotEqual(proc.returncode,0); self.assertIn(b'compute allocation',proc.stderr); self.assertFalse(out.exists())
    def test_duplicate_reporting_preserves_all_ids(self):
        # Test-only geometry objects, not GFF and never described as real reference data.
        class Producer:
            @staticmethod
            def structure(tx): return tx['key']
        txs={'a':{'key':('Chr1','+',1)},'b':{'key':('Chr1','+',1)},'noncoding':{'key':None}}
        groups=job.duplicate_groups(txs,Producer)
        self.assertEqual(groups[0]['transcript_ids'],['a','b']); self.assertEqual(len(txs),3)

if __name__=='__main__': unittest.main()
