"""Real-file archive safety tests; no torch or third-party dependencies."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

MODULE = Path(__file__).resolve().parents[1] / 'revision/scripts/b6/checkpoint_archive.py'
spec = importlib.util.spec_from_file_location('checkpoint_archive', str(MODULE))
archive = importlib.util.module_from_spec(spec) if MODULE.exists() else None
if archive is not None:
    spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root / 'run'
        self.dest = self.root / 'hdd'
        self.run.mkdir()
        self.dest.mkdir()
        for epoch in range(1, 5):
            self.epoch(epoch)
        (self.run / 'best').symlink_to('epoch_01')

    def epoch(self, n, pending=1):
        path = self.run / ('epoch_%02d' % n)
        path.mkdir()
        files = {'model.safetensors': b'weights' + bytes([n]),
                 'accelerate_state/model.safetensors': b'accel',
                 'accelerate_state/optimizer.bin': b'optimizer',
                 'accelerate_state/scheduler.bin': b'scheduler',
                 'accelerate_state/random_states_0.pkl': b'random'}
        if pending:
            files['pending_grads/rank0.safetensors'] = b'gradient'
        for name, data in files.items():
            f = path / name
            f.parent.mkdir(exist_ok=True)
            f.write_bytes(data)
        (path / 'meta.json').write_text(json.dumps(dict(epoch=n, seed=123,
            global_step=n * 10, pending_micro=pending)))
        (path / 'eval.json').write_text(json.dumps(dict(epoch=n, global_step=n*10)))
        return path

    def prepared(self):
        return archive.prepare(str(self.run), 2)

    def copied(self):
        manifest = self.prepared()
        return manifest, archive.transfer(manifest, str(self.dest))

    def test_plan_protects_best_latest_two_and_ignores_non_epochs(self):
        self.assertIsNotNone(archive, 'archive implementation is missing')
        for name in ('latest_state', 'latest_state.tmp', 'epoch_05.tmp', 'step_100', 'PILOT_FINAL'):
            (self.run / name).mkdir()
        result = archive.plan(str(self.run))
        self.assertEqual([c['epoch'] for c in result['candidates']], [2])
        self.assertEqual(result['protected'], [1, 3, 4])

    def test_missing_pending_grad_is_refused(self):
        (self.run / 'epoch_02/pending_grads/rank0.safetensors').unlink()
        self.assertEqual(archive.plan(str(self.run))['candidates'], [])

    def test_bad_metadata_and_incomplete_optimizer_are_refused(self):
        for rel in ('meta.json', 'accelerate_state/optimizer.bin'):
            with self.subTest(rel=rel):
                path = self.run / 'epoch_02' / rel
                data = path.read_bytes()
                path.unlink()
                self.assertEqual(archive.plan(str(self.run))['candidates'], [])
                path.write_bytes(data)

    def test_age_and_total_bytes_limits(self):
        self.assertEqual(archive.plan(str(self.run), min_age_seconds=300)['candidates'], [])
        self.assertEqual(archive.plan(str(self.run), max_bytes=1)['candidates'], [])
        self.assertEqual(archive.plan(str(self.run), max_epoch=1)['candidates'], [])

    def test_prepare_hashes_all_files_and_copy_is_idempotent(self):
        manifest, receipt = self.copied()
        target = Path(receipt['archive_path'])
        self.assertEqual((target / 'model.safetensors').read_bytes(), b'weights\x02')
        self.assertEqual((target / 'pending_grads/rank0.safetensors').read_bytes(), b'gradient')
        self.assertTrue((target / '.archive_receipt.json').is_file())
        self.assertTrue((self.run / 'epoch_02').is_dir())
        self.assertEqual(archive.transfer(manifest, str(self.dest)), receipt)
        self.assertEqual(archive.verify_archive(manifest, str(target)), receipt)

    def test_source_mutation_prevents_transfer(self):
        manifest = self.prepared()
        (self.run / 'epoch_02/model.safetensors').write_bytes(b'changed')
        with self.assertRaises(archive.ArchiveError):
            archive.transfer(manifest, str(self.dest))
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_corrupt_destination_prevents_cleanup_and_retry(self):
        manifest, receipt = self.copied()
        (Path(receipt['archive_path']) / 'model.safetensors').write_bytes(b'corrupt')
        for operation in (lambda: archive.cleanup(manifest, receipt),
                          lambda: archive.transfer(manifest, str(self.dest))):
            with self.assertRaises(archive.ArchiveError):
                operation()
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_cleanup_preserves_best_and_live_state_and_is_idempotent(self):
        live = self.run / 'latest_state'
        live.mkdir()
        (live / 'state').write_bytes(b'live')
        manifest, receipt = self.copied()
        self.assertEqual(archive.cleanup(manifest, receipt)['status'], 'removed')
        self.assertFalse((self.run / 'epoch_02').exists())
        self.assertEqual((live / 'state').read_bytes(), b'live')
        self.assertTrue((self.run / 'epoch_01/model.safetensors').exists())
        self.assertEqual(archive.cleanup(manifest, receipt)['status'], 'removed')

    def test_new_best_after_preparation_prevents_cleanup(self):
        manifest, receipt = self.copied()
        (self.run / 'best').unlink()
        (self.run / 'best').symlink_to('epoch_02')
        with self.assertRaises(archive.ArchiveError):
            archive.cleanup(manifest, receipt)
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_symlink_and_overlapping_roots_refused(self):
        (self.run / 'epoch_02/link').symlink_to(self.root)
        self.assertEqual(archive.plan(str(self.run))['candidates'], [])
        (self.run / 'epoch_02/link').unlink()
        manifest = self.prepared()
        with self.assertRaises(archive.ArchiveError):
            archive.transfer(manifest, str(self.run / 'archive'))

    def test_pool_and_quota_failures_preserve_source(self):
        manifest = self.prepared()
        for check in (lambda p: dict(pool='ddn_ssd', project_id=7, quota_free_bytes=100000),
                      lambda p: dict(pool='ddn_hdd', project_id=7, quota_free_bytes=0)):
            with self.assertRaises(archive.ArchiveError):
                archive.transfer(manifest, str(self.dest), storage_check=check,
                                 required_pool='ddn_hdd', required_project=7)
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_lock_excludes_separate_process(self):
        code = ('import sys; sys.path.insert(0, %r); import checkpoint_archive as a; '
                'a.prepare(%r, 2)' % (str(MODULE.parent), str(self.run)))
        with archive.locked(str(self.run)):
            result = subprocess.run([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'locked', result.stderr)

    def test_previously_verified_archive_can_be_archived_again(self):
        first, receipt = self.copied()
        second = archive.prepare_archive(receipt['archive_path'])
        self.assertEqual(second['upstream_receipt'], receipt)
        final = archive.transfer(second, str(self.root / 'pronghorn'))
        self.assertEqual((Path(final['archive_path']) / 'model.safetensors').read_bytes(), b'weights\x02')
        archive.cleanup(second, final)
        self.assertFalse(Path(receipt['archive_path']).exists())
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_source_changed_after_publication_cannot_be_removed(self):
        manifest, receipt = self.copied()
        (self.run / 'epoch_02/extra').write_bytes(b'unexpected')
        with self.assertRaises(archive.ArchiveError):
            archive.cleanup(manifest, receipt)
        self.assertTrue((self.run / 'epoch_02/extra').exists())

    def test_wrong_remote_acknowledgment_keeps_source(self):
        manifest, receipt = self.copied()
        with self.assertRaises(archive.ArchiveError):
            archive.cleanup(manifest, receipt, verifier=lambda m, r: dict(r, manifest_id='stale'))
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_latest_two_change_before_cleanup_keeps_source(self):
        manifest, receipt = self.copied()
        shutil.rmtree(str(self.run / 'epoch_04'))
        with self.assertRaises(archive.ArchiveError):
            archive.cleanup(manifest, receipt)
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_copy_failure_keeps_source_and_restart_finishes(self):
        manifest = self.prepared()
        def interrupted(src, dst, signature, rate):
            Path(dst).write_bytes(b'partial')
            raise OSError('ENOSPC during copy')
        with mock.patch.object(archive, '_copy', side_effect=interrupted):
            with self.assertRaises(OSError):
                archive.transfer(manifest, str(self.dest))
        self.assertTrue((self.run / 'epoch_02/model.safetensors').exists())
        receipt = archive.transfer(manifest, str(self.dest))
        self.assertEqual((Path(receipt['archive_path']) / 'model.safetensors').read_bytes(), b'weights\x02')

    def test_publish_failure_keeps_source_and_restart_finishes(self):
        manifest = self.prepared()
        rename = os.rename
        def interrupted(src, dst):
            if '.stage-' in src:
                raise OSError('publish failure')
            return rename(src, dst)
        with mock.patch.object(archive.os, 'rename', side_effect=interrupted):
            with self.assertRaises(OSError):
                archive.transfer(manifest, str(self.dest))
        self.assertTrue((self.run / 'epoch_02').exists())
        self.assertFalse(Path(archive.archive_path(manifest, str(self.dest))).exists())
        self.assertTrue(Path(archive.transfer(manifest, str(self.dest))['archive_path']).exists())

    def test_sigkill_mid_cleanup_recovers(self):
        manifest, receipt = self.copied()
        request = self.root / 'request.json'
        request.write_text(json.dumps([manifest, receipt]))
        code = '''import sys, os, signal, json
sys.path.insert(0, sys.argv[1])
import checkpoint_archive as a
m, r = json.load(open(sys.argv[2]))
real = os.unlink
def unlink(path, *args, **kwargs):
    real(path, *args, **kwargs)
    if '.archive-removing-' in path:
        os.kill(os.getpid(), signal.SIGKILL)
a.os.unlink = unlink
a.cleanup(m, r)
'''
        result = subprocess.run([sys.executable, '-c', code, str(MODULE.parent), str(request)])
        self.assertEqual(result.returncode, -9)
        self.assertFalse((self.run / 'epoch_02').exists())
        self.assertTrue(list(self.run.glob('.archive-removing-*')))
        self.assertEqual(archive.cleanup(manifest, receipt)['status'], 'removed')
        self.assertFalse(list(self.run.glob('.archive-removing-*')))

    def test_sigkill_after_publication_keeps_source_then_reuses_archive(self):
        manifest = self.prepared()
        request = self.root / 'request.json'
        request.write_text(json.dumps(manifest))
        code = '''import sys, os, signal, json
sys.path.insert(0, sys.argv[1])
import checkpoint_archive as a
m = json.load(open(sys.argv[2]))
real = os.rename
def rename(src, dst):
    real(src, dst)
    if '.stage-' in src:
        os.kill(os.getpid(), signal.SIGKILL)
a.os.rename = rename
a.transfer(m, sys.argv[3])
'''
        result = subprocess.run([sys.executable, '-c', code, str(MODULE.parent), str(request), str(self.dest)])
        self.assertEqual(result.returncode, -9)
        self.assertTrue((self.run / 'epoch_02').exists())
        receipt = archive.transfer(manifest, str(self.dest))
        self.assertEqual(archive.verify_archive(manifest, receipt['archive_path']), receipt)

    def test_actual_file_pool_mismatch_refused(self):
        manifest = self.prepared()
        def checker(path):
            return dict(pool='ddn_ssd' if os.path.isfile(path) else 'ddn_hdd',
                        project_id=7, quota_free_bytes=10**8)
        with self.assertRaises(archive.ArchiveError):
            archive.transfer(manifest, str(self.dest), storage_check=checker,
                             required_pool='ddn_hdd', required_project=7)
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_unverified_pool_or_wrong_project_refused(self):
        manifest = self.prepared()
        for check in (None, lambda p: dict(pool='ddn_hdd', project_id=8, quota_free_bytes=10**8)):
            with self.assertRaises(archive.ArchiveError):
                archive.transfer(manifest, str(self.dest), storage_check=check,
                                 required_pool='ddn_hdd', required_project=7)

    def test_metadata_only_plan_never_hashes_weights(self):
        with mock.patch.object(archive, '_digest', side_effect=AssertionError('large I/O in plan')):
            self.assertEqual([c['epoch'] for c in archive.plan(str(self.run))['candidates']], [2])

    def test_verification_requires_successful_durability_barriers(self):
        manifest, receipt = self.copied()
        with mock.patch.object(archive.os, 'fsync', side_effect=OSError('disk barrier failed')):
            with self.assertRaises(OSError):
                archive.verify_archive(manifest, receipt['archive_path'])

    def test_locking_unsupported_preserves_source(self):
        with mock.patch.object(archive.fcntl, 'flock', side_effect=OSError('unsupported')):
            with self.assertRaises(archive.ArchiveError):
                self.prepared()
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_checkpoint_symlink_and_destination_symlink_refused(self):
        shutil.rmtree(str(self.run / 'epoch_02'))
        (self.run / 'epoch_02').symlink_to('epoch_01')
        self.assertEqual(archive.plan(str(self.run))['candidates'], [])
        (self.run / 'epoch_02').unlink()
        self.epoch(2)
        alias = self.root / 'hdd_alias'
        alias.symlink_to(self.dest)
        with self.assertRaises(archive.ArchiveError):
            archive.transfer(self.prepared(), str(alias))

    def test_manifest_traversal_refused(self):
        manifest = self.prepared()
        manifest['files'][0]['path'] = '../unrelated'
        manifest['manifest_id'] = archive._manifest_id(manifest)
        with self.assertRaises(archive.ArchiveError):
            archive.transfer(manifest, str(self.dest))

    def test_empty_or_missing_rank_rng_refused(self):
        path = self.run / 'epoch_02/accelerate_state/random_states_0.pkl'
        path.write_bytes(b'')
        self.assertEqual(archive.plan(str(self.run))['candidates'], [])
        path.unlink()
        (path.parent / 'random_states_1.pkl').write_bytes(b'rank1')
        self.assertEqual(archive.plan(str(self.run))['candidates'], [])

    def test_existing_conflicting_receipt_never_overwritten(self):
        manifest, receipt = self.copied()
        path = Path(receipt['archive_path']) / '.archive_receipt.json'
        path.write_text('{}')
        with self.assertRaises(archive.ArchiveError):
            archive.transfer(manifest, str(self.dest))
        self.assertEqual(path.read_text(), '{}')

    def test_b6_pending_grad_and_zero_pending_both_complete(self):
        (self.run / 'epoch_02/pending_grads/rank0.safetensors').unlink()
        (self.run / 'epoch_02/pending_grads_rank0.pt').write_bytes(b'gradient')
        self.assertEqual([c['epoch'] for c in archive.plan(str(self.run))['candidates']], [2])
        meta = self.run / 'epoch_02/meta.json'
        value = json.loads(meta.read_text())
        value['pending_micro'] = 0
        meta.write_text(json.dumps(value))
        (self.run / 'epoch_02/pending_grads_rank0.pt').unlink()
        self.assertEqual([c['epoch'] for c in archive.plan(str(self.run))['candidates']], [2])

    def test_sigkill_mid_copy_source_intact_restart(self):
        manifest = self.prepared()
        request = self.root / 'request.json'
        request.write_text(json.dumps(manifest))
        code = '''import sys, os, signal, json
sys.path.insert(0, sys.argv[1])
import checkpoint_archive as a
m = json.load(open(sys.argv[2]))
def copy(src, dst, signature, rate):
    with open(dst, 'wb') as f:
        f.write(b'partial')
        f.flush()
    os.kill(os.getpid(), signal.SIGKILL)
a._copy = copy
a.transfer(m, sys.argv[3])
'''
        result = subprocess.run([sys.executable, '-c', code, str(MODULE.parent), str(request), str(self.dest)])
        self.assertEqual(result.returncode, -9)
        self.assertEqual((self.run / 'epoch_02/model.safetensors').read_bytes(), b'weights\x02')
        receipt = archive.transfer(manifest, str(self.dest))
        self.assertEqual(archive.verify_archive(manifest, receipt['archive_path']), receipt)


if __name__ == '__main__':
    unittest.main()
