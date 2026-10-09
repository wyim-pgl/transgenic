"""CLI integration tests on disposable files and executable transport boundaries."""
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

HERE = Path(__file__).resolve().parents[1]
CLI = HERE / 'revision/scripts/b6/checkpoint_archive_cli.py'
CORE = CLI.with_name('checkpoint_archive.py')
spec = importlib.util.spec_from_file_location('archive_core_cli_tests', str(CORE))
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "run with 'quotes;$(touch HACK)'"
        self.run.mkdir()
        self.dest = self.root / 'dest with spaces'
        for n in range(1, 5):
            p = self.run / ('epoch_%02d' % n)
            p.mkdir()
            files = {'model.safetensors': b'weights' + bytes([n]),
                     'accelerate_state/model.safetensors': b'accel',
                     'accelerate_state/optimizer.bin': b'optimizer',
                     'accelerate_state/scheduler.bin': b'scheduler',
                     'accelerate_state/random_states_0.pkl': b'rng',
                     "extra 'name;$x'": b'exact bytes'}
            for name, data in files.items():
                path = p / name
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(data)
            (p / 'meta.json').write_text(json.dumps(dict(epoch=n, global_step=n, seed=123, pending_micro=0)))
            (p / 'eval.json').write_text(json.dumps(dict(epoch=n, global_step=n)))
            (p / 'empty').mkdir()
        (self.run / 'best').symlink_to('epoch_01')
        (self.run / 'latest_state').mkdir()
        self.target = dict(name='fixture', transport='local', source_kind='run',
                           source_root=str(self.run), destination_root=str(self.dest),
                           storage=dict(kind='local'), min_age_seconds=0,
                           min_free_bytes=0, max_bytes=10000, max_checkpoints=1)
        self.config = self.root / 'config.json'
        self.env = dict(os.environ)
        self.env['PYTHONDONTWRITEBYTECODE'] = '1'

    def call(self, *args):
        self.assertTrue(CLI.exists(), 'archive CLI implementation is missing')
        self.config.write_text(json.dumps(dict(version=1, targets=[self.target])))
        result = subprocess.run([sys.executable, str(CLI), '--config', str(self.config)] + list(args),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                universal_newlines=True, env=self.env, timeout=10)
        records = [json.loads(line) for line in result.stdout.splitlines()]
        return result, records

    def transport(self):
        binpath = self.root / 'bin'
        binpath.mkdir()
        ssh = binpath / 'ssh'
        ssh.write_text('#!' + sys.executable + '\n' + '''import os, subprocess, sys
if os.environ.get('FAIL_SSH'):
    sys.exit(255)
if os.environ.get('SLEEP_SSH'):
    import time; time.sleep(3)
sys.exit(subprocess.call(sys.argv[-1], shell=True))
''')
        rsync = binpath / 'rsync'
        rsync.write_text('#!' + sys.executable + '\n' + '''import os, shutil, sys
source = sys.argv[-2].split(':', 1)[1].rstrip('/')
dest = sys.argv[-1].rstrip('/')
for root, dirs, names in os.walk(source):
    rel = os.path.relpath(root, source)
    out = os.path.join(dest, rel)
    os.makedirs(out, exist_ok=True)
    for name in names:
        if name == '.archive_receipt.json': continue
        src = os.path.join(root, name)
        if os.path.islink(src): continue
        shutil.copyfile(src, os.path.join(out, name))
if os.environ.get('CORRUPT_RSYNC'):
    with open(os.path.join(dest, 'model.safetensors'), 'wb') as stream: stream.write(b'bad')
if os.environ.get('MUTATE_SOURCE'):
    with open(os.path.join(source, 'model.safetensors'), 'wb') as stream: stream.write(b'changed')
if os.environ.get('FAIL_RSYNC'): sys.exit(23)
''')
        ssh.chmod(0o755)
        rsync.chmod(0o755)
        self.env['PATH'] = str(binpath) + os.pathsep + self.env['PATH']
        self.target.update(transport='ssh-pull', remote=dict(host='fixture', python=sys.executable,
                           helper=str(CLI), timeout=5, connect_timeout=2), rsync_timeout=5)

    def test_default_plan_changes_no_source_or_destination(self):
        before = sorted(str(p.relative_to(self.run)) for p in self.run.rglob('*'))
        result, records = self.call()
        self.assertEqual(result.returncode, 0, result.stderr)
        plan = next(r for r in records if r['status'] == 'planned')
        self.assertEqual([r['epoch'] for r in plan['plan']['candidates']], [2])
        self.assertEqual(plan['plan']['protected'], [1, 3, 4])
        self.assertFalse(self.dest.exists())
        self.assertEqual(before, sorted(str(p.relative_to(self.run)) for p in self.run.rglob('*')))

    def test_copy_remove_and_repeated_idempotent_watch(self):
        result, records = self.call('--apply')
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = next(r['receipt'] for r in records if r['status'] == 'verified')
        saved = Path(receipt['archive_path'])
        self.assertEqual((saved / "extra 'name;$x'").read_bytes(), b'exact bytes')
        self.assertTrue((saved / 'empty').is_dir())
        self.assertTrue((self.run / 'epoch_02').exists())
        result, records = self.call('--apply', '--remove-source', '--watch', '--cycles', '2', '--interval', '0.01')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('removed', [r['status'] for r in records])
        self.assertFalse((self.run / 'epoch_02').exists())
        for name in ('epoch_01', 'epoch_03', 'epoch_04', 'best', 'latest_state'):
            self.assertTrue((self.run / name).exists())

    def test_remove_requires_apply_and_strict_limits(self):
        for args in [('--remove-source',), ('--watch',), ('--watch', '--cycles', '0')]:
            result, records = self.call(*args)
            self.assertNotEqual(result.returncode, 0)
        self.target['max_bytes'] = 0
        result, _ = self.call('--apply')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.dest.exists())

    def test_missing_source_waits_without_creation(self):
        self.target['source_root'] = str(self.root / 'missing')
        result, records = self.call('--apply')
        self.assertEqual(result.returncode, 0)
        self.assertIn('waiting', [r['status'] for r in records])
        self.assertFalse(self.dest.exists())

    def test_pull_exact_bytes_and_source_cleanup(self):
        self.transport()
        result, records = self.call('--apply', '--remove-source')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        receipt = next(r['receipt'] for r in records if r['status'] == 'verified')
        self.assertEqual((Path(receipt['archive_path']) / "extra 'name;$x'").read_bytes(), b'exact bytes')
        self.assertFalse((self.run / 'epoch_02').exists())
        self.assertFalse((self.root / 'HACK').exists())

    def test_pull_auth_corruption_interruption_and_mutation_leave_source(self):
        self.transport()
        for key in ('FAIL_SSH', 'CORRUPT_RSYNC', 'FAIL_RSYNC', 'MUTATE_SOURCE'):
            with self.subTest(key=key):
                self.env[key] = '1'
                result, records = self.call('--apply', '--remove-source')
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn('error', [r['status'] for r in records])
                self.assertTrue((self.run / 'epoch_02').exists())
                self.env.pop(key)
                # Corrupt stages are safely recoverable, but each fault gets fresh destination.
                shutil.rmtree(str(self.dest), ignore_errors=True)

    def test_pull_host_option_injection_is_rejected(self):
        self.transport()
        self.target['remote']['host'] = '-oProxyCommand=touch HACK'
        result, _ = self.call('--apply', '--remove-source')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.dest.exists())

    def test_timeout_leaves_source(self):
        self.transport()
        self.env['SLEEP_SSH'] = '1'
        self.target['remote']['timeout'] = 0.1
        result, records = self.call('--apply', '--remove-source')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_low_space_and_unknown_fields_fail_closed(self):
        self.target['min_free_bytes'] = 10**30
        result, records = self.call('--apply', '--remove-source')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.run / 'epoch_02').exists())
        self.target['min_free_bytes'] = 0
        self.target['unknown_policy'] = 'ignored?'
        result, _ = self.call()
        self.assertNotEqual(result.returncode, 0)


    def test_remote_killed_cleanup_is_discovered_and_resumed(self):
        self.transport()
        helper = self.root / "helper 'quoted'.py"
        helper.write_text("import sys, os, signal, runpy\nsys.path.insert(0, " + repr(str(CLI.parent)) + ")\n" +
                          "import checkpoint_archive as core\noriginal = core.os.unlink\n" +
                          "def kill(path, *args, **kwargs):\n    result = original(path, *args, **kwargs)\n" +
                          "    if '.archive-removing-' in str(path): os.kill(os.getpid(), signal.SIGKILL)\n    return result\n" +
                          "core.os.unlink = kill\nrunpy.run_path(" + repr(str(CLI)) + ", run_name='__main__')\n")
        self.target['remote']['helper'] = str(helper)
        result, records = self.call('--apply', '--remove-source')
        self.assertNotEqual(result.returncode, 0, result.stdout)
        pending = core.pending_cleanup(str(self.run))
        self.assertEqual(len(pending), 1)
        self.assertFalse((self.run / 'epoch_02').exists())
        receipt = pending[0]['receipt']
        self.assertEqual(core.verify_archive(pending[0]['manifest'], receipt['archive_path']), receipt)
        self.target['remote']['helper'] = str(CLI)
        result, records = self.call()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(len(next(r['pending'] for r in records if r['status'] == 'planned')), 1)
        self.assertEqual(len(core.pending_cleanup(str(self.run))), 1)
        self.target['max_bytes'] = 1
        result, records = self.call('--apply', '--remove-source')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('deferred', [r['status'] for r in records])
        self.assertEqual(len(core.pending_cleanup(str(self.run))), 1)
        self.target['max_bytes'] = 10000
        result, records = self.call('--apply', '--remove-source')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual(core.pending_cleanup(str(self.run)), [])
        self.assertFalse(list(self.run.glob('.archive-removing-*')))

    def test_stale_ack_and_eof_before_ack_keep_source(self):
        manifest = core.prepare(str(self.run), 2)
        receipt = core.transfer(manifest, str(self.dest))
        request = dict(action='cleanup', source_kind='run', source_root=str(self.run),
                       manifest_id=manifest['manifest_id'], receipt=receipt, ack_timeout=2)
        for stale in (True, False):
            process = subprocess.Popen([sys.executable, str(CLI), '--helper'], stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
            process.stdin.write((json.dumps(request) + '\n').encode()); process.stdin.flush()
            challenge = json.loads(process.stdout.readline())
            self.assertEqual(challenge['status'], 'ack_required')
            if stale:
                ack = dict(nonce='0'*32, manifest_id=manifest['manifest_id'], receipt=receipt)
                process.stdin.write((json.dumps(ack) + '\n').encode()); process.stdin.flush()
            process.stdin.close(); process.stdin = None
            out, err = process.communicate(timeout=5)
            self.assertNotEqual(process.returncode, 0, out)
            self.assertTrue((self.run / 'epoch_02').exists())
            self.assertEqual(core.pending_cleanup(str(self.run)), [])

    def test_apply_budget_and_plan_checkpoint_cap(self):
        (self.run / 'best').unlink()
        self.target['max_checkpoints'] = 1
        result, records = self.call()
        self.assertEqual(result.returncode, 0)
        candidates = next(r['plan']['candidates'] for r in records if r['status'] == 'planned')
        self.assertEqual([c['epoch'] for c in candidates], [1])
        result, records = self.call('--apply', '--remove-source')
        self.assertEqual(result.returncode, 0)
        self.assertFalse((self.run / 'epoch_01').exists())
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_explicit_receipt_backed_archive_forwarding(self):
        manifest = core.prepare(str(self.run), 2)
        receipt = core.transfer(manifest, str(self.root / 'upstream'))
        self.target.update(source_kind='archive', source_root=receipt['archive_path'])
        self.transport()
        result, records = self.call('--apply', '--remove-source')
        self.assertEqual(result.returncode, 0, result.stdout)
        output = next(r['receipt'] for r in records if r['status'] == 'verified')
        self.assertEqual(output['manifest']['upstream_receipt'], receipt)
        self.assertFalse(Path(receipt['archive_path']).exists())
        self.assertTrue((self.run / 'epoch_02').exists())
        self.assertEqual(core.verify_archive(output['manifest'], output['archive_path']), output)

    def test_missing_archive_receipt_cannot_forward_run_epoch(self):
        self.target.update(source_kind='archive', source_root=str(self.run / 'epoch_02'))
        result, records = self.call('--apply', '--remove-source')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.run / 'epoch_02').exists())

    def test_metadata_only_plan_cannot_read_checkpoint_binaries(self):
        self.config.write_text(json.dumps(dict(version=1, targets=[self.target])))
        script = "import sys,runpy;sys.path.insert(0," + repr(str(CLI.parent)) + ");import checkpoint_archive as c;" + \
                 "c._digest=lambda *a,**k: (_ for _ in ()).throw(AssertionError('binary read'));" + \
                 "sys.argv=[" + repr(str(CLI)) + ",'--config'," + repr(str(self.config)) + "];runpy.run_path(" + repr(str(CLI)) + ",run_name='__main__')"
        result = subprocess.run([sys.executable, '-c', script], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=self.env, universal_newlines=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.dest.exists())

    def test_worker_conflict_and_destination_corruption_leave_source(self):
        result, records = self.call('--apply')
        self.assertEqual(result.returncode, 0)
        receipt = next(r['receipt'] for r in records if r['status'] == 'verified')
        (Path(receipt['archive_path']) / 'model.safetensors').write_bytes(b'bad')
        result, records = self.call('--apply', '--remove-source')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((self.run / 'epoch_02').exists())
        import fcntl
        with open(str(self.dest / '.checkpoint_archive.worker.lock'), 'r+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result, records = self.call('--apply', '--remove-source')
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((self.run / 'epoch_02').exists())


class ReceivedTests(unittest.TestCase):
    setUp = CLITests.setUp
    call = CLITests.call
    def test_received_publish_rehashes_and_recovers_private_stage(self):
        self.assertTrue(hasattr(core, 'publish_received'), 'received publication API is missing')
        manifest = core.prepare(str(self.run), 2)
        def receive(stage):
            for p in (self.run / 'epoch_02').rglob('*'):
                out = Path(stage) / p.relative_to(self.run / 'epoch_02')
                if p.is_dir(): out.mkdir(parents=True, exist_ok=True)
                else:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(str(p), str(out))
        receipt = core.publish_received(manifest, str(self.dest), receive)
        self.assertEqual(core.verify_archive(manifest, receipt['archive_path']), receipt)
        self.assertEqual(core.publish_received(manifest, str(self.dest), receive), receipt)
        (Path(receipt['archive_path']) / 'model.safetensors').write_bytes(b'corrupt')
        with self.assertRaises(core.ArchiveError):
            core.publish_received(manifest, str(self.dest), receive)

    def test_pending_cleanup_discovery_and_restart(self):
        self.assertTrue(hasattr(core, 'pending_cleanup'), 'pending discovery API is missing')
        manifest = core.prepare(str(self.run), 2)
        receipt = core.transfer(manifest, str(self.dest))
        unlink = core.os.unlink
        def fail(path, *args, **kwargs):
            if '.archive-removing-' in str(path): raise OSError('interrupted removal')
            return unlink(path, *args, **kwargs)
        with mock.patch.object(core.os, 'unlink', side_effect=fail):
            with self.assertRaises(OSError): core.cleanup(manifest, receipt)
        pending = core.pending_cleanup(str(self.run))
        self.assertEqual([t['manifest'] for t in pending], [manifest])
        self.assertFalse((self.run / 'epoch_02').exists())
        result, records = self.call('--apply', '--remove-source')
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn('removed', [r['status'] for r in records])
        self.assertEqual(core.pending_cleanup(str(self.run)), [])


if __name__ == '__main__':
    unittest.main()
