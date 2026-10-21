"""Real tiny archive transaction regression; no production checkpoint reads."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FixedArchiveTests(unittest.TestCase):
    def test_fixed_epochs_preserve_unrelated_checkpoint_and_reject_false_receipts(self):
        guard = load('operation_guard', HERE / 'archive_guard.py')
        pins = json.loads((HERE / 'pins.json').read_text())
        config = json.loads((HERE / 'archive.json').read_text())
        source = Path(os.environ.get('ARCHIVE_SOURCE_ROOT', '/tmp/b6-checkpoint-archive'))
        with tempfile.TemporaryDirectory(prefix='b6-archive-fixed-') as temporary:
            root = Path(temporary)
            run, dest, ops, helper = [root / name for name in ('run', 'hdd', 'ops', 'helper')]
            for path in (run, dest, ops, helper):
                path.mkdir()
            for name in pins['helper_sha256']:
                target = helper / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(str(source / name), str(target))
            (helper / 'SOURCE_COMMIT').write_text(pins['source_commit'] + '\n')

            def create_epoch(number):
                directory = run / ('epoch_%02d' % number)
                directory.mkdir()
                files = {
                    'model.safetensors': b'weights' + bytes([number]),
                    'accelerate_state/model.safetensors': b'accel',
                    'accelerate_state/optimizer.bin': b'optimizer',
                    'accelerate_state/scheduler.bin': b'scheduler',
                    'accelerate_state/random_states_0.pkl': b'random',
                    'pending_grads/rank0.safetensors': b'gradient',
                    'meta.json': json.dumps(dict(epoch=number, seed=123, global_step=number * 10,
                                                 pending_micro=1)).encode(),
                    'eval.json': json.dumps(dict(epoch=number, global_step=number * 10)).encode(),
                }
                for name, data in files.items():
                    path = directory / name
                    path.parent.mkdir(exist_ok=True)
                    path.write_bytes(data)
                for path in [directory] + list(directory.rglob('*')):
                    os.utime(str(path), (time.time() - 1000, time.time() - 1000), follow_symlinks=False)

            for epoch in range(9, 15):
                create_epoch(epoch)
            (run / 'latest_state').mkdir()
            (run / 'latest_state/meta.json').write_text('{"global_step":149}\n')
            (run / 'trainer_state.json').write_text('{"epoch":14}\n')
            (run / 'best').symlink_to('epoch_14')
            target = config['targets'][0]
            target.update(source_root=str(run), destination_root=str(dest), storage={'kind': 'local'}, min_free_bytes=0)
            (ops / 'archive.json').write_text(json.dumps(config))
            pins.update(helper_dir=str(helper), source_root=str(run), destination_root=str(dest),
                        config_sha256=guard.sha(ops / 'archive.json'),
                        latest_meta_sha256=guard.sha(run / 'latest_state/meta.json'),
                        trainer_state_sha256=guard.sha(run / 'trainer_state.json'))
            core = load('archive_core', helper / 'revision/scripts/b6/checkpoint_archive.py')
            pins['expected_candidates'] = core.plan(str(run), min_age_seconds=300, max_epoch=11,
                                                     max_bytes=48 * 1024 ** 3)['candidates']
            (ops / 'pins.json').write_text(json.dumps(pins))
            guard.HERE = ops
            previous_job = os.environ.get('SLURM_JOB_ID')
            os.environ['SLURM_JOB_ID'] = '123456'
            try:
                original = (run / 'latest_state/meta.json').read_bytes()
                (run / 'latest_state/meta.json').write_text('{"global_step":150}\n')
                with self.assertRaisesRegex(RuntimeError, 'live resume position changed'):
                    guard.main('pre')
                (run / 'latest_state/meta.json').write_bytes(original)
                guard.main('pre')
                # The old generic CLI plan would now choose08 first. Our frozen
                # operation must still touch only the originally pinned09–11.
                create_epoch(8)
                self.assertEqual(core.plan(str(run), max_epoch=11)['candidates'][0]['epoch'], 8)
                for bad in (8, 12):
                    with self.assertRaisesRegex(RuntimeError, 'outside the pinned transfer scope'):
                        guard.main('transfer', bad)
                for epoch in (9, 10, 11):
                    guard.main('transfer', epoch)
                removed = ops / 'epoch_09_removed.json'
                saved = removed.read_bytes()
                corrupted = json.loads(saved)
                corrupted['manifest_id'] = 'wrong'
                removed.write_text(json.dumps(corrupted))
                with self.assertRaisesRegex(RuntimeError, 'cleanup receipt differs'):
                    guard.main('post')
                self.assertFalse((ops / 'complete.json').exists())
                removed.write_bytes(saved)
                guard.main('post')
                complete = json.loads((ops / 'complete.json').read_text())
                self.assertEqual(set(complete['receipts']), {'epoch_09', 'epoch_10', 'epoch_11'})
                for epoch in (8, 12, 13, 14):
                    self.assertTrue((run / ('epoch_%02d' % epoch)).is_dir())
                self.assertEqual((run / 'latest_state/meta.json').read_bytes(), original)
                self.assertEqual(os.readlink(str(run / 'best')), 'epoch_14')
                for name, receipt in complete['receipts'].items():
                    core.verify_archive(receipt['manifest'], receipt['archive_path'])
                    self.assertFalse((run / name).exists())
            finally:
                if previous_job is None:
                    os.environ.pop('SLURM_JOB_ID', None)
                else:
                    os.environ['SLURM_JOB_ID'] = previous_job


if __name__ == '__main__':
    unittest.main()
