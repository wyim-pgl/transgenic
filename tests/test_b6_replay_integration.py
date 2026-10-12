"""Compiler history lifecycle and state-preserving reconstruction regression tests."""
import json
from pathlib import Path
import unittest

import torch
from tests.test_b6_trainer import TrainerBase, rt


class TrainerHistoryTests(TrainerBase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def test_fresh_compiled_checkpoint_binds_history(self):
        spec = self.spec(stop_at_step=1, compile_backend='aot_eager')
        self.train_arm(spec)
        checkpoint = Path(spec.run_dir) / 'latest_state'
        meta = json.loads((checkpoint / 'meta.json').read_text())
        self.assertIn('compile_history_sha256', meta)
        self.assertEqual(rt.file_sha256(checkpoint / 'compile_history.json'), meta['compile_history_sha256'])
        history = json.loads((checkpoint / 'compile_history.json').read_text())
        self.assertTrue(history['entries'])
        self.assertEqual(history['checkpoint'], dict(epoch=0, step=2, global_step=1, pending_micro=0))

    def test_compiled_resume_refuses_missing_ledger(self):
        spec = self.spec(stop_at_step=1, compile_backend='aot_eager')
        self.train_arm(spec)
        checkpoint = Path(spec.run_dir) / 'latest_state'
        (checkpoint / 'compile_history.json').unlink(missing_ok=True)
        meta = json.loads((checkpoint / 'meta.json').read_text())
        meta.pop('compile_history_sha256', None)
        (checkpoint / 'meta.json').write_text(json.dumps(meta))
        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, 'compile history|compiler history'):
            self.train_arm(self.spec(resume='auto', stop_at_step=2, compile_backend='aot_eager'))


    def test_explicit_bootstrap_rejects_fresh_eager_and_existing_ledger(self):
        for backend in (None, 'aot_eager'):
            with self.subTest(backend=backend), self.assertRaisesRegex(RuntimeError, 'compiled checkpoint resume'):
                self.train_arm(self.spec(compile_backend=backend, legacy_compile_bootstrap='missing.json'))
        self.train_arm(self.spec(stop_at_step=1, compile_backend='aot_eager'))
        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, 'cannot override'):
            self.train_arm(self.spec(resume='auto', compile_backend='aot_eager', legacy_compile_bootstrap='missing.json'))


class ResumeHistoryTests(TrainerBase):
    def setUp(self):
        super().setUp()
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
    def test_resume_replays_before_new_rows_and_preserves_training_result(self):
        from unittest.mock import patch
        from transgenic.training import b6_compile_replay as replay
        from tests.test_b6_trainer import syn
        class Counting(syn.SyntheticB6Dataset):
            def __init__(self):
                super().__init__(8, seed=1)
                self.loaded = []
            def __getitem__(self, key):
                self.loaded.append(key)
                return super().__getitem__(key)
        first = self.spec(stop_at_step=1, compile_backend='aot_eager')
        self.train_arm(first)
        checkpoint = Path(first.run_dir) / 'latest_state'
        journal = json.loads((checkpoint / 'compile_history.json').read_text())
        torch._dynamo.reset()
        ds = Counting()
        boundaries = []
        execute = replay.execute_replay
        def record(*args, **kwargs):
            before = rt.state_digest(args[2])
            rng = torch.get_rng_state().clone()
            result = execute(*args, **kwargs)
            self.assertEqual(rt.state_digest(args[2]), before)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            boundaries.append(list(ds.loaded))
            return result
        with patch.object(replay, 'execute_replay', record):
            result = self.train_arm(self.spec(resume='auto', stop_at_step=2, compile_backend='aot_eager'), train_ds=ds)
        self.assertEqual(result['global_step'], 2)
        self.assertEqual(len(boundaries), 1)
        sampler = syn.sampler_for(ds)
        keys = list(sampler)
        expected = [keys[e['position']['batch_index']] for e in journal['entries']]
        self.assertEqual(boundaries[0], expected)
        # Prepared DataLoaderShard prefetches the next batch before yielding.
        self.assertEqual(ds.loaded[len(expected):], keys[2:5])

    def test_epoch_eval_and_fixed_step_history(self):
        spec = self.spec(max_epochs=1, n_pilot=2, compile_backend='aot_eager')
        self.train_arm(spec, n_rows=4)
        for name, position in [('epoch_01', dict(epoch=1, step=0, global_step=2, pending_micro=0)),
                               ('step_00002', dict(epoch=0, step=4, global_step=2, pending_micro=0))]:
            ckpt = Path(spec.run_dir) / name
            meta = json.loads((ckpt / 'meta.json').read_text())
            self.assertEqual(rt.file_sha256(ckpt / 'compile_history.json'), meta['compile_history_sha256'])
            history = json.loads((ckpt / 'compile_history.json').read_text())
            self.assertEqual(history['checkpoint'], position)
            if name == 'epoch_01':
                self.assertTrue(any(e['position']['phase'] == 'eval' for e in history['entries']))
                self.assertTrue(all(not e['position']['training'] and not e['position']['grad_enabled']
                                    for e in history['entries'] if e['position']['phase'] == 'eval'))

    def test_changed_reconstructed_input_is_refused(self):
        from tests.test_b6_trainer import syn
        class Changed(syn.SyntheticB6Dataset):
            def __getitem__(self, key):
                item = super().__getitem__(key)
                item['ids'][0] = (item['ids'][0] + 1) % 4
                return item
        self.train_arm(self.spec(stop_at_step=1, compile_backend='aot_eager'))
        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, 'input signature mismatch'):
            self.train_arm(self.spec(resume='auto', stop_at_step=2, compile_backend='aot_eager'),
                           train_ds=Changed(8, seed=1))

    def test_tampered_history_is_refused_before_state_load(self):
        self.train_arm(self.spec(stop_at_step=1, compile_backend='aot_eager'))
        path = Path(self.tmp.name) / 'run/latest_state/compile_history.json'
        path.write_bytes(path.read_bytes() + b' ')
        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, 'SHA256 mismatch'):
            self.train_arm(self.spec(resume='auto', stop_at_step=2, compile_backend='aot_eager'))

    def test_epoch_resume_keeps_pending_gradient_and_matches_next_update(self):
        original = self.build()
        spec = self.spec(stop_at_step=2, n_pilot=9, accumulation_steps=3, compile_backend='aot_eager')
        self.train_arm(spec, model=original, n_rows=4)
        expected = rt.state_digest(original)
        epoch_dir = Path(spec.run_dir) / 'epoch_01'
        meta = json.loads((epoch_dir / 'meta.json').read_text())
        self.assertEqual(meta['pending_micro'], 1)
        self.assertTrue((epoch_dir / 'pending_grads_rank0.pt').exists())
        torch._dynamo.reset()
        resumed = self.build()
        self.train_arm(self.spec(resume=str(epoch_dir), stop_at_step=2, n_pilot=9, accumulation_steps=3, compile_backend='aot_eager'),
                       model=resumed, n_rows=4)
        self.assertEqual(rt.state_digest(resumed), expected)


class ReplayHelperTests(unittest.TestCase):
    def test_reconstruct_uses_sampler_keys_and_restores_epoch_generator_on_failure(self):
        from accelerate import Accelerator
        from torch.utils.data import DataLoader
        from tests.test_b6_trainer import syn
        from transgenic.training.b6_compile_replay import reconstruct_batch
        class Counting(syn.SyntheticB6Dataset):
            def __init__(self):
                super().__init__(8, seed=1)
                self.loaded = []
            def __getitem__(self, key):
                self.loaded.append(key)
                return super().__getitem__(key)
        ds = Counting()
        sampler = syn.sampler_for(ds)
        generator = torch.Generator().manual_seed(99)
        loader = Accelerator(cpu=True).prepare(DataLoader(ds, batch_size=2, sampler=sampler,
                        collate_fn=syn.synthetic_collate, generator=generator))
        loader.set_epoch(7)
        before = generator.get_state().clone()
        batch = reconstruct_batch(loader, 2, 2, torch.device('cpu'))
        expected = [(i, 2) for i in sampler.permutation(2)[4:6]]
        self.assertEqual(ds.loaded, expected)
        self.assertEqual(batch['meta']['epoch'], [2, 2])
        self.assertEqual(loader.iteration, 7)
        self.assertEqual(sampler.epoch, 7)
        self.assertTrue(torch.equal(generator.get_state(), before))
        with self.assertRaisesRegex(ValueError, 'outside'):
            reconstruct_batch(loader, 0, 50, torch.device('cpu'))
        self.assertEqual(loader.iteration, 7)
        self.assertEqual(sampler.epoch, 7)
    def test_fingerprint_binds_effective_cache_overrides(self):
        import os
        from types import SimpleNamespace
        from accelerate import Accelerator
        from transformers import PretrainedConfig
        import torch._inductor.config as inductor
        from transgenic.training.b6_compile_replay import runtime_fingerprint
        spec = SimpleNamespace(compile_backend='eager', mixed_precision='no', cfg={})
        model = SimpleNamespace(config=PretrainedConfig())
        acc = Accelerator(cpu=True)
        environment = dict(os.environ)
        before = runtime_fingerprint(spec, model, acc, 'none')
        with inductor.patch(fx_graph_cache=not inductor.fx_graph_cache):
            changed = runtime_fingerprint(spec, model, acc, 'none')
        self.assertNotEqual(before['inductor']['fx_graph_cache'], changed['inductor']['fx_graph_cache'])
        self.assertEqual(dict(os.environ), environment)

    def test_fingerprint_distinguishes_default_cache_roots_without_side_effects(self):
        import os
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers import PretrainedConfig
        from transgenic.training.b6_compile_replay import runtime_fingerprint
        from torch._inductor.runtime.cache_dir_utils import default_cache_dir
        spec = SimpleNamespace(compile_backend='inductor', mixed_precision='no', cfg={})
        model = SimpleNamespace(config=PretrainedConfig())
        acc = SimpleNamespace(device=torch.device('cpu'), num_processes=4)
        environment = {key: value for key, value in os.environ.items()
                       if key not in ('TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR')}
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, environment, clear=True):
            fingerprints, defaults = [], []
            for allocation in ('allocation-a', 'allocation-b'):
                candidate = Path(root) / allocation
                with patch.object(tempfile, 'tempdir', str(candidate)), \
                     patch('os.makedirs', side_effect=AssertionError('fingerprinting creates cache directories')), \
                     patch('torch.cuda._lazy_init', side_effect=AssertionError('CPU fingerprint initializes CUDA')):
                    defaults.append(default_cache_dir())
                    fingerprints.append(runtime_fingerprint(spec, model, acc, 'none'))
                self.assertFalse(candidate.exists())
                self.assertEqual(dict(os.environ), environment)
            self.assertNotEqual(defaults[0], defaults[1])
            self.assertNotEqual(fingerprints[0], fingerprints[1])
            for fingerprint, default in zip(fingerprints, defaults):
                self.assertEqual(fingerprint['cache_paths']['inductor'], default)
                self.assertEqual(fingerprint['cache_paths']['triton'],
                                 {'policy': 'inductor_per_device', 'root': os.path.join(default, 'triton'),
                                  'device_component': 'decimal_device_index'})

    def test_fingerprint_binds_explicit_cache_paths_without_changing_them(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers import PretrainedConfig
        from transgenic.training.b6_compile_replay import runtime_fingerprint
        spec = SimpleNamespace(compile_backend='inductor', mixed_precision='no', cfg={})
        model = SimpleNamespace(config=PretrainedConfig())
        acc = SimpleNamespace(device=torch.device('cpu'), num_processes=4)
        with patch.dict(os.environ, {'TORCHINDUCTOR_CACHE_DIR': '/proof/cache/inductor',
                                    'TRITON_CACHE_DIR': '/proof/cache/triton'}):
            before = dict(os.environ)
            fingerprint = runtime_fingerprint(spec, model, acc, 'none')
            self.assertIn('cache_paths', fingerprint)
            self.assertEqual(fingerprint['cache_paths']['inductor'], '/proof/cache/inductor')
            self.assertEqual(fingerprint['cache_paths']['triton'],
                             {'policy': 'explicit', 'directory': '/proof/cache/triton'})
            self.assertEqual(dict(os.environ), before)

    def test_fingerprint_binds_native_triton_explicit_blank_policy(self):
        import os
        from types import SimpleNamespace
        from unittest.mock import patch
        from transformers import PretrainedConfig
        import triton.runtime.cache as triton_cache
        from transgenic.training.b6_compile_replay import runtime_fingerprint
        spec = SimpleNamespace(compile_backend='inductor', mixed_precision='no', cfg={})
        model = SimpleNamespace(config=PretrainedConfig())
        acc = SimpleNamespace(device=torch.device('cpu'), num_processes=4)
        with patch.dict(os.environ, {'TRITON_CACHE_DIR': '', 'TRITON_HOME': '/proof/triton-home'}):
            before = dict(os.environ)
            fingerprint = runtime_fingerprint(spec, model, acc, 'none')
            # Modern Triton preserves empty (its manager later refuses it); legacy
            # Triton uses the native home fallback. Fingerprinting changes neither.
            expected = '' if hasattr(triton_cache, 'knobs') else '/proof/triton-home/.triton/cache'
            self.assertEqual(fingerprint['cache_paths']['triton']['directory'], expected)
            self.assertEqual(dict(os.environ), before)


class TinyReplayModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.register_buffer('visits', torch.tensor(0))
        self.abort = False
    def forward(self, input_ids, **kwargs):
        from types import SimpleNamespace
        self.visits.add_(1)
        if self.abort:
            raise KeyboardInterrupt('replay interrupted')
        return SimpleNamespace(loss=(input_ids.float() * self.weight * torch.rand_like(input_ids.float())).mean())


class ReplayStateIntegrationTests(unittest.TestCase):
    def test_pending_grads_rng_buffers_and_modes_survive_success_and_baseexception(self):
        from accelerate import Accelerator
        from torch.utils.data import DataLoader
        from tests.test_b6_trainer import syn
        from transgenic.training import b6_compile_replay as replay
        from transgenic.training.b6_compile_history import CompileHistory, encode_rng
        from transgenic.training.b6_replay_state import capture_rng
        acc = Accelerator(cpu=True)
        ds = syn.SyntheticB6Dataset(4, seed=1)
        loader = acc.prepare(DataLoader(ds, sampler=syn.sampler_for(ds), collate_fn=syn.synthetic_collate))
        batch = replay.reconstruct_batch(loader, 0, 0, acc.device)
        inputs = replay.replay_inputs(batch, rt.ARMS['B6'], 'arm')
        history = CompileHistory({}, {}, acc)
        history.observe(replay.forward_position('train', 0, 0, 'arm', 0), inputs, batch['meta'], capture_rng(), 0, 0,
                        activity_before=0, activity_after=1)
        original = TinyReplayModel()
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner
            def forward(self, **inputs):
                return self.inner(**inputs)
        model = Wrapper(original).eval()
        parameter = original.weight
        parameter.grad = torch.tensor(17.)
        gradient = parameter.grad
        original.visits.fill_(9)
        before_rng = encode_rng(capture_rng())
        for mode in ('success', 'interrupt', 'memory', 'delta'):
            original.abort = mode == 'interrupt'
            guard = rt.CompileGuard(3, graphs_before=rt.compiled_graph_count())
            args = (history, model, original, {'train': loader}, rt.ARMS['B6'], acc, 2, guard)
            if mode == 'interrupt':
                with self.assertRaises(KeyboardInterrupt):
                    replay.replay_history(*args)
            elif mode == 'memory':
                with self.assertRaisesRegex(RuntimeError, 'memory cap'):
                    replay.replay_history(*args, memory_probe=lambda: rt.MEMORY_ABORT_BYTES + 1)
            elif mode == 'delta':
                history.entries[0]['ranks'][0]['delta'] = 1
                with self.assertRaisesRegex(RuntimeError, 'graph delta'):
                    replay.replay_history(*args)
            else:
                replay.replay_history(*args)
            self.assertIs(original.weight, parameter)
            self.assertIs(parameter.grad, gradient)
            self.assertEqual(float(gradient), 17.)
            self.assertEqual(int(original.visits), 9)
            self.assertFalse(model.training)
            self.assertEqual(encode_rng(capture_rng()), before_rng)


class LegacyManifestTests(unittest.TestCase):
    def fixture(self, root):
        import hashlib
        from transgenic.training import b6_compile_replay as replay
        ckpt, run = root / 'checkpoint', root / 'run'
        names = ['meta.json', rt.CHECKPOINT_IDENTITY, 'accelerate_state/model.safetensors',
                 'accelerate_state/optimizer.bin', 'accelerate_state/scheduler.bin'] + [
                     f'accelerate_state/random_states_{i}.pkl' for i in range(4)]
        for name in names:
            path = ckpt / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        (run / 'draw_log').mkdir(parents=True)
        hashes = {}
        for rank in range(4):
            data = b''.join(f'row{rank}_{i}\t0\t{"a" * 64}\n'.encode() for i in range(4800))
            (run / 'draw_log' / f'rank{rank:02d}.tsv').write_bytes(data + b'newer row\n')
            hashes[str(rank)] = hashlib.sha256(data).hexdigest()
        manifest = dict(version=1, kind='reconstructed_legacy_compile_bootstrap',
                        checkpoint=dict(epoch=0, step=4800, global_step=200, pending_micro=0), world_size=4,
                        checkpoint_files={name: rt.file_sha256(ckpt/name) for name in names},
                        draw_prefix_lines=4800, draw_prefix_sha256=hashes,
                        slots=list(range(1,16)) + [3915], provenance=replay.LEGACY_PROVENANCE)
        return ckpt, run, manifest

    def test_manifest_validates_all_files_and_original_prefix_without_writes(self):
        import tempfile
        from transgenic.training import b6_compile_replay as replay
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt, run, manifest = self.fixture(root)
            path = root / 'bootstrap.json'
            path.write_text(json.dumps(manifest))
            before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            provenance, rows = replay.validate_legacy_manifest(path, ckpt, run, replay.LEGACY_CHECKPOINT,
                                                               arm='C2', world_size=4, batch_size=1)
            self.assertEqual(provenance['manifest'], manifest)
            self.assertEqual(rows['3']['3915'], ['row3_3914', 0, 'a'*64])
            self.assertEqual({str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}, before)

    def test_manifest_rejects_foreign_incomplete_unsafe_or_changed_evidence(self):
        import copy
        import tempfile
        from transgenic.training import b6_compile_replay as replay
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt, run, original = self.fixture(root)
            path = root / 'bootstrap.json'
            mutations = [lambda x: x.update(slots=list(range(1,16))),
                         lambda x: x.update(world_size=1), lambda x: x.update(version=True),
                         lambda x: x['checkpoint'].update(global_step=201),
                         lambda x: x.update(draw_prefix_lines=4799),
                         lambda x: x['draw_prefix_sha256'].update({'2':'b'*64}),
                         lambda x: x['checkpoint_files'].pop('accelerate_state/optimizer.bin'),
                         lambda x: x['checkpoint_files'].update({'../outside':'a'*64}),
                         lambda x: x['checkpoint_files'].update({'meta.json':'b'*64}),
                         lambda x: x.update(provenance='observed')]
            for mutate in mutations:
                manifest = copy.deepcopy(original)
                mutate(manifest)
                path.write_text(json.dumps(manifest))
                with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                    replay.validate_legacy_manifest(path, ckpt, run, replay.LEGACY_CHECKPOINT,
                                                     arm='C2', world_size=4, batch_size=1)
            path.write_text(json.dumps(original))
            (ckpt/'extra').write_text('extra')
            with self.assertRaisesRegex(ValueError, 'complete file set'):
                replay.validate_legacy_manifest(path, ckpt, run, replay.LEGACY_CHECKPOINT,
                                                 arm='C2', world_size=4, batch_size=1)
            (ckpt/'extra').unlink()
            (ckpt/'symlink').symlink_to(ckpt/'meta.json')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                replay.validate_legacy_manifest(path, ckpt, run, replay.LEGACY_CHECKPOINT,
                                                 arm='C2', world_size=4, batch_size=1)

    def test_shared_executor_retains_all_sixteen_legacy_companions(self):
        from accelerate import Accelerator
        from torch.utils.data import DataLoader
        from tests.test_b6_trainer import syn
        from transgenic.training import b6_compile_replay as replay
        from transgenic.training.b6_compile_history import CompileHistory, encode_rng
        from transgenic.training.b6_replay_state import capture_rng
        acc = Accelerator(cpu=True)
        ds = syn.SyntheticB6Dataset(3916, seed=1)
        sampler = syn.sampler_for(ds)
        keys = list(sampler)
        rows = {'0': {}}
        for slot in list(range(1, 16)) + [3915]:
            sample = ds[keys[slot-1]]
            rows['0'][str(slot)] = [sample['row'], sample['epoch'], sample['uniforms_sha256']]
        loader = acc.prepare(DataLoader(ds, sampler=sampler, collate_fn=syn.synthetic_collate))
        model = TinyReplayModel()
        history = CompileHistory({}, {}, acc)
        history.bootstrap = {'manifest': {'kind': 'reconstructed_legacy_compile_bootstrap'}, 'sha256': 'a'*64}
        before = encode_rng(capture_rng())
        replay.execute_replay(history, model, model, {'train':loader}, rt.ARMS['C2'], acc, 24,
                              rt.CompileGuard(3, graphs_before=rt.compiled_graph_count()), legacy_rows=rows)
        self.assertEqual([entry['position']['batch_index']+1 for entry in history.entries], list(range(1,16))+[3915])
        self.assertTrue(all(entry['origin'] == 'reconstructed_legacy' for entry in history.entries))
        self.assertTrue(all(entry['ranks'][0]['delta'] == 0 for entry in history.entries))
        self.assertEqual(history.entries[-1]['position']['optimizer_step'], 163)
        self.assertEqual(encode_rng(capture_rng()), before)
        self.assertEqual(int(model.visits), 0)


def distributed_replay_worker(rank, rendezvous, output, failure):
    import datetime
    from types import SimpleNamespace
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from accelerate.data_loader import prepare_data_loader
    from accelerate.utils import DistributedType
    from tests.test_b6_compile_history import GlooAccelerator
    from tests.test_b6_trainer import syn
    from transgenic.training import b6_compile_replay as replay
    from transgenic.training.b6_compile_history import CompileHistory
    from transgenic.training.b6_replay_state import capture_rng
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=45))
    try:
        acc = GlooAccelerator(rank)
        acc.distributed_type = DistributedType.MULTI_CPU
        acc.project_configuration = SimpleNamespace(automatic_checkpoint_naming=False)
        acc.is_main_process = rank == 0
        acc.backward = lambda loss: loss.backward()
        class Truncated(syn.SyntheticB6Dataset):
            def __len__(self):
                return 3
        ds = Truncated(4, seed=1)
        dl = DataLoader(ds, sampler=syn.sampler_for(ds), collate_fn=syn.synthetic_collate,
                        generator=torch.Generator().manual_seed(123))
        loader = prepare_data_loader(dl, device=acc.device, num_processes=2, process_index=rank)
        original = TinyReplayModel()
        model = torch.nn.parallel.DistributedDataParallel(rt.compile_b6(original, backend='eager'))
        history = CompileHistory({'identity': 'test'}, {'backend': 'eager'}, acc)
        expected_rows = []
        for index in (0, 1):
            batch = replay.reconstruct_batch(loader, 0, index, acc.device)
            inputs = replay.replay_inputs(batch, rt.ARMS['B6'], 'arm')
            before = rt.compiled_graph_count()
            rng = capture_rng()
            model(**inputs).loss.backward()
            # Retain both actual companion slots, including shard padding.
            history.bootstrap = {'test': 'explicit two-rank reconstruction'}
            history.observe(replay.forward_position('train', 0, index, 'arm', 0), inputs, batch['meta'], rng,
                            before, rt.compiled_graph_count(), origin='reconstructed_legacy', force=True)
            expected_rows.append(batch['meta']['row'])
        torch._dynamo.reset()
        original.weight.grad.fill_(7.)
        before_weight = original.weight.detach().clone()
        before_rng = torch.get_rng_state().clone()
        if failure and rank == 1:
            original_getitem = ds.__class__.__getitem__
            class Changed(ds.__class__):
                def __getitem__(self, key):
                    sample = original_getitem(self, key)
                    sample['ids'][0] += 1
                    return sample
            ds.__class__ = Changed
        try:
            replay.replay_history(history, model, original, {'train':loader}, rt.ARMS['B6'], acc, 2,
                                  rt.CompileGuard(3, graphs_before=rt.compiled_graph_count()))
            result = {'ok':True, 'rows':expected_rows,
                      'protected':torch.equal(original.weight, before_weight) and
                                  float(original.weight.grad) == 7. and torch.equal(torch.get_rng_state(), before_rng)}
        except Exception as error:
            result = {'error': str(error)}
        Path(output, f'{rank}.json').write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


class DistributedReplayIntegrationTests(unittest.TestCase):
    def run_workers(self, failure):
        import multiprocessing
        import tempfile
        import time
        with tempfile.TemporaryDirectory() as tmp:
            ctx = multiprocessing.get_context('spawn')
            workers = [ctx.Process(target=distributed_replay_worker, args=(rank, tmp+'/rdzv', tmp, failure))
                       for rank in range(2)]
            try:
                for worker in workers: worker.start()
                deadline = time.monotonic() + 70
                for worker in workers: worker.join(max(0, deadline-time.monotonic()))
                self.assertTrue(all(not w.is_alive() for w in workers), 'distributed replay hung')
                self.assertEqual([w.exitcode for w in workers], [0,0])
                return [json.loads(Path(tmp, f'{rank}.json').read_text()) for rank in range(2)]
            finally:
                for worker in workers:
                    if worker.is_alive(): worker.terminate()
                    worker.join()

    def test_two_rank_replay_preserves_state_with_real_shard_padding(self):
        results = self.run_workers(False)
        self.assertTrue(all(r.get('ok') and r['protected'] for r in results), results)
        self.assertNotEqual(results[0]['rows'][0], results[1]['rows'][0])

    def test_one_rank_changed_input_refuses_replay_collectively(self):
        results = self.run_workers(True)
        self.assertTrue(all('input signature mismatch' in r.get('error','') for r in results), results)


if __name__ == '__main__':
    unittest.main()
