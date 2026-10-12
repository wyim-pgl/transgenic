"""Checkpoint-bound compile history: real descriptors and safe distributed failures."""
import copy
from datetime import timedelta
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
import tempfile
import time
import unittest

import numpy as np
import torch
import torch.distributed as dist

from transgenic.training.b6_compile_history import (
    CompileHistory, check_inputs, decode_rng, encode_rng, input_signature,
)


class LocalAccelerator:
    num_processes = 1
    process_index = 0
    device = torch.device('cpu')


class GlooAccelerator:
    num_processes = 2
    device = torch.device('cpu')

    def __init__(self, rank):
        self.process_index = rank

    def gather(self, tensor):
        rows = [torch.empty_like(tensor) for _ in range(self.num_processes)]
        dist.all_gather(rows, tensor)
        return torch.cat(rows)


def rng_state():
    return {'python': random.Random(12).getstate(),
            'numpy': np.random.RandomState(13).get_state(),
            'torch_cpu': torch.Generator().manual_seed(14).get_state(),
            'torch_cuda': None}


def position(**changes):
    result = dict(phase='train', epoch=0, batch_index=3, view='arm',
                  training=True, grad_enabled=True, optimizer_step=2)
    result.update(changes)
    return result


CHECKPOINT = dict(epoch=0, step=3, global_step=2, pending_micro=0)


def canonical(payload):
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def distributed_worker(rank, rendezvous, output, mode):
    try:
        dist.init_process_group('gloo', init_method='file://' + rendezvous,
                                rank=rank, world_size=2, timeout=timedelta(seconds=20))
        history = CompileHistory({'run': 'test'}, {'compiler': 'test'}, GlooAccelerator(rank))
        labels = torch.tensor([[1, rank + 4, 2]])
        inputs = {'labels': labels}
        if mode == 'hash_error' and rank == 1:
            inputs['unsupported'] = object()
        context = position(training=False, grad_enabled=False, phase='eval', view='de_novo')
        if mode == 'position_error' and rank == 1:
            context['batch_index'] = 4
        try:
            history.observe(context, inputs, {'sample': rank}, rng_state(), 0, 0 if mode == 'activity' else rank,
                            activity_before=0, activity_after=rank if mode == 'activity' else 0)
            if mode not in ('union', 'activity'):
                raise AssertionError('expected collective descriptor rejection')
            history.save(Path(output) / f'rank{rank}.json', checkpoint=CHECKPOINT)
        except (ValueError, RuntimeError) as error:
            if mode in ('union', 'activity'):
                raise
            (Path(output) / f'rank{rank}.error').write_text(str(error))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class CompileHistoryTest(unittest.TestCase):
    def history(self):
        return CompileHistory({'run': 'test'}, {'compiler': 'test'}, LocalAccelerator())

    def populated(self):
        result = self.history()
        result.observe(position(), {'labels': torch.tensor([[1, 4, 2]])},
                       {'samples': [4]}, rng_state(), 0, 1)
        return result

    def test_tensor_value_change_is_rejected(self):
        inputs = {'labels': torch.tensor([[1, 4, 2]])}
        expected = input_signature(inputs)
        inputs['labels'][0, 1] = 5
        self.assertNotEqual(expected, input_signature(inputs))

    def test_input_signature_binds_layout_dtype_shape_and_dynamic_markers(self):
        values = torch.arange(4).reshape(2, 2)
        entry = self.history()
        entry.observe(position(), {'labels': values}, {}, rng_state(), 0, 1)
        for changed in (values.T.contiguous().T, values.float(), values.reshape(4), values + 1):
            with self.subTest(changed=changed):
                with self.assertRaises(ValueError):
                    check_inputs(entry.entries[0], {'labels': changed}, {}, 0)
        marked = values.clone()
        marked._dynamo_dynamic_indices = {1}
        self.assertEqual(input_signature({'labels': marked})['labels']['dynamic_dims'], [1])
        with self.assertRaises(ValueError):
            check_inputs(entry.entries[0], {'labels': marked}, {}, 0)

    def test_scalar_none_and_bfloat16_inputs(self):
        inputs = {'x': torch.tensor([1, 2], dtype=torch.bfloat16), 'optional': None,
                  'flag': True, 'count': 3, 'scale': 1.5, 'name': 'x'}
        history = self.history()
        history.observe(position(), inputs, {}, rng_state(), 0, 1)
        check_inputs(history.entries[0], inputs, {}, 0)
        inputs['flag'] = 1
        with self.assertRaises(ValueError):
            check_inputs(history.entries[0], inputs, {}, 0)
        for value in (object(), [1], float('nan'), torch.sparse_coo_tensor([[0]], [1], [2])):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, TypeError)):
                    input_signature({'x': value})

    def test_rng_roundtrip_reproduces_independently_saved_next_draws(self):
        state = rng_state()
        py = random.Random(); py.setstate(state['python'])
        numpy = np.random.RandomState(); numpy.set_state(state['numpy'])
        cpu = torch.Generator(); cpu.set_state(state['torch_cpu'])
        expected = (py.random(), numpy.normal(size=5), torch.rand(5, generator=cpu))
        restored = decode_rng(json.loads(canonical(encode_rng(state))))
        py.setstate(restored['python']); numpy.set_state(restored['numpy'])
        cpu.set_state(restored['torch_cpu'])
        self.assertEqual(py.random(), expected[0])
        np.testing.assert_array_equal(numpy.normal(size=5), expected[1])
        self.assertTrue(torch.equal(torch.rand(5, generator=cpu), expected[2]))
        self.assertIsNone(restored['torch_cuda'])
        state['torch_cuda'] = [torch.arange(16, dtype=torch.uint8)]
        self.assertTrue(torch.equal(decode_rng(encode_rng(state))['torch_cuda'][0], state['torch_cuda'][0]))

    def test_rng_roundtrip_preserves_gaussian_cache(self):
        state = rng_state()
        py = random.Random(21); py.gauss(0, 1)
        numpy = np.random.RandomState(22); numpy.normal()
        state['python'], state['numpy'] = py.getstate(), numpy.get_state()
        restored = decode_rng(encode_rng(state))
        py2 = random.Random(); py2.setstate(restored['python'])
        np2 = np.random.RandomState(); np2.set_state(restored['numpy'])
        self.assertEqual(py.gauss(0, 1), py2.gauss(0, 1))
        self.assertEqual(numpy.normal(), np2.normal())

    def test_invalid_rng_state_fails_without_changing_global_rng(self):
        initial = torch.get_rng_state().clone()
        for key, value in [('python', (3, (1,), None)), ('numpy', ('MT19937', np.zeros(2, dtype=np.uint32), 0, 0, 0.0)),
                           ('torch_cpu', torch.zeros(2, dtype=torch.uint8)), ('torch_cuda', [torch.zeros(2)]), ('torch_cuda', [torch.tensor([0] * 8 + [1] + [0] * 7, dtype=torch.uint8)])]:
            with self.subTest(key=key):
                bad = rng_state(); bad[key] = value
                with self.assertRaises((ValueError, TypeError, RuntimeError)):
                    encode_rng(bad)
        self.assertTrue(torch.equal(initial, torch.get_rng_state()))

    def test_observe_skips_non_events_and_records_actual_context(self):
        history = self.history()
        self.assertFalse(history.observe(position(), {}, {}, rng_state(), 2, 2))
        self.assertEqual(history.entries, [])
        context = position(phase='eval', training=False, grad_enabled=False, view='de_novo')
        self.assertTrue(history.observe(context, {}, {}, rng_state(), 2, 3))
        self.assertEqual(history.entries[0]['position'], context)
        self.assertEqual(history.entries[0]['ranks'][0]['delta'], 1)

    def test_caller_mutations_cannot_change_recorded_or_returned_payload(self):
        identity, fingerprint = {'run': ['test']}, {'settings': {'dynamic': True}}
        history = CompileHistory(identity, fingerprint, LocalAccelerator())
        meta, context, state = {'samples': [4]}, position(), rng_state()
        history.observe(context, {'x': torch.tensor([1])}, meta, state, 0, 1)
        before = canonical(history.to_payload())
        identity['run'].append('changed'); fingerprint['settings']['dynamic'] = False
        meta['samples'].append(9); context['training'] = False; state['torch_cpu'].zero_()
        payload = history.to_payload(); payload['entries'].clear()
        self.assertEqual(canonical(history.to_payload()), before)

    def test_save_load_binds_digest_identity_fingerprint_and_world(self):
        history = self.populated()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'history.json'; digest = history.save(path, checkpoint=CHECKPOINT)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            restored = CompileHistory.load(path, digest, history.identity, history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)
            self.assertEqual(restored.to_payload(), history.to_payload())
            check_inputs(restored.entries[0], {'labels': torch.tensor([[1, 4, 2]])}, {'samples': [4]}, 0)
            for identity, fingerprint in [({'run': 'other'}, history.fingerprint), (history.identity, {'compiler': 'other'})]:
                with self.assertRaises(ValueError):
                    CompileHistory.load(path, digest, identity, fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)
            for expected in (None, '', '0' * 64):
                with self.assertRaises(ValueError):
                    CompileHistory.load(path, expected, history.identity, history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                CompileHistory.load(path, digest, history.identity, history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)
            path.unlink()
            with self.assertRaises((ValueError, OSError)):
                CompileHistory.load(path, digest, history.identity, history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)

    def test_recomputed_digest_cannot_hide_schema_context_rank_or_rng_damage(self):
        history = self.populated()
        mutations = [lambda p: p['checkpoint'].update(global_step=999), lambda p: p['entries'][0]['position'].update(phase='bad'), lambda p: p.update(version=2), lambda p: p.update(world_size=2),
                     lambda p: p.update(identity={'run': 'bad'}), lambda p: p.update(fingerprint={}),
                     lambda p: p['entries'][0].update(ordinal=1),
                     lambda p: p['entries'].append(copy.deepcopy(p['entries'][0])),
                     lambda p: p['entries'][0]['ranks'].clear(),
                     lambda p: p['entries'][0]['ranks'][0].update(rank=1),
                     lambda p: p['entries'][0]['ranks'][0].update(delta=-1),
                     lambda p: p['entries'][0]['position'].pop('grad_enabled'),
                     lambda p: p['entries'][0]['position'].update(training='yes'),
                     lambda p: p['entries'][0]['ranks'][0]['inputs']['labels'].update(dtype='torch.invented'),
                     lambda p: p['entries'][0]['ranks'][0]['inputs']['labels'].update(shape=[-1]),
                     lambda p: p['entries'][0]['ranks'][0]['inputs']['labels'].update(sha256='bad'),
                     lambda p: p['entries'][0]['ranks'][0]['rng_before'].clear(),
                     lambda p: p['entries'][0].update(origin='reconstructed_legacy')]
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'history.json'
            for index, mutation in enumerate(mutations):
                with self.subTest(index=index):
                    payload = history.to_payload(); payload['checkpoint'] = dict(CHECKPOINT); mutation(payload)
                    raw = canonical(payload); path.write_bytes(raw)
                    with self.assertRaises((ValueError, TypeError, RuntimeError)):
                        CompileHistory.load(path, hashlib.sha256(raw).hexdigest(), history.identity, history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)

    def test_bootstrap_assignment_copies_caller_provenance(self):
        history = self.history()
        provenance = {'source': ['explicit']}
        history.bootstrap = provenance
        history.observe(position(), {}, {}, rng_state(), 0, 0, origin='reconstructed_legacy', force=True)
        provenance['source'].append('changed')
        self.assertEqual(history.to_payload()['bootstrap'], {'source': ['explicit']})

    def test_malformed_cuda_rng_bytes_are_rejected(self):
        state = rng_state()
        state['torch_cuda'] = [torch.zeros(2, dtype=torch.uint8)]
        with self.assertRaises(ValueError):
            encode_rng(state)

    def test_malformed_typed_rng_payloads_fail_closed(self):
        for mutate in (lambda p: p['torch_cpu'].update(data='!bad'),
                       lambda p: p['torch_cpu'].update(shape=[2]),
                       lambda p: p['numpy']['state'].update(dtype='object'),
                       lambda p: p['python'].update(type='pickle'),
                       lambda p: p.update(torch_cuda=[])):
            with self.subTest(mutate=mutate):
                payload = encode_rng(rng_state()); mutate(payload)
                with self.assertRaises(ValueError):
                    decode_rng(payload)

    def test_duplicate_json_fields_fail_closed_even_with_recomputed_digest(self):
        history = self.populated()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'history.json'
            history.save(path, checkpoint=CHECKPOINT)
            raw = path.read_bytes().replace(b'"version":1', b'"version":2,"version":1')
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                CompileHistory.load(path, hashlib.sha256(raw).hexdigest(), history.identity,
                                    history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)

    def test_requires_grad_is_part_of_the_replay_signature(self):
        history = self.history()
        value = torch.tensor([1.0])
        history.observe(position(), {'x': value}, {}, rng_state(), 0, 1)
        value.requires_grad_(True)
        with self.assertRaises(ValueError):
            check_inputs(history.entries[0], {'x': value}, {}, 0)

    def test_checkpoint_fields_and_expected_world_are_strict(self):
        history = self.populated()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'history.json'
            for context in (None, {'global_step': 2}, dict(CHECKPOINT, step=True), dict(CHECKPOINT, pending_micro=-1)):
                with self.subTest(context=context):
                    with self.assertRaises(ValueError):
                        history.save(path, checkpoint=context)
            digest = history.save(path, checkpoint=CHECKPOINT)
            world_two = GlooAccelerator(0)
            with self.assertRaises(ValueError):
                CompileHistory.load(path, digest, history.identity, history.fingerprint, world_two, checkpoint=CHECKPOINT)
            with self.assertRaises(ValueError):
                CompileHistory.load(path, digest, history.identity, history.fingerprint, LocalAccelerator(),
                                    checkpoint=dict(CHECKPOINT, global_step=3))

    def test_activity_counters_require_both_monotonic_values(self):
        history = self.history()
        for before, after in ((1, None), (None, 1), (3, 2), (-1, 0), (True, 1)):
            with self.subTest(before=before, after=after):
                with self.assertRaises(ValueError):
                    history.observe(position(), {}, {}, rng_state(), 0, 0,
                                    activity_before=before, activity_after=after)

    def test_check_inputs_rejects_metadata_and_rank_changes(self):
        entry = self.populated().entries[0]
        with self.assertRaises(ValueError):
            check_inputs(entry, {'labels': torch.tensor([[1, 4, 2]])}, {'samples': [5]}, 0)
        with self.assertRaises(ValueError):
            check_inputs(entry, {'labels': torch.tensor([[1, 4, 2]])}, {'samples': [4]}, 1)

    def test_reconstruction_requires_explicit_origin_and_provenance(self):
        history = self.history()
        with self.assertRaises(ValueError):
            history.observe(position(), {}, {}, rng_state(), 0, 0, force=True)
        with self.assertRaises(ValueError):
            history.observe(position(), {}, {}, rng_state(), 0, 0, origin='reconstructed_legacy', force=True)
        history.bootstrap = {'original_checkpoint_sha256': 'a' * 64, 'proof': 'explicit fixture only'}
        self.assertTrue(history.observe(position(), {}, {}, rng_state(), 0, 0, origin='reconstructed_legacy', force=True))
        self.assertEqual(history.entries[0]['origin'], 'reconstructed_legacy')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'history.json'; digest = history.save(path, checkpoint=CHECKPOINT)
            restored = CompileHistory.load(path, digest, history.identity, history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT)
            self.assertEqual(restored.bootstrap, history.bootstrap)

    def test_empty_history_roundtrip_and_negative_graph_delta_rejection(self):
        history = self.history()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'history.json'; digest = history.save(path, checkpoint=CHECKPOINT)
            self.assertEqual(CompileHistory.load(path, digest, history.identity, history.fingerprint, LocalAccelerator(), checkpoint=CHECKPOINT).entries, [])
        with self.assertRaises(ValueError):
            history.observe(position(), {}, {}, rng_state(), 4, 3)


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo unavailable')
class DistributedCompileHistoryTest(unittest.TestCase):
    def run_workers(self, mode):
        with tempfile.TemporaryDirectory() as root:
            context = multiprocessing.get_context('spawn')
            workers = [context.Process(target=distributed_worker,
                       args=(rank, str(Path(root) / 'rendezvous'), root, mode)) for rank in range(2)]
            try:
                for worker in workers:
                    worker.start()
                deadline = time.monotonic() + 45
                for worker in workers:
                    worker.join(max(0, deadline - time.monotonic()))
                self.assertFalse(any(worker.is_alive() for worker in workers), 'distributed history hung')
                self.assertEqual([worker.exitcode for worker in workers], [0, 0])
                suffix = 'json' if mode in ('union', 'activity') else 'error'
                return [(Path(root) / f'rank{rank}.{suffix}').read_bytes() for rank in range(2)]
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(5)

    def test_rank1_compile_records_both_real_companions_identically(self):
        rows = self.run_workers('union')
        self.assertEqual(rows[0], rows[1])
        payload = json.loads(rows[0]); ranks = payload['entries'][0]['ranks']
        self.assertEqual([r['rank'] for r in ranks], [0, 1])
        self.assertEqual([r['delta'] for r in ranks], [0, 1])
        self.assertEqual([r['meta'] for r in ranks], [{'sample': 0}, {'sample': 1}])
        self.assertNotEqual(ranks[0]['inputs']['labels']['sha256'], ranks[1]['inputs']['labels']['sha256'])
        self.assertFalse(payload['entries'][0]['position']['grad_enabled'])

    def test_rank1_backend_activity_records_event_without_graph_delta(self):
        rows = self.run_workers('activity')
        self.assertEqual(rows[0], rows[1])
        ranks = json.loads(rows[0])['entries'][0]['ranks']
        self.assertEqual([r['delta'] for r in ranks], [0, 0])
        self.assertEqual([r['activity_delta'] for r in ranks], [0, 1])

    def test_rank1_hash_error_propagates_to_both_workers(self):
        rows = self.run_workers('hash_error')
        self.assertTrue(all(b'unsupported' in row.lower() for row in rows))

    def test_rank_position_mismatch_is_rejected_on_both_workers(self):
        rows = self.run_workers('position_error')
        self.assertTrue(all(b'position' in row.lower() for row in rows))


if __name__ == '__main__':
    unittest.main()
