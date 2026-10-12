"""Offline regressions for the full-checkpoint history proof; no remote writes."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

import torch
from safetensors.torch import save_file
from transgenic.training.b6_compile_history import CompileHistory, encode_rng, input_signature
from transgenic.training.b6_replay_state import capture_rng

ROOT = Path(__file__).parent
spec = importlib.util.spec_from_file_location('history_proof', ROOT / 'history_proof.py')
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


class HistoryProofTests(unittest.TestCase):
    def test_trace_excludes_primers_and_records_real_gradients(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.))

            def forward(self, value):
                return types.SimpleNamespace(loss=self.weight * value.sum())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.jsonl'
            model = Model()
            trace = proof.TrainingTrace(path, rank=0, start_micro=23)
            trace.attach(model)
            with trace.priming():
                output = model(value=torch.tensor([3.]))
                output.loss.backward()
                trace.after_backward(model)
            model.zero_grad(set_to_none=True)
            output = model(value=torch.tensor([4.]))
            output.loss.backward()
            trace.after_backward(model)
            trace.close()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row['event'] for row in rows], ['forward', 'accumulation_boundary'])
            self.assertEqual([row['micro'] for row in rows], [24, 24])
            self.assertEqual(rows[0]['loss'], 8.)
            self.assertEqual(rows[0]['context'], {'training': True, 'grad_enabled': True, 'view': 'arm'})
            self.assertEqual(rows[1]['gradients'], proof.probe.digest({'weight': torch.tensor(4.)}))

    def test_bootstrap_binds_complete_original_checkpoint_and_committed_draws(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            state = run / 'latest_state/accelerate_state'
            state.mkdir(parents=True)
            meta = dict(epoch=0, step=4800, global_step=200, pending_micro=0)
            (state.parent / 'meta.json').write_text(json.dumps(meta))
            (state.parent / 'identity.json').write_text('{}')
            for name in ['model.safetensors', 'optimizer.bin', 'scheduler.bin'] + [f'random_states_{r}.pkl' for r in range(4)]:
                (state / name).write_bytes(name.encode())
            receipt = {'sha256': {str(p.relative_to(run)): proof.probe.sha(p)
                                  for p in (run / 'latest_state').rglob('*') if p.is_file()}}
            (run / 'draw_log').mkdir()
            prefixes = {}
            for rank in range(4):
                raw = (f'{rank}\t0\t' + 'a' * 64 + '\n').encode() * 4800
                prefixes[str(rank)] = proof.hashlib.sha256(raw).hexdigest()
                (run / 'draw_log' / f'rank{rank:02d}.tsv').write_bytes(raw + b'failed-tail\n')
            before = {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()}
            manifest = proof.make_bootstrap(run, receipt, prefixes)
            self.assertEqual(manifest['slots'], list(range(1, 16)) + [3915])
            self.assertEqual(manifest['checkpoint'], meta)
            self.assertEqual(len(manifest['checkpoint_files']), 9)
            self.assertEqual(before, {str(p.relative_to(run)): p.read_bytes() for p in run.rglob('*') if p.is_file()})
            (state / 'optimizer.bin').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                proof.make_bootstrap(run, receipt, prefixes)
            (state / 'optimizer.bin').write_bytes(b'optimizer.bin')
            prefixes['0'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'prefix'):
                proof.make_bootstrap(run, receipt, prefixes)

    def fixture(self, root):
        original_receipt = json.loads((ROOT / 'original_checkpoint_receipt.json').read_text())
        manifest = dict(version=1, kind='reconstructed_legacy_compile_bootstrap', slots=list(range(1, 16)) + [3915],
                        checkpoint=dict(epoch=0, step=4800, global_step=200, pending_micro=0), world_size=4,
                        checkpoint_files={name.removeprefix('latest_state/'): digest
                                          for name, digest in original_receipt['sha256'].items() if name.startswith('latest_state/')},
                        draw_prefix_lines=4800, draw_prefix_sha256=dict(proof.PREFIX_SHA256),
                        provenance='metadata-derived candidate; not an observed compiler ledger')
        (root / 'bootstrap.json').write_text(json.dumps(manifest))
        fingerprint = dict(world_size=4, backend='inductor', mixed_precision='bf16', checkpointing='nonreentrant',
                           deterministic=True, effective_caches={'fx_local': True, 'aot_local': True},
                           inductor={'force_disable_caches': False},
                           cache_env={'TORCHINDUCTOR_CACHE_DIR': '/work/compiler_cache/c2_history_v1/inductor',
                                      'TRITON_CACHE_DIR': '/work/compiler_cache/c2_history_v1/triton'})
        identity = {'arm': 'C2'}
        for name in ('continuous', 'resumed', 'at210'):
            run = root / name
            state = run / 'latest_state/accelerate_state'
            state.mkdir(parents=True)
            step = 210 if name == 'at210' else 220
            meta = dict(epoch=0, global_step=step, step=step * 24, pending_micro=0)
            history = CompileHistory(identity, fingerprint, types.SimpleNamespace(num_processes=4, process_index=0))
            history.bootstrap = {'manifest': manifest, 'sha256': proof.probe.sha(root / 'bootstrap.json')}
            for ordinal, slot in enumerate(manifest['slots']):
                history.entries.append(dict(ordinal=ordinal, origin='reconstructed_legacy',
                    position=dict(phase='train', epoch=0, batch_index=slot - 1, view='arm', training=True,
                                  grad_enabled=True, optimizer_step=(slot - 1) // 24),
                    ranks=[dict(rank=rank, inputs=input_signature({'labels': torch.tensor([[1, 2, 3]])}),
                                meta={'row': [str(rank)]}, rng_before=encode_rng(capture_rng()), delta=0, activity_delta=0)
                           for rank in range(4)]))
            meta['compile_history_sha256'] = history.save(state.parent / 'compile_history.json', checkpoint=dict(meta))
            (state.parent / 'meta.json').write_text(json.dumps(meta))
            (state.parent / 'identity.json').write_text(json.dumps(identity))
            save_file({'weight': torch.tensor([1., 2.])}, state / 'model.safetensors')
            for file in ['optimizer.bin', 'scheduler.bin'] + [f'random_states_{rank}.pkl' for rank in range(4)]:
                torch.save({'payload': torch.tensor([3.])}, state / file)
        snapshot = root / 'at210'
        receipt = {'sha256': {str(path.relative_to(snapshot)): proof.probe.sha(path)
                             for path in (snapshot / 'latest_state').rglob('*') if path.is_file()}}
        (snapshot / 'DIAGNOSTIC_SOURCE.json').write_text(json.dumps(receipt))
        for name in ('continuous', 'resumed'):
            run = root / name
            lo = 4801 if name == 'continuous' else 5041
            for rank in range(4):
                rows = [dict(event='prepared', rank=rank, micro=lo - 1,
                             reducer={'rank': rank, 'world_size': 4, 'backend_name': 'nccl'})]
                for micro in range(lo, 5281):
                    loss = proof.HISTORICAL_210_LOSS if micro == 5040 and rank == 0 else 2.0
                    rows.append(dict(event='forward', rank=rank, micro=micro, inputs='same', shapes={},
                                     rng_before={}, rng_after={}, loss=loss,
                                     context={'training': True, 'grad_enabled': True, 'view': 'arm'}))
                    if micro % 24 == 0:
                        rows.append(dict(event='accumulation_boundary', rank=rank, micro=micro, gradients='same', rng={}))
                    if name == 'continuous' and rank == 0 and micro == 5040:
                        rows.append(dict(event='snapshot_published', rank=rank, micro=micro, path=str(snapshot),
                                         checkpoint=dict(epoch=0, step=5040, global_step=210, pending_micro=0),
                                         receipt_sha256=proof.probe.sha(snapshot / 'DIAGNOSTIC_SOURCE.json')))
                rows.append(dict(event='finished', rank=rank, micro=5280,
                                 result={'preempted': True, 'global_step': 220, 'history': []}, rng={}))
                (run / f'trace_rank{rank}.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in rows))

    def test_comparison_detects_weight_optimizer_trace_and_historical_loss_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            self.assertTrue(proof.compare_pair(root)['passed'])
            optimizer = root / 'resumed/latest_state/accelerate_state/optimizer.bin'
            torch.save({'payload': torch.tensor([4.])}, optimizer)
            self.assertFalse(proof.compare_pair(root)['passed'])
            torch.save({'payload': torch.tensor([3.])}, optimizer)
            weight = root / 'resumed/latest_state/accelerate_state/model.safetensors'
            save_file({'weight': torch.tensor([1., 9.])}, weight)
            self.assertFalse(proof.compare_pair(root)['passed'])
            save_file({'weight': torch.tensor([1., 2.])}, weight)
            trace = root / 'resumed/trace_rank2.jsonl'
            original = trace.read_text()
            trace.write_text(original.replace('"inputs": "same"', '"inputs": "different"', 1))
            self.assertFalse(proof.compare_pair(root)['passed'])
            trace.write_text(original)
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
            boundary = next(row for row in rows if row['event'] == 'accumulation_boundary')
            boundary['rng'] = {'cpu': 'changed'}
            trace.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertFalse(proof.compare_pair(root)['passed'])
            trace.write_text(original)
            trace = root / 'continuous/trace_rank0.jsonl'
            trace.write_text(trace.read_text().replace(str(proof.HISTORICAL_210_LOSS), '3.256150245666504'))
            self.assertFalse(proof.compare_pair(root)['passed'])

    def test_comparison_refuses_missing_trace_record_and_unbound_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            p = root / 'resumed/trace_rank3.jsonl'
            original = p.read_text()
            p.write_text('\n'.join([original.splitlines()[0]] + original.splitlines()[2:]) + '\n')
            with self.assertRaisesRegex(ValueError, 'trace'):
                proof.compare_pair(root)
            p.write_text(original)
            p = root / 'resumed/latest_state/compile_history.json'
            p.write_text(p.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'history'):
                proof.compare_pair(root)

    def test_history_schema_and_operating_cache_options_are_acceptance_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / 'resumed/latest_state/compile_history.json'
            meta_path = path.with_name('meta.json')
            original = json.loads(path.read_text())
            for mutation in ('missing_rank', 'disabled_cache', 'wrong_bootstrap_slot', 'wrong_bootstrap_update'):
                value = json.loads(json.dumps(original))
                if mutation == 'missing_rank':
                    value['entries'][0]['ranks'].pop()
                elif mutation == 'disabled_cache':
                    value['fingerprint']['effective_caches']['fx_local'] = False
                elif mutation == 'wrong_bootstrap_slot':
                    value['entries'][0]['position']['batch_index'] = 1
                else:
                    value['entries'][0]['position']['optimizer_step'] = 999
                path.write_text(json.dumps(value))
                meta = json.loads(meta_path.read_text())
                meta['compile_history_sha256'] = proof.probe.sha(path)
                meta_path.write_text(json.dumps(meta))
                reason = {'missing_rank': 'rank', 'disabled_cache': 'cache',
                          'wrong_bootstrap_slot': 'bootstrap', 'wrong_bootstrap_update': 'bootstrap'}[mutation]
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, reason):
                    proof.compare_pair(root)

    def test_manifest_cannot_substitute_a_different_original_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / 'bootstrap.json'
            manifest = json.loads(path.read_text())
            manifest['checkpoint_files']['accelerate_state/model.safetensors'] = 'a' * 64
            path.write_text(json.dumps(manifest))
            for name in ('continuous', 'resumed', 'at210'):
                path = root / name / 'latest_state/compile_history.json'
                payload = json.loads(path.read_text())
                payload['bootstrap'] = {'manifest': manifest, 'sha256': proof.probe.sha(root / 'bootstrap.json')}
                path.write_text(json.dumps(payload))
                meta_path = path.with_name('meta.json')
                meta = json.loads(meta_path.read_text())
                meta['compile_history_sha256'] = proof.probe.sha(path)
                meta_path.write_text(json.dumps(meta))
            snapshot = root / 'at210'
            receipt_path = snapshot / 'DIAGNOSTIC_SOURCE.json'
            receipt = json.loads(receipt_path.read_text())
            receipt['sha256'] = {name: proof.probe.sha(snapshot / name) for name in receipt['sha256']}
            receipt_path.write_text(json.dumps(receipt))
            trace_path = root / 'continuous/trace_rank0.jsonl'
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            next(row for row in rows if row['event'] == 'snapshot_published')['receipt_sha256'] = proof.probe.sha(receipt_path)
            trace_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'original checkpoint'):
                proof.compare_pair(root)

    def test_rank_lifecycle_and_actual_forward_context_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            path = root / 'resumed/trace_rank3.jsonl'
            original = path.read_text()
            for mutation in ('rank', 'prepared', 'finished', 'context'):
                rows = [json.loads(line) for line in original.splitlines()]
                if mutation == 'rank':
                    for row in rows:
                        row['rank'] = 99
                elif mutation in ('prepared', 'finished'):
                    rows = [row for row in rows if row['event'] != mutation]
                else:
                    rows[1]['context']['grad_enabled'] = False
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    proof.compare_pair(root)
            path.write_text(original)
            path = root / 'continuous/trace_rank0.jsonl'
            original = path.read_text()
            rows = [json.loads(line) for line in original.splitlines()]
            snapshot = next(row for row in rows if row['event'] == 'snapshot_published')
            for altered in ([row for row in rows if row['event'] != 'snapshot_published'], rows[:-1] + [snapshot] + rows[-1:]):
                path.write_text(''.join(json.dumps(row) + '\n' for row in altered))
                with self.assertRaisesRegex(ValueError, 'snapshot'):
                    proof.compare_pair(root)


if __name__ == '__main__':
    unittest.main()
