"""Offline regressions for the full-checkpoint history proof; no remote writes."""
import copy
import importlib.util
import json
import os
import hashlib
import subprocess
import sys
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

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

    def fixture(self, root, *, validation=False):
        snapshot_step, stop_step = (400, 410) if validation else (210, 220)
        snapshot_name = f'at{snapshot_step}'
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
        for name in ('continuous', 'resumed', snapshot_name):
            run = root / name
            state = run / 'latest_state/accelerate_state'
            state.mkdir(parents=True)
            step = snapshot_step if name == snapshot_name else stop_step
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
            if validation:
                history.entries.append(dict(ordinal=16, origin='observed',
                    position=dict(phase='train', epoch=0, batch_index=6717, view='arm', training=True,
                                  grad_enabled=True, optimizer_step=279),
                    ranks=[dict(rank=rank, inputs=input_signature({
                        'input_ids': torch.zeros((1, 129024), dtype=torch.long),
                        'labels': torch.zeros((1, 3924), dtype=torch.long)}),
                        meta={'row': [str(rank)]}, rng_before=encode_rng(capture_rng()),
                        delta=1 if rank == 3 else 0, activity_delta=0) for rank in range(4)]))
            meta['compile_history_sha256'] = history.save(state.parent / 'compile_history.json', checkpoint=dict(meta))
            (state.parent / 'meta.json').write_text(json.dumps(meta))
            (state.parent / 'identity.json').write_text(json.dumps(identity))
            save_file({'weight': torch.tensor([1., 2.])}, state / 'model.safetensors')
            for file in ['optimizer.bin', 'scheduler.bin'] + [f'random_states_{rank}.pkl' for rank in range(4)]:
                torch.save({'payload': torch.tensor([3.])}, state / file)
        snapshot = root / snapshot_name
        receipt = {'sha256': {str(path.relative_to(snapshot)): proof.probe.sha(path)
                             for path in (snapshot / 'latest_state').rglob('*') if path.is_file()}}
        (snapshot / 'DIAGNOSTIC_SOURCE.json').write_text(json.dumps(receipt))
        for name in ('continuous', 'resumed'):
            run = root / name
            lo = 4801 if name == 'continuous' else snapshot_step * 24 + 1
            for rank in range(4):
                rows = [dict(event='prepared', rank=rank, micro=lo - 1,
                             reducer={'rank': rank, 'world_size': 4, 'backend_name': 'nccl'})]
                for micro in range(lo, stop_step * 24 + 1):
                    loss = proof.HISTORICAL_210_LOSS if micro == 5040 and rank == 0 else 2.0
                    rows.append(dict(event='forward', rank=rank, micro=micro, inputs='same', shapes={},
                                     rng_before={}, rng_after={}, loss=loss,
                                     context={'training': True, 'grad_enabled': True, 'view': 'arm'}))
                    if micro % 24 == 0:
                        rows.append(dict(event='accumulation_boundary', rank=rank, micro=micro, gradients='same', rng={}))
                    if name == 'continuous' and rank == 0 and micro == snapshot_step * 24:
                        rows.append(dict(event='snapshot_published', rank=rank, micro=micro, path=str(snapshot),
                                         checkpoint=dict(epoch=0, step=snapshot_step * 24, global_step=snapshot_step, pending_micro=0),
                                         receipt_sha256=proof.probe.sha(snapshot / 'DIAGNOSTIC_SOURCE.json')))
                rows.append(dict(event='finished', rank=rank, micro=stop_step * 24,
                                 result={'preempted': True, 'global_step': stop_step, 'history': []}, rng={}))
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

    def test_validate400_accepts_complete_checkpoint_and_observed_specialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, validation=True)
            result = proof.compare_pair(root, profile='validate400')
            self.assertTrue(result['passed'])
            self.assertEqual(result['validation400'], dict(checkpoint_step=400, stop_step=410,
                rank3_specialization_slot=6718, rank3_specialization_recorded=True))

    def test_prior_passed_flag_cannot_hide_missing_proof_fields(self):
        with self.assertRaises(ValueError):
            proof.validate_prior_proof({'passed': True})

    def prior_report(self):
        return dict(passed=True, model_equal=True, history_equal=True, different_model_keys=[],
            states_equal={key: True for key in ('optimizer.bin', 'scheduler.bin',
                'random_states_0.pkl', 'random_states_1.pkl', 'random_states_2.pkl', 'random_states_3.pkl')},
            traces={str(rank): {key: [] for key in ('inputs', 'shapes', 'rng_before', 'rng_after',
                'loss', 'context', 'gradients', 'boundary_rng')} for rank in range(4)},
            historical210=dict(expected=3.2567708492279053, actual=3.2567708492279053, equal=True))

    def test_prior_validator_rejects_forged_historical_state_and_rank_evidence(self):
        original = self.prior_report()
        proof.validate_prior_proof(original)
        mutations = [lambda p: p.pop('model_equal'),
            lambda p: p.update(passed='true'), lambda p: p.update(history_equal=False),
            lambda p: p.update(different_model_keys=['weight']),
            lambda p: p['states_equal'].pop('random_states_3.pkl'),
            lambda p: p['states_equal'].update({'optimizer.bin': 'true'}),
            lambda p: p['states_equal'].update({'scheduler.bin': False}),
            lambda p: p['traces'].pop('3'),
            lambda p: p['traces']['3'].pop('boundary_rng'),
            lambda p: p['traces']['2'].update(loss=''),
            lambda p: p['traces']['1'].update(gradients=[5064]),
            lambda p: p['historical210'].update(actual=3.256150245666504),
            lambda p: p['historical210'].pop('expected'),
            lambda p: p['historical210'].update(equal='true'),
            lambda p: p.update(profile='validate400')]
        for index, mutate in enumerate(mutations):
            payload = copy.deepcopy(original)
            mutate(payload)
            with self.subTest(index=index), self.assertRaises(ValueError):
                proof.validate_prior_proof(payload)

    def test_validate400_rejects_removed_or_mutated_rank3_specialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, validation=True)
            paths = [root / name / 'latest_state/compile_history.json' for name in ('continuous', 'resumed')]
            originals = [json.loads(path.read_text()) for path in paths]
            for mutation in ('removed', 'slot', 'update', 'shape', 'rank3_delta'):
                for path, original in zip(paths, originals):
                    payload = copy.deepcopy(original)
                    event = payload['entries'][-1]
                    if mutation == 'removed':
                        payload['entries'].pop()
                    elif mutation == 'slot':
                        event['position']['batch_index'] = 6718
                    elif mutation == 'update':
                        event['position']['optimizer_step'] = 280
                    elif mutation == 'shape':
                        event['ranks'][3]['inputs']['labels']['shape'] = [1, 3923]
                    else:
                        event['ranks'][3]['delta'] = 0
                        event['ranks'][0]['delta'] = 1
                    path.write_text(json.dumps(payload))
                    meta_path = path.with_name('meta.json')
                    meta = json.loads(meta_path.read_text())
                    meta['compile_history_sha256'] = proof.probe.sha(path)
                    meta_path.write_text(json.dumps(meta))
                with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'specialization'):
                    proof.compare_pair(root, profile='validate400')

    def test_validate400_requires_snapshot_and_exact_boundary_rng(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, validation=True)
            path = root / 'resumed/trace_rank3.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            next(row for row in rows if row['event'] == 'accumulation_boundary')['rng'] = {'cpu': 'changed'}
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            self.assertFalse(proof.compare_pair(root, profile='validate400')['passed'])
            path = root / 'continuous/trace_rank0.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows if row['event'] != 'snapshot_published'))
            with self.assertRaisesRegex(ValueError, 'snapshot'):
                proof.compare_pair(root, profile='validate400')

    def test_snapshot_bounds_preserve_legacy_default_and_reject_invalid_positions(self):
        self.assertEqual(proof.snapshot_step(types.SimpleNamespace(stop=220)), 210)
        self.assertEqual(proof.snapshot_step(types.SimpleNamespace(stop=410, snapshot_step=400)), 400)
        for step, stop in ((280, 410), (400, 400), (400, 399), (True, 410)):
            with self.subTest(step=step, stop=stop), self.assertRaises(ValueError):
                proof.snapshot_step(types.SimpleNamespace(stop=stop, snapshot_step=step))

    def test_check_prior_binds_job_id_and_checkpoint_without_loading_model_tensors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            (root / 'comparison.json').write_text(json.dumps(self.prior_report()))
            (root / 'verified_record.json').write_text(json.dumps({'job_id': '3369338'}))
            # No model files remain: a small metadata preflight must still work.
            for path in root.glob('*/latest_state/accelerate_state/model.safetensors'):
                path.unlink()
            proof.check_prior(root, '3369338')
            process = subprocess.run([sys.executable, str(ROOT / 'history_proof.py'), 'check-prior',
                '--root', str(root), '--job-id', '3369338'], capture_output=True, text=True)
            self.assertEqual(process.returncode, 0, process.stderr)
            self.assertEqual(json.loads(process.stdout)['job_id'], '3369338')
            with self.assertRaisesRegex(ValueError, 'job'):
                proof.check_prior(root, '3369339')
            for job_id in ('', 'bad', '3369338;touch /tmp/invalid'):
                with self.subTest(job_id=job_id), self.assertRaises(ValueError):
                    proof.check_prior(root, job_id)
            meta_path = root / 'resumed/latest_state/meta.json'
            meta = json.loads(meta_path.read_text())
            meta['global_step'] = 410
            meta_path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, 'checkpoint'):
                proof.check_prior(root, '3369338')

    def test_checkpoint_digests_bind_complete_state_for_promotion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root)
            result = proof.compare_pair(root)
            self.assertIn('checkpoint_digests', result)
            initial = result['checkpoint_digests']
            self.assertEqual(initial['continuous'], initial['resumed'])
            self.assertRegex(initial['continuous'], r'^[0-9a-f]{64}$')
            scheduler = root / 'resumed/latest_state/accelerate_state/scheduler.bin'
            torch.save({'payload': torch.tensor([7.])}, scheduler)
            changed = proof.compare_pair(root)['checkpoint_digests']
            self.assertEqual(initial['continuous'], changed['continuous'])
            self.assertNotEqual(initial['resumed'], changed['resumed'])

    def test_validation_receipt_binds_verified_source_jobs_and_final_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, prior = base / 'validation', base / 'prior'
            root.mkdir()
            prior.mkdir()
            self.fixture(root, validation=True)
            self.fixture(prior)
            (prior / 'comparison.json').write_text(json.dumps(self.prior_report()))
            (prior / 'verified_record.json').write_text(json.dumps({'job_id': '3369338'}))
            (root / 'verified_record.json').write_text(json.dumps({'job_id': '3369999'}))
            source_file = base / 'SOURCE_COMMIT'
            source_file.write_text('a' * 40 + '\n')
            report = proof.compare_pair(root, profile='validate400')
            comparison = root / 'comparison.json'
            comparison.write_text(json.dumps(report))
            kwargs = dict(source_commit='a' * 40, predecessor_root=prior,
                          predecessor_job_id='3369338', source_commit_file=source_file)
            with patch.dict(os.environ, {'SLURM_JOB_ID': '3369999'}):
                receipt = proof.write_validation_receipt(root, report, **kwargs)
                self.assertEqual(receipt['candidate']['global_step'], 410)
                self.assertEqual(receipt['candidate']['relative_run'], 'resumed')
                self.assertEqual(receipt['candidate']['canonical_checkpoint_sha256'], report['checkpoint_digests']['resumed'])
                self.assertEqual(receipt['validation_job_id'], '3369999')
                self.assertEqual(receipt['predecessor_job_id'], '3369338')
                self.assertEqual(receipt['comparison_sha256'], proof.probe.sha(comparison))
                receipt_path = root / 'validation_receipt.json'
                self.assertEqual(json.loads(receipt_path.read_text()), receipt)
                receipt_path.unlink()
                for mutation in ('passed', 'missing_proof', 'source', 'job', 'meta', 'digest', 'history'):
                    value, options = copy.deepcopy(report), dict(kwargs)
                    if mutation == 'passed':
                        value['passed'] = False
                    elif mutation == 'missing_proof':
                        value.pop('states_equal')
                    elif mutation == 'source':
                        options['source_commit'] = 'b' * 40
                    elif mutation == 'job':
                        options['predecessor_job_id'] = '3369339'
                    elif mutation == 'meta':
                        path = root / 'resumed/latest_state/meta.json'
                        meta = json.loads(path.read_text())
                        meta['global_step'] = 409
                        path.write_text(json.dumps(meta))
                    elif mutation == 'history':
                        history_path = root / 'resumed/latest_state/compile_history.json'
                        original_history = history_path.read_text()
                        meta_path = history_path.with_name('meta.json')
                        original_meta = meta_path.read_text()
                        history = json.loads(original_history)
                        history['entries'][-1]['ranks'][0]['meta']['row'] = ['changed']
                        history_path.write_text(json.dumps(history))
                        meta = json.loads(original_meta)
                        meta['compile_history_sha256'] = proof.probe.sha(history_path)
                        meta_path.write_text(json.dumps(meta))
                    else:
                        value['checkpoint_digests']['resumed'] = '0' * 64
                    with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                        proof.write_validation_receipt(root, value, **options)
                    self.assertFalse(receipt_path.exists())
                    if mutation == 'history':
                        history_path.write_text(original_history)
                        meta_path.write_text(original_meta)
                    if mutation == 'meta':
                        meta['global_step'] = 410
                        path.write_text(json.dumps(meta))
                with patch.dict(os.environ, {'SLURM_JOB_ID': '3370000'}), self.assertRaisesRegex(ValueError, 'job'):
                    proof.write_validation_receipt(root, report, **kwargs)

    def test_batch_preflight_rejects_uncompleted_predecessor_before_storage_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            commands, repo, ops = (base / name for name in ('commands', 'repo', 'ops'))
            for directory in (commands, repo, ops):
                directory.mkdir()
            for directory, manifest in ((repo, 'validation_sources.sha256'), (ops, 'manifest.sha256')):
                (directory / 'payload').write_bytes(b'pinned')
                (directory / manifest).write_text(hashlib.sha256(b'pinned').hexdigest() + '  payload\n')
            seen, forbidden = base / 'sacct_seen', base / 'storage_seen'
            scripts = {'sacct': f'#!/bin/bash\ntouch "{seen}"\nprintf "3369338|PENDING|0:0\\n"\n',
                       'apptainer': '#!/bin/bash\nexit 0\n',
                       'python3': f'#!/bin/bash\ntouch "{forbidden}"\nexit 77\n'}
            for name, content in scripts.items():
                path = commands / name
                path.write_text(content)
                path.chmod(0o755)
            env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
                       REPO_DIR=str(repo), OPS_DIR=str(ops), PROOF_JOB_ID='3369338',
                       SLURM_JOB_ID='3369999', SLURM_GPUS_ON_NODE='4')
            result = subprocess.run(['bash', str(ROOT / 'history_validation.sbatch')], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue(seen.exists(), result.stderr)
            self.assertFalse(forbidden.exists())

    def test_prior_validator_imports_on_python36_grammar_without_training_dependencies(self):
        # A fresh interpreter avoids modules imported by this torch-based test suite.
        script = """
import ast
import builtins
import importlib.util
import json
from pathlib import Path
import sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'accelerate', 'dataclasses'}:
        raise ImportError('blocked module import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
path = Path(sys.argv[1])
for source in (path, path.with_name('probe.py')):
    ast.parse(source.read_text(), filename=str(source), feature_version=(3, 6))
spec = importlib.util.spec_from_file_location('standalone_history_proof', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
payload = json.loads(sys.stdin.read())
module.validate_prior_proof(payload)
for invalid in ({'passed': True}, dict(payload, passed='true')):
    try:
        module.validate_prior_proof(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid pure prior proof accepted')
assert not any(name.split('.')[0] in {'torch', 'numpy', 'accelerate', 'dataclasses'} for name in sys.modules)
print('PURE_PRIOR_PY36_COMPATIBLE')
"""
        result = subprocess.run([sys.executable, '-S', '-c', script, str(ROOT / 'history_proof.py')],
                                input=json.dumps(self.prior_report()), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'PURE_PRIOR_PY36_COMPATIBLE')


if __name__ == '__main__':
    unittest.main()
