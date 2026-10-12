"""Diagnostic acceptance for implemented compiler history; never a production launcher."""
import argparse
import contextlib
import dataclasses
import functools
import hashlib
import importlib.util
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace

_spec = importlib.util.spec_from_file_location('history_proof_base', Path(__file__).with_name('probe.py'))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

HISTORICAL_210_LOSS = 3.2567708492279053
PROFILES = {'resume210': (200, 210, 220), 'validate400': (200, 400, 410)}
STATE_KEYS = {'optimizer.bin', 'scheduler.bin'} | {f'random_states_{rank}.pkl' for rank in range(4)}
TRACE_KEYS = {'inputs', 'shapes', 'rng_before', 'rng_after', 'loss', 'context', 'gradients', 'boundary_rng'}
SLOTS = list(range(1, 16)) + [3915]
PREFIX_SHA256 = {
    '0': 'f6c2eecc10921cf621479e005db1361e61dfc8e81d5aa8c5b6ca676c84650de5',
    '1': 'ff893d0ba22c901d7e4f7e76934d5a041e884584205d2f7224d3abe3c6590ce5',
    '2': '8926cd9db4a34d5fb1d0f825d836e37bf052f3eacb94bcfeef367d99bbf6dfd3',
    '3': 'd7a298d40fb53375a5fcfdc652592134856a40228de874f88528ddbad3fb2ca9',
}


def rng_digest():
    import random
    import numpy as np
    import torch
    return dict(cpu=probe.digest(torch.get_rng_state()), python=probe.digest(random.getstate()),
                numpy=probe.digest(np.random.get_state()),
                cuda=probe.digest(torch.cuda.get_rng_state()) if torch.cuda.is_initialized() else None)


class TrainingTrace:
    """Actual trainer forwards and accumulation gradients, excluding replay work."""
    def __init__(self, path, rank, start_micro):
        self.stream = open(path, 'x', buffering=1)
        self.rank = rank
        self.micro = start_micro
        self.depth = 0
        self.handles = []
        self.before = None

    def emit(self, event, **values):
        self.stream.write(json.dumps(dict(event=event, rank=self.rank, micro=self.micro, **values),
                                     sort_keys=True, allow_nan=False) + '\n')

    def attach(self, model):
        self.handles.extend([model.register_forward_pre_hook(self.pre, with_kwargs=True),
                             model.register_forward_hook(self.post, with_kwargs=True)])

    @contextlib.contextmanager
    def priming(self):
        self.depth += 1
        try:
            yield
        finally:
            self.depth -= 1

    def pre(self, module, positional, inputs):
        import torch
        if self.depth:
            return
        self.micro += 1
        self.before = dict(inputs=probe.digest(inputs), rng_before=rng_digest(),
                           context=dict(training=bool(module.training), grad_enabled=bool(torch.is_grad_enabled()), view='arm'),
                           shapes={key: dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype))
                                   for key, value in inputs.items() if isinstance(value, torch.Tensor)})

    def post(self, module, positional, inputs, output):
        if not self.depth:
            self.emit('forward', **self.before, rng_after=rng_digest(), loss=float(output.loss.detach()))

    def after_backward(self, model):
        if not self.depth and self.micro % 24 == 0:
            gradients = {name: value.grad for name, value in model.named_parameters() if value.grad is not None}
            self.emit('accumulation_boundary', gradients=probe.digest(gradients), rng=rng_digest())

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.stream.close()


def snapshot_step(args):
    selected = getattr(args, 'snapshot_step', 210)
    require(type(selected) is int and selected in (210, 400), 'unsupported snapshot step')
    require(type(args.stop) is int and selected < args.stop, 'snapshot must precede stop')
    return selected


def run(args):
    """Run a bounded real4GPU proof link after this allocation verified all inputs."""
    selected_snapshot = snapshot_step(args)
    import accelerate
    import torch
    from transgenic.training import b6_compile_replay as replay
    trainer = probe.load_trainer()
    payload = json.loads(Path(args.record).read_text())
    require(payload['job_id'] == os.environ['SLURM_JOB_ID'], 'verification record belongs to another allocation')
    require(probe.sha(args.bindings) == payload['bindings_sha256'], 'bindings changed since verification')
    for path, expected in payload['input_stats'].items():
        stat = Path(path).stat()
        require([stat.st_size, stat.st_mtime_ns] == expected, 'verified input changed: ' + path)

    def verified(paths, *, roles, cache, b6_bindings, aug_seed):
        require(paths == payload['paths'] and aug_seed == trainer.rt.AUG_SEED_BASE + 123, 'input identity changed')
        require(str(cache.mmap_dir) == payload['mmap_dir'], 'observation cache path changed')
        trainer.rt.identity_from_bindings(payload['record'], b6_bindings.canonical_record_sha256)
        return dict(payload['record'])

    trainer.verify_training_inputs = verified
    _, _, _, context, _, _ = trainer._wp3()
    original_corpus_init = context.B6Corpus.__init__

    def corpus_init(self, corpus_path, cache, expected_sha256=None, **kwargs):
        require(str(corpus_path) == payload['paths']['corpus'], 'corpus path changed')
        require(expected_sha256 == payload['record']['corpus_sha256'], 'corpus digest changed')
        return original_corpus_init(self, corpus_path, cache, expected_sha256=None, **kwargs)

    context.B6Corpus.__init__ = corpus_init
    start = json.loads((Path(args.run) / 'latest_state/meta.json').read_text())
    rank = int(os.environ['RANK'])
    trace = TrainingTrace(Path(args.run) / f'trace_rank{rank}.jsonl', rank, start['step'])
    prepared = {}
    original_execute = replay.execute_replay

    @functools.wraps(original_execute)
    def execute(*values, **kwargs):
        with trace.priming():
            return original_execute(*values, **kwargs)

    replay.execute_replay = execute
    original_fingerprint = replay.runtime_fingerprint

    @functools.wraps(original_fingerprint)
    def fingerprint(*values, **kwargs):
        result = original_fingerprint(*values, **kwargs)
        require_operating_fingerprint(result)
        return result

    replay.runtime_fingerprint = fingerprint
    original_prepare = accelerate.Accelerator.prepare

    def prepare(self, *objects, **kwargs):
        result = original_prepare(self, *objects, **kwargs)
        model = result[0]
        require(self.num_processes == 4 and isinstance(model, torch.nn.parallel.DistributedDataParallel),
                'actual four-rank DDP required')
        trace.attach(model)
        prepared['model'] = model
        trace.emit('prepared', reducer=model._get_ddp_logging_data())
        return result

    accelerate.Accelerator.prepare = prepare
    original_backward = accelerate.Accelerator.backward

    def backward(self, *values, **kwargs):
        result = original_backward(self, *values, **kwargs)
        trace.after_backward(prepared['model'])
        return result

    accelerate.Accelerator.backward = backward
    original_run = trainer.run_b6

    def bounded(spec, *values, **kwargs):
        return original_run(dataclasses.replace(spec, stop_at_step=args.stop, save_every_n_steps=10,
                                                legacy_compile_bootstrap=args.bootstrap), *values, **kwargs)

    trainer.run_b6 = bounded
    if args.snapshot:
        original_rename = os.rename

        def rename(source, destination, *values, **kwargs):
            result = original_rename(source, destination, *values, **kwargs)
            if (rank == 0 and str(source) == str(Path(args.run) / 'latest_state.tmp') and
                    str(destination) == str(Path(args.run) / 'latest_state')):
                meta = json.loads((Path(destination) / 'meta.json').read_text())
                if meta['global_step'] == selected_snapshot:
                    probe.clone(args.run, args.snapshot)
                    trace.emit('snapshot_published', path=args.snapshot,
                               checkpoint={key: meta[key] for key in ('epoch', 'step', 'global_step', 'pending_micro')},
                               receipt_sha256=probe.sha(Path(args.snapshot) / 'DIAGNOSTIC_SOURCE.json'))
            return result

        os.rename = rename
    try:
        result = trainer.main(['--config', 'configs/b6_400m_win_v3_wp5_pilot.json', '--arm', 'C2', '--seed', '123',
                               '--bindings', args.bindings, '--output-dir', args.run, '--resume', 'auto',
                               '--stop-after-updates', '4099', '--save-every-n-steps', '10',
                               '--mixed-precision', 'bf16', '--deterministic'])
        require(result.get('preempted') and result['global_step'] == args.stop, 'unexpected proof stop: ' + repr(result))
        trace.emit('finished', result=result, rng=rng_digest())
    finally:
        trace.close()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def make_bootstrap(run, receipt, prefix_sha256):
    """Validate the copied committed state against previously retained evidence."""
    run = Path(run)
    checkpoint = run / 'latest_state'
    meta = json.loads((checkpoint / 'meta.json').read_text())
    expected_position = dict(epoch=0, step=4800, global_step=200, pending_micro=0)
    require(all(meta.get(k) == v for k, v in expected_position.items()), 'foreign checkpoint position')
    required = {'meta.json', 'identity.json'} | {
        'accelerate_state/' + name for name in
        ['model.safetensors', 'optimizer.bin', 'scheduler.bin'] + [f'random_states_{r}.pkl' for r in range(4)]}
    declared = {name.removeprefix('latest_state/'): value for name, value in receipt['sha256'].items()
                if name.startswith('latest_state/')}
    actual = set()
    for path in checkpoint.rglob('*'):
        require(not path.is_symlink(), 'symlink in checkpoint')
        if path.is_file():
            actual.add(str(path.relative_to(checkpoint)))
    require(actual == set(declared) == required, 'incomplete or extra checkpoint files')
    for name, digest in declared.items():
        require(probe.sha(checkpoint / name) == digest, 'changed checkpoint file: ' + name)
    require(set(prefix_sha256) == {'0', '1', '2', '3'}, 'incomplete draw prefix evidence')
    for rank in range(4):
        lines = (run / 'draw_log' / f'rank{rank:02d}.tsv').read_bytes().splitlines(keepends=True)
        require(len(lines) >= 4800 and all(line.endswith(b'\n') for line in lines[:4800]), 'short draw prefix')
        require(hashlib.sha256(b''.join(lines[:4800])).hexdigest() == prefix_sha256[str(rank)],
                f'changed draw prefix rank{rank}')
    return dict(version=1, kind='reconstructed_legacy_compile_bootstrap', checkpoint=expected_position,
                world_size=4, checkpoint_files=declared, draw_prefix_lines=4800,
                draw_prefix_sha256=dict(prefix_sha256), slots=list(SLOTS),
                provenance='metadata-derived candidate; not an observed compiler ledger')


def _trace(run, rank, *, start_step, snapshot_step, stop_step):
    rows = [json.loads(line) for line in (run / f'trace_rank{rank}.jsonl').read_text().splitlines()]
    require(all(type(row.get('rank')) is int and row['rank'] == rank for row in rows), 'wrong trace rank')
    prepared = [row for row in rows if row['event'] == 'prepared']
    finished = [row for row in rows if row['event'] == 'finished']
    start = (start_step if run.name == 'continuous' else snapshot_step) * 24
    require(len(prepared) == len(finished) == 1, 'missing/duplicate trace lifecycle records')
    require(rows[0] == prepared[0] and rows[-1] == finished[0], 'trace lifecycle order')
    require(prepared[0]['micro'] == start, 'trace starts at wrong checkpoint')
    reducer = prepared[0].get('reducer', {})
    require(reducer.get('world_size') == 4 and reducer.get('rank') == rank and reducer.get('backend_name') == 'nccl',
            'trace is not four-rank NCCL DDP')
    result = finished[0].get('result', {})
    require(finished[0]['micro'] == stop_step * 24 and result.get('preempted') is True and result.get('global_step') == stop_step
            and result.get('history') == [] and not result.get('done'),
            'trace worker did not complete the required updates')
    snapshots = [row for row in rows if row['event'] == 'snapshot_published']
    if run.name == 'continuous' and rank == 0:
        snapshot = run.parent / f'at{snapshot_step}'
        require(len(snapshots) == 1 and snapshots[0]['micro'] == snapshot_step * 24, 'missing/duplicate trace snapshot')
        expected = dict(epoch=0, step=snapshot_step * 24, global_step=snapshot_step, pending_micro=0)
        require(snapshots[0].get('checkpoint') == expected, 'trace snapshot checkpoint mismatch')
        require(Path(snapshots[0]['path']).name == f'at{snapshot_step}', 'trace snapshot destination mismatch')
        receipt_path = snapshot / 'DIAGNOSTIC_SOURCE.json'
        require(snapshots[0].get('receipt_sha256') == probe.sha(receipt_path), 'trace snapshot receipt mismatch')
        receipt = json.loads(receipt_path.read_text())
        files = {str(path.relative_to(snapshot)) for path in (snapshot / 'latest_state').rglob('*') if path.is_file()}
        declared = {name for name in receipt['sha256'] if name.startswith('latest_state/')}
        require(files == declared, 'trace snapshot receipt file set mismatch')
        for name in declared:
            require(probe.sha(snapshot / name) == receipt['sha256'][name], 'trace snapshot bytes changed')
    else:
        require(not snapshots, 'unexpected trace snapshot')
    require(all(row['event'] in ('prepared', 'forward', 'accumulation_boundary', 'snapshot_published', 'finished')
                for row in rows), 'unexpected trace event')
    result = {}
    for kind in ('forward', 'accumulation_boundary'):
        selected = [row for row in rows if row['event'] == kind]
        indexed = {row['micro']: row for row in selected}
        require(len(indexed) == len(selected), f'duplicate trace {run.name} rank{rank} {kind}')
        result[kind] = indexed
    require(all(row.get('context') == {'training': True, 'grad_enabled': True, 'view': 'arm'}
                for row in result['forward'].values()), 'wrong actual forward context')
    return result


def require_operating_fingerprint(value):
    for key, expected in dict(world_size=4, backend='inductor', mixed_precision='bf16',
                              checkpointing='nonreentrant', deterministic=True).items():
        require(value.get(key) == expected, 'wrong proof operating fingerprint: ' + key)
    cache = value.get('effective_caches', {})
    require(cache.get('fx_local') is True and cache.get('aot_local') is True, 'local FX/AOT caches must be enabled')
    require(value.get('inductor', {}).get('force_disable_caches') is False, 'compiler caches were disabled')
    env = value.get('cache_env', {})
    require(env.get('TORCHINDUCTOR_CACHE_DIR') == '/work/compiler_cache/c2_history_v1/inductor' and
            env.get('TRITON_CACHE_DIR') == '/work/compiler_cache/c2_history_v1/triton', 'wrong shared cache namespace')


def _checkpoint_binding(run, position, *, specialization=False):
    from transgenic.training.b6_compile_history import CompileHistory
    checkpoint = run / 'latest_state'
    meta = json.loads((checkpoint / 'meta.json').read_text())
    require(all(type(meta.get(key)) is int and meta[key] == value for key, value in position.items()),
            f'wrong checkpoint position: {run}')
    history_path = checkpoint / 'compile_history.json'
    require(history_path.is_file() and not history_path.is_symlink(), 'missing history')
    require(meta.get('compile_history_sha256') == probe.sha(history_path), 'unbound history digest')
    history = json.loads(history_path.read_text())
    fingerprint = history.get('fingerprint', {})
    require_operating_fingerprint(fingerprint)
    validated = CompileHistory.load(history_path, meta['compile_history_sha256'],
                                    json.loads((checkpoint / 'identity.json').read_text()), fingerprint,
                                    SimpleNamespace(num_processes=4, process_index=0), checkpoint=position)
    manifest_path = run.parent / 'bootstrap.json'
    manifest = json.loads(manifest_path.read_text())
    require(validated.bootstrap == {'manifest': manifest, 'sha256': probe.sha(manifest_path)},
            'unbound bootstrap manifest')
    require(manifest.get('slots') == SLOTS and manifest.get('kind') == 'reconstructed_legacy_compile_bootstrap',
            'wrong legacy bootstrap manifest')
    entries = validated.entries
    require(len(entries) >= 16 and all(entry['origin'] == 'reconstructed_legacy' for entry in entries[:16]) and
            all(entry['origin'] == 'observed' for entry in entries[16:]), 'missing or repeated bootstrap history')
    expected_positions = [dict(phase='train', epoch=0, batch_index=slot - 1, view='arm', training=True,
                               grad_enabled=True, optimizer_step=(slot - 1) // 24) for slot in SLOTS]
    require([entry['position'] for entry in entries[:16]] == expected_positions,
            'wrong bootstrap history slots')
    if specialization:
        position = dict(phase='train', epoch=0, batch_index=6717, view='arm', training=True,
                        grad_enabled=True, optimizer_step=279)
        events = [entry for entry in entries if entry['origin'] == 'observed' and entry['position'] == position]
        require(len(events) == 1, 'missing/duplicate rank3 specialization at slot6718')
        rank3 = events[0]['ranks'][3]
        require(rank3['delta'] > 0, 'rank3 specialization lacks an observed graph event')
        require(rank3['inputs'].get('input_ids', {}).get('shape') == [1, 129024] and
                rank3['inputs'].get('labels', {}).get('shape') == [1, 3924], 'wrong rank3 specialization shapes')
    return meta['compile_history_sha256']


def _pinned_manifest(root):
    manifest = json.loads((root / 'bootstrap.json').read_text())
    receipt = json.loads(Path(__file__).with_name('original_checkpoint_receipt.json').read_text())
    expected_manifest = dict(version=1, kind='reconstructed_legacy_compile_bootstrap',
        checkpoint=dict(epoch=0, step=4800, global_step=200, pending_micro=0), world_size=4,
        checkpoint_files={name.removeprefix('latest_state/'): digest for name, digest in receipt['sha256'].items()
                          if name.startswith('latest_state/')}, draw_prefix_lines=4800,
        draw_prefix_sha256=PREFIX_SHA256, slots=SLOTS,
        provenance='metadata-derived candidate; not an observed compiler ledger')
    require(json.dumps(manifest, sort_keys=True, allow_nan=False) == json.dumps(expected_manifest, sort_keys=True),
            'bootstrap does not match the pinned original checkpoint and draw evidence')


def compare_pair(root, *, profile='resume210'):
    """Require complete four-rank numerical equality and the original210 scalar."""
    require(profile in PROFILES, 'unsupported proof profile')
    start_step, selected_snapshot, stop_step = PROFILES[profile]
    start_micro, snapshot_micro, stop_micro = (step * 24 for step in PROFILES[profile])
    root = Path(root)
    _pinned_manifest(root)
    left, right = root / 'continuous', root / 'resumed'
    histories = [_checkpoint_binding(run, dict(epoch=0, global_step=stop_step, step=stop_micro, pending_micro=0),
                                     specialization=profile == 'validate400')
                 for run in (left, right)]
    _checkpoint_binding(root / f'at{selected_snapshot}',
                        dict(epoch=0, global_step=selected_snapshot, step=snapshot_micro, pending_micro=0))
    a, b = probe.checkpoint(left), probe.checkpoint(right)
    different = sorted(key for key in set(a['model']) | set(b['model']) if a['model'].get(key) != b['model'].get(key))
    states = {key: a['state'].get(key) == b['state'].get(key) for key in set(a['state']) | set(b['state'])}
    traces = {}
    historical = None
    for rank in range(4):
        coordinates = dict(start_step=start_step, snapshot_step=selected_snapshot, stop_step=stop_step)
        x, y = _trace(left, rank, **coordinates), _trace(right, rank, **coordinates)
        require(set(x['forward']) == set(range(start_micro + 1, stop_micro + 1)), f'incomplete continuous trace rank{rank}')
        require(set(y['forward']) == set(range(snapshot_micro + 1, stop_micro + 1)), f'incomplete resumed trace rank{rank}')
        for rows in (x['forward'], y['forward']):
            require(all(math.isfinite(row['loss']) for row in rows.values()), 'nonfinite trace loss')
        traces[str(rank)] = {key: [micro for micro in range(snapshot_micro + 1, stop_micro + 1)
                                  if x['forward'][micro][key] != y['forward'][micro][key]]
                            for key in ('inputs', 'shapes', 'rng_before', 'rng_after', 'loss', 'context')}
        require(set(x['accumulation_boundary']) == set(range(start_micro + 24, stop_micro + 1, 24)), 'incomplete continuous gradient trace')
        require(set(y['accumulation_boundary']) == set(range(snapshot_micro + 24, stop_micro + 1, 24)), 'incomplete resumed gradient trace')
        traces[str(rank)]['gradients'] = [micro for micro in range(snapshot_micro + 24, stop_micro + 1, 24)
                                        if x['accumulation_boundary'][micro]['gradients'] !=
                                        y['accumulation_boundary'][micro]['gradients']]
        traces[str(rank)]['boundary_rng'] = [micro for micro in range(snapshot_micro + 24, stop_micro + 1, 24)
                                           if x['accumulation_boundary'][micro]['rng'] != y['accumulation_boundary'][micro]['rng']]
        if rank == 0:
            historical = x['forward'][5040]['loss']
    report = dict(profile=profile, checkpoint_history_sha256=dict(zip(('continuous', 'resumed'), histories)),
                  checkpoint_digests={'continuous': probe.digest(a), 'resumed': probe.digest(b)},
                  model_equal=not different, different_model_keys=different, states_equal=states,
                  history_equal=histories[0] == histories[1], traces=traces,
                  historical210=dict(expected=HISTORICAL_210_LOSS, actual=historical,
                                     equal=historical == HISTORICAL_210_LOSS))
    report['passed'] = (not different and set(states) == STATE_KEYS and all(states.values()) and report['history_equal'] and
                        report['historical210']['equal'] and
                        all(not values for rank in traces.values() for values in rank.values()))
    if profile == 'validate400':
        report['validation400'] = dict(checkpoint_step=400, stop_step=410,
                                       rank3_specialization_slot=6718, rank3_specialization_recorded=True)
    return report


def _validate_numerical_proof(report):
    """Require every successful numerical acceptance field with its exact type."""
    require(type(report) is dict, 'prior proof must be a mapping')
    require(all(report.get(key) is True for key in ('passed', 'model_equal', 'history_equal')),
            'prior proof success fields missing or false')
    require(type(report.get('different_model_keys')) is list and report['different_model_keys'] == [],
            'prior proof has model differences')
    states = report.get('states_equal')
    require(type(states) is dict and set(states) == STATE_KEYS and all(value is True for value in states.values()),
            'prior proof state evidence incomplete or false')
    traces = report.get('traces')
    require(type(traces) is dict and set(traces) == {'0', '1', '2', '3'}, 'prior proof ranks incomplete')
    for rank in traces.values():
        require(type(rank) is dict and set(rank) == TRACE_KEYS and
                all(type(value) is list and value == [] for value in rank.values()),
                'prior proof rank mismatches incomplete or nonempty')
    historical = report.get('historical210')
    require(type(historical) is dict and set(historical) == {'expected', 'actual', 'equal'} and
            historical.get('equal') is True and
            all(type(historical.get(key)) in (int, float) and historical[key] == HISTORICAL_210_LOSS
                for key in ('expected', 'actual')), 'prior proof historical210 evidence invalid')
    return report


def validate_prior_proof(report):
    """Validate the complete initial-proof result, including immutable v1 reports."""
    _validate_numerical_proof(report)
    require(report.get('profile', 'resume210') == 'resume210', 'wrong prior proof profile')
    return report


def check_prior(root, job_id):
    """Small receipt/history preflight; never read model or optimizer tensors."""
    require(type(job_id) is str and job_id.isascii() and job_id.isdecimal(), 'job ID must be digits')
    root = Path(root)
    validate_prior_proof(json.loads((root / 'comparison.json').read_text()))
    record = json.loads((root / 'verified_record.json').read_text())
    require(record.get('job_id') == job_id, 'prior verification job ID mismatch')
    _pinned_manifest(root)
    histories = [_checkpoint_binding(root / name, dict(epoch=0, global_step=220, step=5280, pending_micro=0))
                 for name in ('continuous', 'resumed')]
    require(histories[0] == histories[1], 'prior checkpoint histories differ')
    _checkpoint_binding(root / 'at210', dict(epoch=0, global_step=210, step=5040, pending_micro=0))
    return dict(passed=True, job_id=job_id, profile='resume210')


def write_validation_receipt(root, report, *, source_commit, predecessor_root, predecessor_job_id,
                             source_commit_file='/work/repo/SOURCE_COMMIT'):
    """Publish a small promotion receipt only for a complete, bound validation."""
    root = Path(root)
    _validate_numerical_proof(report)
    require(report.get('profile') == 'validate400' and report.get('validation400') ==
            dict(checkpoint_step=400, stop_step=410, rank3_specialization_slot=6718,
                 rank3_specialization_recorded=True), 'missing validation400 acceptance fields')
    digests = report.get('checkpoint_digests')
    require(type(digests) is dict and set(digests) == {'continuous', 'resumed'} and
            all(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) for value in digests.values()) and
            digests['continuous'] == digests['resumed'], 'invalid canonical checkpoint digests')
    comparison = root / 'comparison.json'
    require(json.loads(comparison.read_text()) == report, 'comparison report or digest changed')
    require(type(source_commit) is str and re.fullmatch('[0-9a-f]{40}', source_commit) and
            Path(source_commit_file).read_text().strip() == source_commit, 'source commit mismatch')
    job_id = os.environ.get('SLURM_JOB_ID', '')
    require(job_id.isascii() and job_id.isdecimal(), 'validation job ID must be digits')
    require(json.loads((root / 'verified_record.json').read_text()).get('job_id') == job_id,
            'validation verification job ID mismatch')
    check_prior(predecessor_root, predecessor_job_id)
    position = dict(epoch=0, step=9840, global_step=410, pending_micro=0)
    history_sha = _checkpoint_binding(root / 'resumed', position, specialization=True)
    continuous_history_sha = _checkpoint_binding(root / 'continuous', position, specialization=True)
    require(report.get('checkpoint_history_sha256') ==
            {'continuous': continuous_history_sha, 'resumed': history_sha}, 'checkpoint history changed after comparison')
    _checkpoint_binding(root / 'at400', dict(epoch=0, step=9600, global_step=400, pending_micro=0))
    receipt = dict(version=1, kind='c2_history_validation_ready', source_commit=source_commit,
        validation_job_id=job_id, predecessor_job_id=predecessor_job_id,
        predecessor_comparison_sha256=probe.sha(Path(predecessor_root) / 'comparison.json'),
        comparison_sha256=probe.sha(comparison),
        original_checkpoint_receipt_sha256=probe.sha(Path(__file__).with_name('original_checkpoint_receipt.json')),
        candidate=dict(relative_run='resumed', relative_checkpoint='latest_state', **position,
            meta_sha256=probe.sha(root / 'resumed/latest_state/meta.json'), history_sha256=history_sha,
            canonical_checkpoint_sha256=digests['resumed']))
    with (root / 'validation_receipt.json').open('x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    manifest = commands.add_parser('manifest')
    manifest.add_argument('--run', required=True)
    manifest.add_argument('--receipt', required=True)
    manifest.add_argument('--out', required=True)
    compare = commands.add_parser('compare')
    compare.add_argument('--root', required=True)
    compare.add_argument('--profile', choices=tuple(PROFILES), default='resume210')
    compare.add_argument('--source-commit')
    compare.add_argument('--predecessor-root')
    compare.add_argument('--predecessor-job-id')
    prior = commands.add_parser('check-prior')
    prior.add_argument('--root', required=True)
    prior.add_argument('--job-id', required=True)
    training = commands.add_parser('run')
    for name in ('bindings', 'record', 'run'):
        training.add_argument('--' + name, required=True)
    training.add_argument('--stop', type=int, default=220)
    training.add_argument('--snapshot-step', type=int, choices=(210, 400), default=210)
    training.add_argument('--bootstrap')
    training.add_argument('--snapshot')
    args = parser.parse_args()
    if args.command == 'manifest':
        result = make_bootstrap(args.run, json.loads(Path(args.receipt).read_text()), PREFIX_SHA256)
        with open(args.out, 'x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2)
        return 0
    if args.command == 'run':
        run(args)
        return 0
    if args.command == 'check-prior':
        print(json.dumps(check_prior(args.root, args.job_id)), flush=True)
        return 0
    if args.profile == 'validate400':
        require(all((args.source_commit, args.predecessor_root, args.predecessor_job_id)),
                'validation receipt requires source commit and predecessor identity')
    result = compare_pair(args.root, profile=args.profile)
    probe.write_json(Path(args.root) / 'comparison.json', result)
    if args.profile == 'validate400' and result['passed']:
        write_validation_receipt(args.root, result, source_commit=args.source_commit,
                                 predecessor_root=args.predecessor_root, predecessor_job_id=args.predecessor_job_id)
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
