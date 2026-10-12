"""Diagnostic only: real C2 trainer, immutable inputs, fresh-process resume comparison."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with open(path, 'x') as f:
        json.dump(value, f, sort_keys=True, indent=2)
        f.write('\n')


def clone(source, target):
    """Copy only committed state/prefixes; preserve every source and verify destination bytes."""
    source, target = Path(source), Path(target)
    assert not target.exists(), target
    meta = json.loads((source / 'latest_state/meta.json').read_text())
    assert meta['pending_micro'] == 0 and meta['step'] == meta['global_step'] * 24
    target.mkdir()
    records = {}
    files = list((source / 'latest_state').rglob('*'))
    files += [source / n for n in ('run_config_b6.json', 'runtime_identity.txt', 'data_order.jsonl')]
    for p in files:
        if p.is_dir():
            continue
        assert p.is_file() and not p.is_symlink(), p
        rel = p.relative_to(source)
        q = target / rel
        q.parent.mkdir(parents=True, exist_ok=True)
        before = p.stat()
        shutil.copy2(p, q)
        digest = sha(p)
        assert sha(q) == digest and p.stat().st_mtime_ns == before.st_mtime_ns, p
        records[str(rel)] = digest
    (target / 'draw_log').mkdir()
    for rank in range(4):
        n = 'rank%02d.tsv' % rank
        lines = (source / 'draw_log' / n).read_bytes().splitlines(keepends=True)
        assert len(lines) >= meta['step'] and all(x.endswith(b'\n') for x in lines)
        (target / 'draw_log' / n).write_bytes(b''.join(lines[:meta['step']]))
    write_json(target / 'DIAGNOSTIC_SOURCE.json', dict(source=str(source), meta=meta, sha256=records))


def load_trainer():
    spec = importlib.util.spec_from_file_location('c2_actual_trainer', '/work/repo/train/train_HyenaTransgenic_B6.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree(value):
    """Canonical value tree, including exact tensor bytes (not serialization order)."""
    import torch
    import numpy as np
    if isinstance(value, torch.Tensor):
        x = value.detach().cpu().contiguous()
        h = hashlib.sha256(memoryview(x.reshape(-1).view(torch.uint8).numpy())).hexdigest()
        return dict(tensor=True, shape=list(x.shape), dtype=str(x.dtype), sha256=h)
    if isinstance(value, np.ndarray):
        x = np.ascontiguousarray(value)
        return dict(ndarray=True, shape=list(x.shape), dtype=str(x.dtype), sha256=hashlib.sha256(memoryview(x)).hexdigest())
    if isinstance(value, dict):
        return {str(type(k).__name__) + ':' + str(k): tree(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (tuple, list)):
        return dict(sequence=type(value).__name__, values=[tree(v) for v in value])
    if isinstance(value, np.generic):
        return value.item()
    assert value is None or isinstance(value, (str, int, float, bool)), type(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(tree(value), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def rng():
    import torch
    import numpy as np
    return dict(cpu=digest(torch.get_rng_state()), cuda=digest(torch.cuda.get_rng_state()),
                python=digest(random.getstate()), numpy=digest(np.random.get_state()))


def verify(args):
    t = load_trainer()
    paths = json.loads(Path(args.bindings).read_text())
    _, _, bd, _, oc, roles = t._wp3()
    cache = oc.ObsCache(paths['cache'])
    record = t.verify_training_inputs(paths, roles=roles.load_roles(paths['roles']), cache=cache,
                                     b6_bindings=bd, aug_seed=t.rt.AUG_SEED_BASE + 123)
    ident = json.loads((Path(args.source) / 'latest_state/identity.json').read_text())
    binding = json.loads(json.dumps(t.rt.identity_from_bindings(record, bd.canonical_record_sha256)))
    for k, v in binding.items():
        assert ident[k] == v, k
    # Record metadata of all input files, including prebuilt cache shards. Every phase rechecks these.
    stats = {}
    inventory_paths = dict(paths, actual_mmap_cache=str(cache.mmap_dir))
    assert Path(cache.mmap_dir).is_dir(), 'required prebuilt cache missing'
    for name, value in inventory_paths.items():
        vals = value.values() if isinstance(value, dict) else [value]
        for val in vals:
            if not isinstance(val, str):
                continue
            p = Path(val)
            candidates = p.rglob('*') if p.is_dir() else [p]
            for q in candidates:
                if q.is_file():
                    s = q.stat()
                    stats[str(q)] = [s.st_size, s.st_mtime_ns]
    write_json(args.record, dict(record=record, paths=paths, input_stats=stats, mmap_dir=str(cache.mmap_dir),
                                bindings_sha256=sha(args.bindings), verified_unix=time.time(),
                                job_id=os.environ['SLURM_JOB_ID']))
    # Canonical sampler check against every saved row; includes r0 row5040 for the loss comparison.
    train_rns, _, _ = t.corpus_rows(paths['corpus'])
    order = list(t.rt.b5_data_order(train_rns, 123, 0))
    report = {}
    for dirname in (args.source, '/work/runs/b6_wp5_pilot_C2_recovery_20261011'):
        for rank in range(4):
            lines = (Path(dirname) / 'draw_log' / ('rank%02d.tsv' % rank)).read_text().splitlines()
            rows = [int(x.split('\t')[0]) for x in lines]
            expected = order[rank::4][:len(rows)]
            assert rows == expected, (dirname, rank, 'canonical order mismatch')
            report[dirname + '/rank%d' % rank] = dict(rows=len(rows), canonical_order_equal=True,
                                                    expected_row5040=order[(5040 - 1) * 4 + rank])
    write_json(str(Path(args.record).with_name('canonical_order.json')), report)
    print('VERIFIED_INPUT_AND_ORDER', record['record_sha256'], flush=True)


def run(args):
    import torch
    import accelerate
    t = load_trainer()
    payload = json.loads(Path(args.record).read_text())
    assert payload['job_id'] == os.environ['SLURM_JOB_ID'], 'record must be freshly verified within this allocation'
    assert sha(args.bindings) == payload['bindings_sha256']
    for p, old in payload['input_stats'].items():
        s = Path(p).stat()
        assert [s.st_size, s.st_mtime_ns] == old, p

    def verified(paths, *, roles, cache, b6_bindings, aug_seed):
        assert paths == payload['paths'] and aug_seed == t.rt.AUG_SEED_BASE + 123
        assert str(cache.mmap_dir) == payload['mmap_dir']
        t.rt.identity_from_bindings(payload['record'], b6_bindings.canonical_record_sha256)
        return dict(payload['record'])
    t.verify_training_inputs = verified
    # B6Corpus normally repeats the already completed corpus hash. This exception is diagnostic-only.
    _, _, _, context, _, _ = t._wp3()
    original_corpus_init = context.B6Corpus.__init__
    def corpus_init(self, corpus_path, cache, expected_sha256=None, **kwargs):
        assert str(corpus_path) == payload['paths']['corpus']
        assert expected_sha256 == payload['record']['corpus_sha256']
        return original_corpus_init(self, corpus_path, cache, expected_sha256=None, **kwargs)
    context.B6Corpus.__init__ = corpus_init

    start = json.loads((Path(args.run) / 'latest_state/meta.json').read_text())
    rank = int(os.environ['RANK'])
    trace = open(Path(args.run) / ('trace_rank%d.jsonl' % rank), 'x', buffering=1)
    counter = {'micro': start['step'], 'pre': None}
    prepared = {}
    def emit(event, **values):
        trace.write(json.dumps(dict(event=event, rank=rank, micro=counter['micro'], **values), sort_keys=True) + '\n')
    def pre(module, positional, kw):
        counter['micro'] += 1
        shapes = {k: dict(shape=list(v.shape), stride=list(v.stride()), dtype=str(v.dtype)) for k, v in kw.items() if isinstance(v, torch.Tensor)}
        counter['pre'] = dict(inputs=digest(kw), shapes=shapes, rng_before=rng())
    def post(module, positional, kw, output):
        emit('forward', **counter['pre'], rng_after=rng(), loss=float(output.loss.detach()),
             graphs=t.rt.compiled_graph_count())
    original_prepare = accelerate.Accelerator.prepare
    def prepare(self, *objects, **kwargs):
        result = original_prepare(self, *objects, **kwargs)
        model = result[0]
        assert isinstance(model, torch.nn.parallel.DistributedDataParallel), type(model)
        model.register_forward_pre_hook(pre, with_kwargs=True)
        model.register_forward_hook(post, with_kwargs=True)
        prepared['model'] = model
        emit('prepared', reducer=model._get_ddp_logging_data())
        return result
    accelerate.Accelerator.prepare = prepare
    original_load = accelerate.Accelerator.load_state
    def load(self, *a, **kw):
        result = original_load(self, *a, **kw)
        actual = rng()
        saved = torch.load(Path(a[0]) / ('random_states_%d.pkl' % rank), map_location='cpu', weights_only=False)
        expected = dict(cpu=digest(saved['torch_manual_seed']),
                        cuda=digest(saved['torch_cuda_manual_seed'][torch.cuda.current_device()]),
                        python=digest(saved['random_state']), numpy=digest(saved['numpy_random_seed']))
        emit('restored', rng=actual, rng_expected=expected, rng_matches_checkpoint=actual == expected,
             model=digest(prepared['model'].state_dict()))
        return result
    accelerate.Accelerator.load_state = load
    original_backward = accelerate.Accelerator.backward
    def backward(self, *a, **kw):
        result = original_backward(self, *a, **kw)
        if counter['micro'] % 24 == 0:
            grads = {n: p.grad for n, p in prepared['model'].named_parameters() if p.grad is not None}
            emit('accumulation_boundary', rng=rng(), gradients=digest(grads),
                 reducer=prepared['model']._get_ddp_logging_data())
        return result
    accelerate.Accelerator.backward = backward
    original_run = t.run_b6
    def bounded(spec, *a, **kw):
        return original_run(dataclasses.replace(spec, stop_at_step=args.stop, save_every_n_steps=10), *a, **kw)
    t.run_b6 = bounded
    if args.snapshot:
        original_rename = os.rename
        def rename(src, dst, *a, **kw):
            result = original_rename(src, dst, *a, **kw)
            if rank == 0 and str(src) == str(Path(args.run) / 'latest_state.tmp') and str(dst) == str(Path(args.run) / 'latest_state'):
                meta = json.loads((Path(dst) / 'meta.json').read_text())
                if meta['global_step'] == 210:
                    clone(args.run, args.snapshot)
                    emit('snapshot_published', path=args.snapshot, rng=rng())
            return result
        os.rename = rename
    result = t.main(['--config', 'configs/b6_400m_win_v3_wp5_pilot.json', '--arm', 'C2', '--seed', '123',
                     '--bindings', args.bindings, '--output-dir', args.run, '--resume', 'auto',
                     '--stop-after-updates', '4099', '--save-every-n-steps', '10',
                     '--mixed-precision', 'bf16', '--deterministic'])
    assert result.get('preempted') and result['global_step'] == args.stop, result
    emit('finished', result=result, rng=rng())
    trace.close()


def checkpoint(path):
    import torch
    from safetensors import safe_open
    state = Path(path) / 'latest_state/accelerate_state'
    with safe_open(state / 'model.safetensors', framework='pt', device='cpu') as f:
        weights = {k: tree(f.get_tensor(k)) for k in f.keys()}
    other = {}
    for n in ['optimizer.bin', 'scheduler.bin'] + ['random_states_%d.pkl' % r for r in range(4)]:
        other[n] = tree(torch.load(state / n, map_location='cpu', weights_only=False))
    return dict(model=weights, state=other)


def compare(args):
    import torch
    torch.set_num_threads(1)
    root = Path(args.root)
    for name, step in [('A', 210), ('B210', 210), ('B', 220), ('C', 220)]:
        m = json.loads((root / name / 'latest_state/meta.json').read_text())
        assert (m['global_step'], m['step'], m['pending_micro']) == (step, step * 24, 0), (name, m)
    a, b210, b220, c = [checkpoint(root / x) for x in ('A', 'B210', 'B', 'C')]
    def diff(x, y):
        return dict(model_equal=x['model'] == y['model'],
                    different_model_keys=[k for k in x['model'] if x['model'][k] != y['model'].get(k)],
                    states_equal={k: x['state'][k] == y['state'].get(k) for k in x['state']})
    report = dict(policy=diff(a, b210), resume=diff(b220, c), trace={})
    for left, right, lo, hi in [('A', 'B', 4800, 5040), ('B', 'C', 5040, 5280)]:
        for rank in range(4):
            def rows(case):
                return {x['micro']: x for x in map(json.loads, (root / case / ('trace_rank%d.jsonl' % rank)).read_text().splitlines())
                        if x['event'] == 'forward' and lo < x['micro'] <= hi}
            x, y = rows(left), rows(right)
            assert set(x) == set(y) == set(range(lo + 1, hi + 1))
            assert all(math.isfinite(v['loss']) for values in (x, y) for v in values.values())
            report['trace']['%s_%s_rank%d' % (left, right, rank)] = {
                k: [m for m in x if x[m][k] != y[m][k]][:12]
                for k in ('inputs', 'shapes', 'rng_before', 'rng_after', 'loss')}
            def boundaries(case):
                return {v['micro']: v['gradients'] for v in map(json.loads, (root / case / ('trace_rank%d.jsonl' % rank)).read_text().splitlines())
                        if v['event'] == 'accumulation_boundary' and lo < v['micro'] <= hi}
            gx, gy = boundaries(left), boundaries(right)
            assert set(gx) == set(gy) == set(range(lo + 24, hi + 1, 24))
            report['trace']['%s_%s_rank%d' % (left, right, rank)]['gradients'] = [m for m in gx if gx[m] != gy[m]][:12]
    report['passed'] = all(report[k]['model_equal'] and all(report[k]['states_equal'].values()) for k in ('policy', 'resume'))
    report['passed'] &= all(not indices for pair in report['trace'].values() for indices in pair.values())
    write_json(root / 'comparison.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['passed'] else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['clone', 'verify', 'run', 'compare'])
    for name in ('source', 'target', 'bindings', 'record', 'run', 'snapshot', 'root'):
        p.add_argument('--' + name)
    p.add_argument('--stop', type=int)
    args = p.parse_args()
    if args.mode == 'clone':
        clone(args.source, args.target)
    elif args.mode == 'verify':
        verify(args)
    elif args.mode == 'run':
        run(args)
    else:
        sys.exit(compare(args))
