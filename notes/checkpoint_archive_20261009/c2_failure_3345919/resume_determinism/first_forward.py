"""Diagnostic intervention only: prime compiler history, restore state, measure one actual forward."""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import probe


class DiagnosticComplete(BaseException):
    pass


def snapshot_module(model):
    parameters = {n: (p, p.detach().cpu().clone()) for n, p in model.named_parameters()}
    buffers = [(m, {n: (b, None if b is None else b.detach().cpu().clone()) for n, b in m._buffers.items()},
                set(m._non_persistent_buffers_set)) for m in model.modules()]
    return parameters, buffers


def restore_module(model, snapshot):
    import torch
    parameters, buffers = snapshot
    current = dict(model.named_parameters())
    assert current.keys() == parameters.keys()
    with torch.no_grad():
        for name, (original, saved) in parameters.items():
            assert current[name] is original, ('parameter object changed', name)
            original.copy_(saved)
        for module, saved, nonpersistent in buffers:
            for name in list(module._buffers):
                if name not in saved:
                    del module._buffers[name]
            for name, (original, value) in saved.items():
                if original is not None:
                    original.copy_(value)
                module._buffers[name] = original
            module._non_persistent_buffers_set = set(nonpersistent)


def module_digest(model):
    data = {'parameters': dict(model.named_parameters()), 'buffers': {}}
    for path, module in model.named_modules():
        data['buffers'][path] = dict(values=dict(module._buffers), nonpersistent=sorted(module._non_persistent_buffers_set))
    return probe.digest(data)


def main(args):
    import accelerate
    import numpy as np
    import torch
    import torch._inductor.config
    import torch._functorch.config
    from accelerate.optimizer import AcceleratedOptimizer
    t = probe.load_trainer()
    # Same diagnostic setting in every case: no FX graph cache can hide the reset intervention.
    torch._inductor.config.fx_graph_cache = False
    torch._inductor.config.fx_graph_remote_cache = False
    torch._functorch.config.enable_autograd_cache = False
    torch._functorch.config.enable_remote_autograd_cache = False
    payload = json.loads(Path(args.record).read_text())
    assert payload['job_id'] == os.environ['SLURM_JOB_ID']
    assert probe.sha(args.bindings) == payload['bindings_sha256']
    for path, expected in payload['input_stats'].items():
        s = Path(path).stat()
        assert [s.st_size, s.st_mtime_ns] == expected, path
    def verified(paths, *, roles, cache, b6_bindings, aug_seed):
        assert paths == payload['paths'] and str(cache.mmap_dir) == payload['mmap_dir']
        assert aug_seed == t.rt.AUG_SEED_BASE + 123
        t.rt.identity_from_bindings(payload['record'], b6_bindings.canonical_record_sha256)
        return dict(payload['record'])
    t.verify_training_inputs = verified
    _, _, _, context, _, _ = t._wp3()
    original_corpus_init = context.B6Corpus.__init__
    def corpus_init(self, corpus_path, cache, expected_sha256=None, **kw):
        assert str(corpus_path) == payload['paths']['corpus']
        assert expected_sha256 == payload['record']['corpus_sha256']
        return original_corpus_init(self, corpus_path, cache, expected_sha256=None, **kw)
    context.B6Corpus.__init__ = corpus_init
    def forbid_optimizer_step(*a, **kw):
        raise RuntimeError('first-forward diagnostic must never update the optimizer')
    AcceleratedOptimizer.step = forbid_optimizer_step

    rank = int(os.environ['RANK'])
    evidence = Path(args.evidence)
    old_rows = [json.loads(x) for x in (evidence / 'B' / ('trace_rank%d.jsonl' % rank)).read_text().splitlines()]
    warm = {x['micro']: x for x in old_rows if x['event'] == 'forward'}
    cold = next(x for x in map(json.loads, (evidence / 'C' / ('trace_rank%d.jsonl' % rank)).read_text().splitlines())
                if x['event'] == 'forward')
    draw = (evidence / 'B/draw_log' / ('rank%02d.tsv' % rank)).read_text().splitlines()
    assert cold['micro'] == 5041
    assert warm[5041]['inputs'] == cold['inputs'] and warm[5041]['rng_before'] == cold['rng_before']
    state = {'phase': 'first', 'dataset': None, 'collate': None, 'done': False, 'primers': []}
    original_run = t.run_b6
    def run(spec, model, train_ds, eval_ds, **kw):
        assert spec.required_world_size == 4 and spec.arm.name == 'C2' and spec.deterministic
        state.update(dataset=train_ds, collate=kw['collate_fn'])
        return original_run(spec, model, train_ds, eval_ds, **kw)
    t.run_b6 = run

    original_prepare = accelerate.Accelerator.prepare
    def prepare(acc, *objects, **kw):
        result = original_prepare(acc, *objects, **kw)
        model = result[0]
        assert isinstance(model, torch.nn.parallel.DistributedDataParallel) and acc.num_processes == 4
        guard = t.rt.CompileGuard(3, graphs_before=t.rt.compiled_graph_count())
        kernels_before = t.rt.inductor_activity()
        graphs_before = t.rt.compiled_graph_count()
        def pre(module, positional, inputs):
            nonlocal guard
            if state['phase'] == 'priming':
                return
            assert state['phase'] == 'first'
            assert probe.digest(inputs) == cold['inputs'], 'actual target tensor bytes differ'
            assert probe.rng() == cold['rng_before'], 'pre-intervention RNG differs'
            state['phase'] = 'priming'
            initial = module_digest(model)
            saved_rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                             python=random.getstate(), numpy=np.random.get_state())
            snapshot = snapshot_module(model)
            if args.case in ('primed', 'primed_reset'):
                for micro in range(4801, 4806):
                    rn, epoch, expected_draw = draw[micro - 1].split('\t')
                    batch = state['collate']([state['dataset'][(int(rn), int(epoch))]])
                    assert batch['meta']['uniforms_sha256'][0] == expected_draw
                    priming = t.rt.model_inputs(t.rt.apply_arm(batch, t.rt.ARMS['C2']))
                    priming = {k: v.to(acc.device) if isinstance(v, torch.Tensor) else v for k, v in priming.items()}
                    assert probe.digest(priming) == warm[micro]['inputs'], ('primer bytes differ', rank, micro)
                    torch._dynamo.maybe_mark_dynamic(priming['labels'], 1)
                    out = model(**priming)
                    guard.observe(1, tuple(priming['input_ids'].shape), t.rt.compiled_graph_count(),
                                  bool(torch._dynamo.config.suppress_errors))
                    acc.backward(out.loss / 24)
                    state['primers'].append(dict(micro=micro, row=int(rn), graphs=t.rt.compiled_graph_count()))
                    del out, priming, batch
            # Both cases take the same snapshot/restore path. Every parameter and registered buffer is restored,
            # including nonpersistent buffers absent from state_dict. No optimizer update is permitted.
            restore_module(model, snapshot)
            del snapshot
            model.zero_grad(set_to_none=True)
            random.setstate(saved_rng['python'])
            np.random.set_state(saved_rng['numpy'])
            torch.set_rng_state(saved_rng['cpu'])
            torch.cuda.set_rng_state_all(saved_rng['cuda'])
            if args.case == 'primed_reset':
                torch._dynamo.reset()
                guard = t.rt.CompileGuard(3, graphs_before=t.rt.compiled_graph_count())
            restored = module_digest(model)
            assert restored == initial
            assert probe.rng() == cold['rng_before']
            assert all(p.grad is None for p in model.parameters())
            state.update(phase='target', initial_state=initial, restored_state=restored,
                         input_sha256=probe.digest(inputs), rng_before=probe.rng(),
                         graphs_before_target=t.rt.compiled_graph_count())
        def post(module, positional, inputs, out):
            if state['phase'] == 'priming':
                return
            assert state['phase'] == 'target'
            guard.observe(1, tuple(inputs['input_ids'].shape), t.rt.compiled_graph_count(),
                          bool(torch._dynamo.config.suppress_errors))
            t.rt.check_compiled(graphs_before, t.rt.compiled_graph_count())
            t.rt.check_inductor_kernels('inductor', kernels_before, t.rt.inductor_activity())
            report = dict(case=args.case, rank=rank, micro=5041, loss=float(out.loss.detach()),
                          logits_sha256=probe.digest(out.logits), inputs=state['input_sha256'],
                          initial_model_buffers=state['initial_state'], restored_model_buffers=state['restored_state'],
                          rng_before=state['rng_before'], rng_after=probe.rng(), primers=state['primers'],
                          graphs_before=state['graphs_before_target'], graphs_after=t.rt.compiled_graph_count(),
                          warm_expected_loss=warm[5041]['loss'], cold_expected_loss=cold['loss'],
                          optimizer_updates=0, reducer=model._get_ddp_logging_data())
            report['fx_graph_cache'] = torch._inductor.config.fx_graph_cache
            report['autograd_cache'] = torch._functorch.config.enable_autograd_cache
            assert report['rng_after'] == cold['rng_after']
            assert int(torch.cuda.max_memory_reserved()) <= t.rt.MEMORY_ABORT_BYTES
            probe.write_json(Path(args.run) / ('first_forward_rank%d.json' % rank), report)
            state['done'] = True
            acc.wait_for_everyone()
            raise DiagnosticComplete()
        model.register_forward_pre_hook(pre, with_kwargs=True)
        model.register_forward_hook(post, with_kwargs=True)
        return result
    accelerate.Accelerator.prepare = prepare
    try:
        t.main(['--config', 'configs/b6_400m_win_v3_wp5_pilot.json', '--arm', 'C2', '--seed', '123',
                '--bindings', args.bindings, '--output-dir', args.run, '--resume', 'auto',
                '--stop-after-updates', '4099', '--mixed-precision', 'bf16', '--deterministic'])
    except DiagnosticComplete:
        assert state['done']
    else:
        raise RuntimeError('diagnostic failed to stop after its one target forward')
    torch.distributed.destroy_process_group()
    print('FIRST_FORWARD_COMPLETE', args.case, rank, flush=True)


def compare(root):
    reports = {}
    for rank in range(4):
        a, b, control = [json.loads((Path(root) / c / ('first_forward_rank%d.json' % rank)).read_text())
                        for c in ('cold', 'primed', 'primed_reset')]
        for key in ('inputs', 'initial_model_buffers', 'restored_model_buffers', 'rng_before', 'rng_after'):
            assert a[key] == b[key] == control[key], (rank, key)
        assert a['optimizer_updates'] == b['optimizer_updates'] == 0
        assert not a['primers'] and len(b['primers']) == len(control['primers']) == 5
        assert control['optimizer_updates'] == 0
        reports[rank] = dict(cold_loss=a['loss'], primed_loss=b['loss'],
                             cold_reproduced=a['loss'] == a['cold_expected_loss'],
                             warm_reproduced=b['loss'] == b['warm_expected_loss'],
                             reset_loss=control['loss'], reset_matches_cold=control['logits_sha256'] == a['logits_sha256'],
                             logits_equal=a['logits_sha256'] == b['logits_sha256'],
                             loss_equal=a['loss'] == b['loss'])
    probe.write_json(Path(root) / 'first_forward_comparison.json', reports)
    print(json.dumps(reports, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['run', 'compare'])
    for name in ('record', 'bindings', 'run', 'evidence', 'root'):
        p.add_argument('--' + name)
    p.add_argument('--case', choices=['cold', 'primed', 'primed_reset'])
    args = p.parse_args()
    if args.mode == 'run':
        main(args)
    else:
        compare(args.root)
