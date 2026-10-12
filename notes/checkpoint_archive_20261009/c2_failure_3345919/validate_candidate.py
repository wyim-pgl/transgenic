"""One-off diagnostic: live guard on real replay, then train/eval specialization boundaries. No run writes."""
import json
import math
from pathlib import Path

from revision.scripts.b6 import diag_gpu_memory as diag


original_run_real = diag.run_real


def validate(T, real):
    torch, rt = T.torch, diag.rt
    originals = list(range(1, 73)) + list(range(3913, 3937)) + list(range(6625, 6719))
    assert len(real.samples) == len(originals) == 190
    baseline = json.loads(Path('/work/runs/c2_recompile_diag/3360978/replay.json').read_text())
    assert baseline['status'] == 'completed' and len(baseline['samples']) == 190
    guard = rt.CompileGuard(3, graphs_before=rt.compiled_graph_count())
    ctx = dict(phase='real', calls=0, update=1000)
    T.report['validation'] = dict(status='real_replay', guard_events=[], matrix=[])
    validation = T.report['validation']

    def observe(module, args, inputs, output):
        assert not torch._dynamo.config.automatic_dynamic_shapes
        if ctx['phase'] == 'real':
            row = originals[ctx['calls']]
            ctx['calls'] += 1
            update = (row - 1) // 24 + 1
        else:
            row, update = None, ctx['update']
        key = (() if torch.is_grad_enabled() else ('eval',)) + tuple(inputs['input_ids'].shape)
        before = guard.graphs
        now = rt.compiled_graph_count()
        guard.observe(update, key, now, bool(torch._dynamo.config.suppress_errors))
        if now != before:
            event = dict(phase=ctx['phase'], original_row=row, update=update, key=list(key),
                         labels=int(inputs['labels'].shape[1]), before=before, after=now)
            validation['guard_events'].append(event)
            print('GUARD_EVENT ' + json.dumps(event), flush=True)

    hook = T.model.register_forward_hook(observe, with_kwargs=True)
    try:
        rc = original_run_real(T, real)
        assert rc == 0 and T.report['status'] == 'completed', T.report['status']
        assert ctx['calls'] == 190
        for actual, expected in zip(T.report['samples'], baseline['samples']):
            assert actual['status'] == 'ok' and actual['draw_sha_match'] is True
            assert actual['memory']['max_reserved_bytes'] < rt.MEMORY_ABORT_BYTES, actual['order']
            assert (actual['rn'], actual['sample_sha']) == (expected['rn'], expected['sample_sha'])
            assert math.isfinite(actual['loss']) and actual['loss'] == expected['loss'], (
                actual['order'], actual['rn'], actual['loss'], expected['loss'])
        for update in T.report['updates']:
            assert update['result'] == 'stepped' and not update['skipped'], update
            assert update['max_reserved_bytes'] < rt.MEMORY_ABORT_BYTES, update
        validation['real_losses_exact'] = True
        validation['real_samples'] = 190
        validation['status'] = 'matrix_running'
        T.report['status'] = 'validating_matrix'
        T.save()

        labels = [3, 80, 3413, 3414, 3924, 4095, 4096, 4501, 8192]
        stages = [('train', True, labels), ('eval', False, list(reversed(labels))),
                  ('train_reverse', True, list(reversed(labels)))]
        index = 0
        for phase, training, sizes in stages:
            T.model.train(training)
            tiers = (30720, 61440, 129024) if phase == 'train' else (129024, 61440, 30720)
            for tier in tiers:
                for length in sizes:
                    batch = diag.make_batch(torch, T.cfg, T.args, tier, length, index, T.device)
                    rt.validate_b6_batch(batch)
                    views = ([('training', rt.apply_arm(batch, T.arm))] if training else
                             [('de_novo', rt.de_novo_view(batch)), ('evidence', rt.apply_arm(batch, T.arm))])
                    for view, data in views:
                        index += 1
                        ctx.update(phase=phase + '/' + view, update=1000 + index)
                        T.optimizer.zero_grad(set_to_none=True)
                        torch.cuda.reset_peak_memory_stats(T.device)
                        inputs = rt.model_inputs(data)
                        torch._dynamo.maybe_mark_dynamic(inputs['labels'], 1)
                        before = rt.compiled_graph_count()
                        with torch.enable_grad() if training else torch.no_grad():
                            out = T.model(**inputs)
                            assert torch.isfinite(out.loss.detach()).item()
                            if training:
                                T.acc.backward(out.loss / T.accum)
                        torch.cuda.synchronize(T.device)
                        record = dict(phase=phase, view=view, tier=tier, labels=length,
                                      graphs_before=before, graphs_after=rt.compiled_graph_count(),
                                      **diag._mem(torch, T.device))
                        assert record['max_reserved_bytes'] < rt.MEMORY_ABORT_BYTES, record
                        validation['matrix'].append(record)
                        print('MATRIX ' + json.dumps(record), flush=True)
                        del out, inputs
                        T.save()
                    del views, batch, data
        assert len(validation['matrix']) == 108
        validation['events_by_shape'] = [{'key': list(k), 'events': n} for k, n in guard.events.items()]
        validation['compile_events'] = sum(guard.events.values())
        assert validation['compile_events'] <= 18
        validation['status'] = 'passed'
        T.report['status'] = 'validation_complete'
        T.report['final_compiled_graphs'] = rt.compiled_graph_count()
        T.save()
        print('VALIDATION_COMPLETE ' + json.dumps({k: validation[k] for k in
              ['real_losses_exact', 'real_samples', 'compile_events', 'events_by_shape']}), flush=True)
        return 0
    except BaseException as exc:
        validation.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        T.report['status'] = 'validation_failed'
        T.save()
        raise
    finally:
        hook.remove()


diag.run_real = validate
if __name__ == '__main__':
    raise SystemExit(diag.main())
