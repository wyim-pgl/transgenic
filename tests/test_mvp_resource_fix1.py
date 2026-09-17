"""CPU regressions only; never CUDA/resource certification."""
import ast
import inspect
import json
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_mvp_resource_probe import fixture
import resource_probe as resource


@pytest.mark.parametrize('convert', [bool, float, str])
@pytest.mark.parametrize('field', [
    'schema_version', 'nodes', 'ranks', 'microbatches_per_rank', 'seed', 'rank',
    'maxlength', 'valid_count', 'capacity_bytes', 'microbatches', 'optimizer_steps',
    'rn', 'input_length', 'label_length', 'generated_kernels',
    'est_signal_microbatches', 'inventory_max_label_length', 'peak_cuda_bytes',
    'peak_reserved_bytes'])
def test_integer_fields_reject_coercions(field, convert):
    budget, ranks = fixture()
    row = ranks[0]
    tier = row['tiers']['30720']
    update = tier['updates'][1]
    if field == 'schema_version': target, key = budget, field
    elif field in ('nodes', 'ranks', 'microbatches_per_rank'): target, key = budget['resource_envelope'], field
    elif field == 'seed': target, key = budget['seeds'], 0
    elif field == 'maxlength': target, key = row['label_contract'], field
    elif field in ('rank', 'valid_count', 'capacity_bytes'): target, key = row, field
    elif field == 'inventory_max_label_length': target, key = tier, field
    elif field in ('rn', 'input_length', 'label_length'):
        target, key = update[{'rn': 'rns', 'input_length': 'input_lengths', 'label_length': 'label_lengths'}[field]], 0
    else: target, key = update, field
    target[key] = convert(target[key])
    with pytest.raises(ValueError):
        resource.project_budget(budget, ranks)


def progress_callable(profile, row, local, budget):
    # Execute the actual nested function and its sequence initialization, without
    # pretending CUDA setup/training succeeded. Keep the nonlocal closure intact.
    tree = ast.parse(inspect.getsource(resource.run))
    body = tree.body[0].body
    nodes = [node for node in body if
             (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'progress_sequence' for t in node.targets)) or
             (isinstance(node, ast.FunctionDef) and node.name == 'progress')]
    factory = ast.parse('def factory():\n    pass\n').body[0]
    factory.body = nodes + [ast.Return(ast.Name('progress', ast.Load()))]
    namespace = dict(resource.__dict__, profile=profile, row=row, local=local,
                     budget=budget, started=time.monotonic(), accelerator=SimpleNamespace(process_index=0))
    exec(compile(ast.fix_missing_locations(ast.Module([factory], [])), '<actual-progress>', 'exec'), namespace)
    return namespace['factory']()


@pytest.mark.parametrize('failure', ['collision', 'deadline'])
def test_actual_progress_preserves_snapshots_and_propagates_failure(tmp_path, failure):
    from transgenic.training.mvp_finalize import _collective_preflight
    accelerator = SimpleNamespace(num_processes=1)
    budget = {'max_probe_wall_seconds': 100}
    row = {'stages': {'setup': {}}}
    progress = progress_callable(tmp_path, row, lambda op: _collective_preflight(accelerator, op), budget)
    progress()
    first = tmp_path / 'rank0.000000.profiling.json'
    original = first.read_bytes()
    row['stages']['inventory'] = {}
    progress()
    assert first.read_bytes() == original
    second = tmp_path / 'rank0.000001.profiling.json'
    assert set(json.loads(second.read_text())['measurement']['stages']) == {'setup', 'inventory'}
    third = tmp_path / 'rank0.000002.profiling.json'
    if failure == 'collision':
        third.write_bytes(b'preserve existing')
        message = 'fresh'
    else:
        budget['max_probe_wall_seconds'] = 0
        message = 'cap exhausted'
    with pytest.raises(RuntimeError, match=message):
        progress()
    if failure == 'collision':
        assert third.read_bytes() == b'preserve existing'
    assert first.read_bytes() == original
    assert not list(tmp_path.glob('*.DONE'))


def _invalid_rank_worker(rank, root):
    import datetime
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    dist.init_process_group('gloo', init_method='file://' + str(Path(root) / 'rendezvous'),
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=20))
    try:
        def gather(value):
            outputs = [torch.empty_like(value) for _ in range(2)]
            dist.all_gather(outputs, value)
            return torch.cat(outputs)
        accelerator = SimpleNamespace(num_processes=2, process_index=rank,
                                      is_main_process=rank == 0, device=torch.device('cpu'), gather=gather)
        model = DistributedDataParallel(torch.nn.Linear(2, 2))
        model.module.register_buffer('sync_buffer', torch.zeros(1))
        calls = []
        model.register_forward_pre_hook(lambda *args: calls.append('forward'))
        batch = [torch.ones(1, 2), torch.ones(1, 2), torch.ones(1, 2, dtype=torch.long),
                 None, None, None, None, None, torch.ones(1, 2, 8),
                 {'rn': torch.tensor([rank], dtype=torch.float32 if rank == 1 else torch.int64),
                  'has_evidence': torch.tensor([True])}]
        # Only CUDA synchronization is unavailable in this explicitly CPU test.
        torch.cuda.synchronize = lambda device: None
        try:
            resource.paired_validation(accelerator, model, [batch], [0, 1])
            outcome = {'pass': True}
        except Exception as error:
            outcome = {'pass': False, 'error': str(error), 'kind': type(error).__name__}
        outcome['forward_calls'] = len(calls)
        (Path(root) / f'rank{rank}.json').write_text(json.dumps(outcome))
    finally:
        dist.destroy_process_group()


def test_rank1_invalid_rn_rejects_all_before_ddp_forward(tmp_path):
    context = multiprocessing.get_context('spawn')
    children = [context.Process(target=_invalid_rank_worker, args=(rank, str(tmp_path))) for rank in range(2)]
    try:
        for child in children: child.start()
        deadline = time.monotonic() + 60
        for child in children: child.join(max(0, deadline - time.monotonic()))
        assert all(not child.is_alive() and child.exitcode == 0 for child in children), 'Gloo fault test hung/failed'
    finally:
        for child in children:
            if child.is_alive(): child.terminate()
        for child in children: child.join(5)
    outcomes = [json.loads((tmp_path / f'rank{rank}.json').read_text()) for rank in range(2)]
    assert outcomes[0] == outcomes[1]
    assert outcomes[0]['pass'] is False and outcomes[0]['forward_calls'] == 0
    assert outcomes[0]['kind'] == 'ValueError'
    assert 'one integer rn' in outcomes[0]['error']
    assert not list(tmp_path.glob('*.DONE'))
