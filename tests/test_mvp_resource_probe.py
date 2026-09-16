"""CPU arithmetic/contracts only. These fixtures are NOT CUDA certificates."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision/scripts/mvp'))


def fixture():
    budget = dict(schema_version=1, authorized_scope='exploratory_mvp',
        controller_record_reference='test-only-not-an-authorization',
        resource_envelope=dict(nodes=1, ranks=4, microbatches_per_rank=24),
        seeds=[1, 2, 3], intent='three independent 250-update EST training runs',
        bindings_sha256='a' * 64, safety_margin=1.5, memory_headroom_fraction=0.1,
        margin_rationale='test arithmetic only', max_wall_seconds=10000,
        max_three_seed_node_hours=10, max_probe_wall_seconds=18000, prior_smoke_node_hours=0,
        launcher_overhead_seconds=60)
    ranks = []
    for rank in range(4):
        stages = {name: dict(wall_seconds=2., peak_cuda_bytes=40,
                            peak_reserved_bytes=50) for name in
                  ('setup', 'inventory', 'validation', 'publication', 'revalidation')}
        tiers = {}
        for tier in (30720, 61440, 129024):
            updates = [dict(wall_seconds=3. + rank, peak_cuda_bytes=40,
                peak_reserved_bytes=50, microbatches=24, optimizer_steps=1,
                label_lengths=[100] * 24, rns=list(range(rank * 24, (rank + 1) * 24)),
                input_lengths=[tier] * 24, generated_kernels=0, est_signal_microbatches=12) for _ in range(3)]
            updates[0]['generated_kernels'] = 1
            tiers[str(tier)] = dict(updates=updates, inventory_max_label_length=100)
        ranks.append(dict(rank=rank, capacity_bytes=100, stages=stages, tiers=tiers,
                          label_contract=dict(gff_vocab_version='v3', maxlength=8192),
                          elapsed_seconds=100., valid_count=20, valid_inventory_sha256='b'*64))
    return budget, ranks


def test_worst_rank_tier_and_all_elapsed_costs():
    from resource_probe import project_budget
    budget, ranks = fixture()
    ranks[2]['tiers']['129024']['updates'][2]['wall_seconds'] = 8
    result = project_budget(budget, ranks)
    assert result['measured_step_seconds'] == 8
    assert result['projected_run_seconds'] == (250 * 8 + 100) * 1.5 + 60
    assert result['projected_three_seed_node_hours'] == 3 * result['projected_run_seconds'] / 3600


@pytest.mark.parametrize('damage', ['stage', 'rank', 'tier', 'steady', 'micro', 'nan',
    'memory', 'wall', 'total', 'cap', 'labels', 'compile', 'valid'])
def test_incomplete_or_unsafe_measurement_rejected(damage):
    from resource_probe import project_budget
    budget, ranks = fixture()
    row = ranks[0]
    update = row['tiers']['129024']['updates'][2]
    if damage == 'stage': del row['stages']['publication']
    if damage == 'rank': ranks.pop()
    if damage == 'tier': del row['tiers']['129024']
    if damage == 'steady': row['tiers']['129024']['updates'].pop()
    if damage == 'micro': update['microbatches'] = 23
    if damage == 'nan': update['wall_seconds'] = float('nan')
    if damage == 'memory': update['peak_reserved_bytes'] = 91
    if damage == 'wall': budget['max_wall_seconds'] = 100
    if damage == 'total': budget['max_three_seed_node_hours'] = 0.01
    if damage == 'cap': row['elapsed_seconds'] = 18001
    if damage == 'labels': row['tiers']['129024']['inventory_max_label_length'] = 101
    if damage == 'compile': update['generated_kernels'] = 1
    if damage == 'valid': row['valid_inventory_sha256'] = 'c' * 64
    with pytest.raises(ValueError): project_budget(budget, ranks)


def test_prior_smoke_time_cannot_disappear():
    from resource_probe import validate_budget
    budget, _ = fixture()
    budget['prior_smoke_node_hours'] = 1
    with pytest.raises(ValueError, match='cap'): validate_budget(budget, 'a'*64)
    budget['max_probe_wall_seconds'] = 14400
    validate_budget(budget, 'a'*64)


def test_unsupported_compiler_refused_without_fallback():
    from resource_probe import strict_compile_limits
    from types import SimpleNamespace
    with pytest.raises(RuntimeError, match='unsupported CUDA compiler'):
        strict_compile_limits(SimpleNamespace(suppress_errors=False))
    config = SimpleNamespace(fail_on_cache_limit_hit=False, suppress_errors=True)
    strict_compile_limits(config)
    assert config.fail_on_cache_limit_hit and not config.suppress_errors


@pytest.mark.parametrize('key', ['controller_record_reference', 'seeds', 'safety_margin',
    'resource_envelope', 'intent', 'bindings_sha256', 'margin_rationale'])
def test_missing_authorization_field(key):
    from resource_probe import validate_budget
    budget, _ = fixture()
    del budget[key]
    with pytest.raises(ValueError): validate_budget(budget, 'a' * 64)


def test_boolean_authorization_and_changed_binding_fail():
    from resource_probe import validate_budget
    with pytest.raises(ValueError): validate_budget({'approved': True}, 'a'*64)
    budget, _ = fixture()
    with pytest.raises(ValueError): validate_budget(budget, 'c'*64)
    budget['safety_margin'] = True
    with pytest.raises(ValueError): validate_budget(budget, 'a'*64)


def test_resource_runtime_flags_alone_never_certify():
    from run_bounded import validate_measurement
    with pytest.raises(ValueError):
        validate_measurement('resource_budget', dict(device='cuda', real_runtime=True,
            measured_step_seconds=1, peak_memory_bytes=1, memory_limit_bytes=2,
            authorized_wall_seconds=1000, projected_run_seconds=500,
            authorization_sha256='a'*64))


def test_v2_or_short_cap_cannot_certify_resource():
    from resource_probe import require_v3_dataset
    from types import SimpleNamespace
    cfg = dict(gff_vocab_version='v3', max_decoder_position_embeddings=8192, vocab_size=3)
    ds = SimpleNamespace(gff_vocab_version='v2', maxlength=2048,
                         dt=SimpleNamespace(vocab={'a': 0, 'b': 1, 'c': 2}))
    with pytest.raises(ValueError, match='v3/8192'): require_v3_dataset(ds, cfg)
    ds.gff_vocab_version = 'v3'
    with pytest.raises(ValueError, match='v3/8192'): require_v3_dataset(ds, cfg)
    ds.maxlength = 8192
    require_v3_dataset(ds, cfg)


def test_missing_budget_file_and_mode_inputs_fail(tmp_path):
    from resource_probe import run
    from make_launch_readiness import json_snapshot
    from types import SimpleNamespace
    with pytest.raises(ValueError, match='budget-authorization'): run(SimpleNamespace())
    with pytest.raises((OSError, ValueError)):
        json_snapshot(str(tmp_path / 'absent.json'), 'a' * 64)


@pytest.mark.parametrize('damage', ['duplicate', 'undercount', 'no_est', 'boolean_time'])
def test_unmeasured_coverage_or_undercount_rejected(damage):
    from resource_probe import project_budget
    budget, ranks = fixture()
    update = ranks[0]['tiers']['30720']['updates'][1]
    if damage == 'duplicate':
        update['rns'] = ranks[1]['tiers']['30720']['updates'][1]['rns']
    if damage == 'undercount': ranks[0]['elapsed_seconds'] = 1
    if damage == 'no_est': update['est_signal_microbatches'] = 0
    if damage == 'boolean_time': update['wall_seconds'] = True
    with pytest.raises(ValueError): project_budget(budget, ranks)

