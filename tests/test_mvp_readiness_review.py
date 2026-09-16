"""CPU regressions for independent review A1/A2; no GPU certification."""
import json
from pathlib import Path
import sys

import pytest

from test_mvp_readiness_tools import (TOOLS, input_args, launch_fixture, put,
                                     readiness, stage_args, staging)


@pytest.mark.parametrize('name', ['config.json', 'meta.json', 'runtime.json'])
@pytest.mark.parametrize('mutation', ['add', 'remove', 'change'])
def test_sidecar_change_invalidates_check_and_preflight_bindings(tmp_path, monkeypatch, name, mutation):
    args, ready, _, _ = launch_fixture(tmp_path, monkeypatch,
                                       sidecars={name: {'version': 1}} if mutation != 'add' else None)
    before = json.loads((tmp_path / 'bindings.json').read_text())
    target = tmp_path / 'checkpoint' / name
    if mutation == 'remove':
        target.unlink()
    else:
        # Includes the reviewer's invalid-JSON config addition reproduction.
        target.write_text('not JSON' if name == 'config.json' else '{"version":2}')
    assert readiness.main(['check', *args, '--readiness', str(ready),
                           '--readiness-sha256', readiness._sha256(ready)]) == 2
    current = readiness._source_checkpoint_binding(tmp_path / 'checkpoint')
    assert current != before['bindings']['source_checkpoint']
    # Reusing the old six preflight receipts must fail during generation too.
    pins = [item for name, file in [('weights', tmp_path / 'checkpoint/model.safetensors'),
                                   ('run-config', tmp_path / 'run_config.json'),
                                   ('runtime-identity', tmp_path / 'runtime_identity.txt')]
            for item in ('--expected-' + name + '-sha256', readiness._sha256(file))]
    assert readiness.main(['generate', *args, *pins, '--manifest-output', str(tmp_path / 'code.json'),
                           '--preflight-dir', str(tmp_path), '--output', str(tmp_path / 'other.json')]) == 2
    assert not (tmp_path / 'other.json').exists()


@pytest.mark.parametrize('name', ['config.json', 'meta.json', 'runtime.json'])
@pytest.mark.parametrize('kind', ['symlink', 'dangling', 'directory'])
def test_optional_sidecars_must_be_canonical_regular_files(tmp_path, monkeypatch, name, kind):
    args = stage_args(tmp_path, monkeypatch)
    target = tmp_path / 'epoch_05' / name
    if kind == 'directory':
        target.mkdir()
    else:
        other = tmp_path / 'other.json'
        if kind == 'symlink':
            put(other, {})
        target.symlink_to(other)
    with pytest.raises((ValueError, SystemExit)):
        readiness._source_checkpoint_binding(tmp_path / 'epoch_05')
    with pytest.raises((ValueError, SystemExit)):
        staging.main(args)
    assert not (tmp_path / 'staged').exists()


@pytest.mark.parametrize('present', [False, True])
def test_staging_preserves_complete_loader_inventory(tmp_path, monkeypatch, present):
    args = stage_args(tmp_path, monkeypatch)
    names = ('config.json', 'meta.json', 'runtime.json')
    if present:
        for name in names:
            put(tmp_path / 'epoch_05' / name, {'name': name})
    assert staging.main(args) == 0
    root = tmp_path / 'staged'
    receipt = json.loads((root / 'staging_receipt.json').read_text())
    assert set(receipt.get('optional_sidecars', {})) == set(names)
    for name in names:
        target = root / 'checkpoint' / name
        if present:
            assert target.read_bytes() == (tmp_path / 'epoch_05' / name).read_bytes()
            assert receipt['optional_sidecars'][name]['sha256'] == readiness._sha256(target)
        else:
            assert receipt['optional_sidecars'][name] is None and not target.exists()


@pytest.mark.parametrize('mutation', ['add', 'remove', 'change'])
def test_staging_refuses_loader_inventory_drift_during_copy(tmp_path, monkeypatch, mutation):
    args = stage_args(tmp_path, monkeypatch)
    sidecar = tmp_path / 'epoch_05/runtime.json'
    if mutation != 'add':
        put(sidecar, {'version': 1})
    original_copy = staging._copy_verified
    def copy_then_mutate(source, destination, expected, label):
        result = original_copy(source, destination, expected, label)
        if label == 'weights':
            if mutation == 'remove':
                sidecar.unlink()
            else:
                put(sidecar, {'version': 2})
        return result
    monkeypatch.setattr(staging, '_copy_verified', copy_then_mutate)
    with pytest.raises((ValueError, SystemExit)):
        staging.main(args)
    assert not (tmp_path / 'staged').exists()


def test_explicit_accelerate_config_survives_conflicting_default(tmp_path, monkeypatch):
    from accelerate.commands import launch
    from accelerate.commands.config import config_args
    import run_training
    args, ready, budget, budget_sha = launch_fixture(tmp_path, monkeypatch)
    training_args = [*args, '--output-dir', str(tmp_path / 'model'), '--launch-readiness', str(ready),
                     '--launch-readiness-sha256', readiness._sha256(ready)]
    argv_file = tmp_path / 'argv.json'
    argv_sha = put(argv_file, training_args)
    commands = []
    def execute(command, log, timeout):
        commands.append(command)
        return {'exit_code': 0, 'signal': None, 'timed_out': False}
    monkeypatch.setattr(run_training, 'run_command', execute)
    assert run_training.main(['--arguments', str(argv_file), '--arguments-sha256', argv_sha,
        '--budget-authorization', str(budget), '--budget-sha256', budget_sha,
        '--timeout-seconds', '600', '--status-dir', str(tmp_path / 'status')]) == 0
    command = commands[-1]
    # Actual installed parser + actual config loader; only CPU bf16 capability
    # is substituted, not configuration merging or argv handling.
    ambient = tmp_path / 'ambient.json'
    put(ambient, dict(compute_environment='LOCAL_MACHINE', distributed_type='MULTI_GPU',
                     mixed_precision='no', use_cpu=False, debug=False, num_processes=2,
                     num_machines=2, gpu_ids='2,3', main_process_ip='192.0.2.1',
                     main_process_port=29999, rdzv_backend='c10d',
                     dynamo_config={'dynamo_backend': 'EAGER'}))
    monkeypatch.setattr(launch, 'default_config_file', str(ambient))
    monkeypatch.setattr(config_args, 'default_config_file', str(ambient))
    monkeypatch.setattr(launch, 'is_bf16_available', lambda *args: True)
    parsed, _, _ = launch._validate_launch_command(launch.launch_command_parser().parse_args(command[3:]))
    assert parsed.gpu_ids == '0,1,2,3'
    assert parsed.dynamo_backend.upper() == 'NO'
    assert (parsed.num_processes, parsed.num_machines, parsed.machine_rank) == (4, 1, 0)
    assert parsed.mixed_precision == 'bf16'
    assert (parsed.main_process_ip, parsed.main_process_port, parsed.rdzv_backend) == ('127.0.0.1', 29500, 'static')
    assert parsed.max_restarts == 0
    binding = json.loads(ready.read_text())['bindings']['accelerate_config']
    assert parsed.config_file == binding['path']
    assert readiness._sha256(Path(parsed.config_file)) == binding['sha256']
    assert parsed.training_script_args == training_args


@pytest.mark.parametrize('mutation', ['change', 'remove', 'recreate'])
def test_accelerate_config_identity_checked(tmp_path, monkeypatch, mutation):
    assert hasattr(readiness, 'ACCELERATE_CONFIG'), 'explicit reviewed config missing'
    copied = tmp_path / 'accelerate.json'
    copied.write_bytes(readiness.ACCELERATE_CONFIG.read_bytes())
    monkeypatch.setattr(readiness, 'ACCELERATE_CONFIG', copied)
    args, ready, _, _ = launch_fixture(tmp_path, monkeypatch)
    if mutation == 'change':
        copied.write_text(copied.read_text() + '\n')
    elif mutation == 'remove':
        copied.unlink()
    else:
        copied.unlink()
        copied.write_text('{}')
    assert readiness.main(['check', *args, '--readiness', str(ready),
                           '--readiness-sha256', readiness._sha256(ready)]) == 2
    pins = [item for name, file in [('weights', tmp_path / 'checkpoint/model.safetensors'),
                                   ('run-config', tmp_path / 'run_config.json'),
                                   ('runtime-identity', tmp_path / 'runtime_identity.txt')]
            for item in ('--expected-' + name + '-sha256', readiness._sha256(file))]
    assert readiness.main(['generate', *args, *pins, '--manifest-output', str(tmp_path / 'code.json'),
                           '--preflight-dir', str(tmp_path), '--output', str(tmp_path / 'other.json')]) == 2
    assert not (tmp_path / 'other.json').exists()


def test_prepare_cannot_leave_accelerate_config_unbound_for_later_addition(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'CPU_UNIT_FIXTURE')
    monkeypatch.setattr(readiness, 'ACCELERATE_CONFIG', tmp_path / 'absent.json')
    args, pins = input_args(tmp_path)
    assert readiness.main(['prepare', *args, *pins, '--manifest-output', str(tmp_path / 'code.json'),
                           '--output', str(tmp_path / 'bindings.json')]) == 2
    assert not (tmp_path / 'bindings.json').exists()


@pytest.mark.parametrize('kind', ['symlink', 'directory'])
def test_accelerate_config_must_be_regular_and_nonsymlink(tmp_path, monkeypatch, kind):
    config = tmp_path / 'config'
    if kind == 'symlink':
        config.symlink_to(readiness.ACCELERATE_CONFIG)
    else:
        config.mkdir()
    monkeypatch.setattr(readiness, 'ACCELERATE_CONFIG', config)
    with pytest.raises(ValueError):
        readiness.accelerate_config_binding()


def test_changed_accelerate_config_stops_before_training_after_warmup(tmp_path, monkeypatch):
    import run_training
    copied = tmp_path / 'accelerate.json'
    copied.write_bytes(readiness.ACCELERATE_CONFIG.read_bytes())
    monkeypatch.setattr(readiness, 'ACCELERATE_CONFIG', copied)
    args, ready, budget, budget_sha = launch_fixture(tmp_path, monkeypatch)
    argv_file = tmp_path / 'argv.json'
    sha = put(argv_file, [*args, '--output-dir', str(tmp_path / 'model'),
                         '--launch-readiness', str(ready), '--launch-readiness-sha256', readiness._sha256(ready)])
    calls = []
    def warmup(command, log, timeout):
        calls.append(command)
        copied.write_text(copied.read_text() + '\n')
        return {'exit_code': 0, 'signal': None, 'timed_out': False}
    monkeypatch.setattr(run_training, 'run_command', warmup)
    assert run_training.main(['--arguments', str(argv_file), '--arguments-sha256', sha,
        '--budget-authorization', str(budget), '--budget-sha256', budget_sha,
        '--timeout-seconds', '600', '--status-dir', str(tmp_path / 'status')]) == 2
    assert len(calls) == 1  # Warmup only; no Accelerate attempt.
    assert 'Accelerate config changed' in json.loads((tmp_path / 'status/FAILED.json').read_text())['error']
