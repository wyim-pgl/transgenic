"""CPU contract tests only: not GPU evidence or a training authorization."""
import hashlib
import json
from types import SimpleNamespace

import pytest

LEGACY = 'historical-three-seed'
DEV = 'development-single'


def test_profile_contract():
    from transgenic.training import mvp_profiles as p
    assert (p.LEGACY_PROFILE, p.DEVELOPMENT_PROFILE) == (LEGACY, DEV)
    for profile, seed in [(LEGACY, 1), (LEGACY, 2), (LEGACY, 3), (DEV, 123)]:
        assert p.validate_seed_profile(profile, seed) is None
    for profile, seed in [(LEGACY, 123), (DEV, 1), (DEV, True), (LEGACY, True),
                          (DEV, 123.0), (DEV, '123'), (None, 1), ([], 1), ('x', 1)]:
        with pytest.raises(ValueError):
            p.validate_seed_profile(profile, seed)


def test_cli_seed123_explicit_profile_and_early_rejection(capsys):
    from transgenic.training.mvp_cli import build_parser, main
    args = build_parser().parse_args(['--profile', DEV, '--seed', '123'])
    assert args.seed == 123 and args.profile == DEV
    argv = []
    for key in ('db', 'evidence-db', 'panel', 'acceptance-record', 'acceptance-sha256',
                'inventory-record', 'receipts-record', 'config', 'init-from', 'output-dir', 'sif'):
        argv += ['--' + key, '/absent']
    assert main(argv + ['--seed', '123']) == 2
    assert 'profile' in capsys.readouterr().err


@pytest.mark.parametrize('profile,seed', [(DEV, 1), (LEGACY, 123), (None, 1), (DEV, True)])
def test_direct_execute_rejects_before_inputs(profile, seed):
    from transgenic.training.mvp_cli_runtime import execute
    with pytest.raises(ValueError, match='profile'):
        execute(SimpleNamespace(profile=profile, seed=seed), None, validate_only=True)


def test_direct_private_train_and_probe_model_reject_baseline_role_before_inputs():
    from transgenic.training.mvp_cli_runtime import _train
    from real_probes import load_model
    bindings = dict(profile=DEV, seed=123, run_role='b5_baseline')
    with pytest.raises(ValueError, match='run_role'):
        _train(123, 0, None, None, None, None, {'bindings': bindings}, 'a'*64)
    with pytest.raises(ValueError, match='run_role'):
        load_model(None, bindings)


def readiness_fixture(tmp_path, seed=123, profile=DEV):
    from test_mvp_cli import canonical, readiness_record
    from transgenic.training.mvp_cli import CandidateRecords
    from transgenic.training.mvp_cli_runtime import _expected_code_files
    from importlib.metadata import version
    paths = tuple(tmp_path / name for name in ('db', 'evidence', 'panel', 'sif', 'config', 'weights', 'run_config', 'manifest'))
    for path in paths[:-1]:
        path.write_text('{"seed":123}' if path.name == 'run_config' else path.name)
    canonical(paths[-1], {'files': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in _expected_code_files()},
                         'external_packages': {n: version(n) for n in ('accelerate', 'duckdb', 'safetensors', 'torch', 'transformers')}})
    candidate = CandidateRecords({}, {}, {}, dict(acceptance='a'*64, inventory='b'*64, receipts='c'*64))
    record = readiness_record(paths, candidate, seed=seed)
    if profile is not None:
        record['bindings'].update(profile=profile, run_role='mvp_finetune')
    return record, candidate, paths[:-1]


def validate_ready(tmp_path, record, candidate, paths, profile=DEV, seed=123):
    from transgenic.training.mvp_cli_runtime import validate_launch_readiness
    from transgenic.training.mvp_runtime import MVPRecipe
    path = tmp_path / 'ready.json'
    path.write_text(json.dumps(record))
    return validate_launch_readiness(path, hashlib.sha256(path.read_bytes()).hexdigest(), candidate,
                                     paths, seed=seed, profile=profile, recipe=MVPRecipe())


def test_readiness_dev_and_legacy(tmp_path):
    record, candidate, paths = readiness_fixture(tmp_path)
    assert validate_ready(tmp_path, record, candidate, paths)['bindings']['seed'] == 123
    record['bindings'].pop('profile')
    record['bindings'].pop('run_role')
    record['bindings']['seed'] = 1
    validate_ready(tmp_path, record, candidate, paths, LEGACY, 1)
    with pytest.raises(ValueError, match='profile'):
        validate_ready(tmp_path, record, candidate, paths)


@pytest.mark.parametrize('damage', ['missing', 'legacy', 'role', 'bool', 'float', 'unknown'])
def test_readiness_profile_tamper_rejected_even_rehashed(tmp_path, damage):
    record, candidate, paths = readiness_fixture(tmp_path)
    bindings = record['bindings']
    if damage == 'missing': del bindings['profile']
    if damage == 'legacy': bindings['profile'] = LEGACY
    if damage == 'role': bindings['run_role'] = 'b5_baseline'
    if damage == 'bool': bindings['seed'] = True
    if damage == 'float': bindings['seed'] = 123.0
    if damage == 'unknown': bindings['profile'] = None
    with pytest.raises(ValueError): validate_ready(tmp_path, record, candidate, paths)


def test_final_metadata_dev_preserves_production_hardware_guard():
    from test_mvp_finalize import metadata, summary
    from transgenic.training.mvp_finalize import _validate_metadata
    m = metadata('production')
    m.update(profile=DEV, run_role='mvp_finetune')
    m['seed_augmentation'] = dict(seed=123, aug_seed=20261037)
    _validate_metadata(m, summary('production'))
    for role in (None, 'b5_baseline'):
        m['run_role'] = role
        with pytest.raises(ValueError): _validate_metadata(m, summary('production'))
    m['run_role'] = 'mvp_finetune'
    m['runtime_hardware']['test_hardware'] = True
    with pytest.raises(ValueError): _validate_metadata(m, summary('production'))


def dev_budget():
    from test_mvp_resource_probe import fixture
    budget, ranks = fixture()
    budget.update(schema_version=2, profile=DEV, seeds=[123], max_training_node_hours=1)
    del budget['max_three_seed_node_hours']
    return budget, ranks


def test_development_budget_one_run_arithmetic():
    from resource_probe import project_budget
    budget, ranks = dev_budget()
    result = project_budget(budget, ranks)
    assert result['projected_run_seconds'] == 2460
    assert result['projected_training_node_hours'] == 2460 / 3600
    assert result['projected_campaign_node_hours'] == 2560 / 3600
    assert 'projected_three_seed_node_hours' not in result
    budget['max_training_node_hours'] = 2460 / 3600
    with pytest.raises(ValueError): project_budget(budget, ranks)


@pytest.mark.parametrize('key,value', [('schema_version', True), ('schema_version', 1),
    ('profile', LEGACY), ('profile', None), ('seeds', [1, 2, 3]), ('seeds', [123.0]),
    ('seeds', [True]), ('max_training_node_hours', True), ('max_training_node_hours', '1')])
def test_dev_budget_strict_schema(key, value):
    from resource_probe import validate_budget
    budget, _ = dev_budget()
    budget[key] = value
    with pytest.raises(ValueError): validate_budget(budget, 'a'*64)


def test_budget_binding_profile_must_match():
    from resource_probe import validate_budget
    budget, _ = dev_budget()
    validate_budget(budget, 'a'*64, bindings=dict(profile=DEV, seed=123))
    with pytest.raises(ValueError):
        validate_budget(budget, 'a'*64, bindings=dict(seed=1))


def test_training_wrapper_rejects_cross_profile_before_readiness(tmp_path):
    from run_training import training_arguments
    argv = ['--profile', DEV, '--seed', '1']
    for key in ('db', 'evidence-db', 'panel', 'acceptance-record', 'acceptance-sha256',
                'inventory-record', 'receipts-record', 'config', 'init-from', 'output-dir',
                'sif', 'launch-readiness', 'launch-readiness-sha256'):
        argv += ['--' + key, '/absent']
    path = tmp_path / 'argv.json'
    path.write_text(json.dumps(argv))
    with pytest.raises(ValueError, match='profile'):
        training_arguments(path, hashlib.sha256(path.read_bytes()).hexdigest())
    argv[3] = '123'
    path.write_text(json.dumps(argv))
    _, args = training_arguments(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert args.profile == DEV and args.seed == 123


@pytest.mark.parametrize('bad_profile,bad_seed,bad_role', [(LEGACY, 1, 'mvp_finetune'),
    (DEV, 123.0, 'mvp_finetune'), (DEV, 123, 'b5_baseline'), (None, 123, 'mvp_finetune')])
def test_scoped_receipt_rejects_wrong_measurement_identity(tmp_path, bad_profile, bad_seed, bad_role):
    import make_launch_readiness as r
    from test_mvp_readiness_tools import put, measurement
    name = 'real_tokenizer'
    measured = dict(measurement(name), profile=bad_profile, seed=bad_seed, run_role=bad_role)
    sha = put(tmp_path / (name + '.json'), {'pass': True, 'preflight': r.EVIDENCE_RECEIPTS[name],
        'bindings_sha256': 'a'*64, 'exit_code': 0, 'signal': None, 'timed_out': False, 'measurement': measured})
    (tmp_path / (name + '.DONE')).write_text(sha)
    with pytest.raises(ValueError):
        r._load_receipt(tmp_path, name, 'a'*64, bindings=dict(profile=DEV, seed=123, run_role='mvp_finetune'))


def test_prepare_generate_dev_receipts_cannot_replay_legacy(tmp_path, monkeypatch):
    import make_launch_readiness as r
    import resource_probe
    from test_mvp_readiness_tools import input_args, put, measurement
    # CPU identity test: resource measurement's CUDA certification is tested separately.
    monkeypatch.setattr(resource_probe, 'validate_resource', lambda value: None)
    monkeypatch.setenv('SLURM_JOB_ID', 'CPU_UNIT_FIXTURE')
    args, pins = input_args(tmp_path)
    args[-1] = '123'
    args += ['--profile', DEV]
    manifest, prepared, ready = (tmp_path / n for n in ('code.json', 'prepared.json', 'ready.json'))
    assert r.main(['prepare', *args, *pins, '--manifest-output', str(manifest), '--output', str(prepared)]) == 0
    record = json.loads(prepared.read_text())
    assert record['bindings']['profile'] == DEV
    assert record['bindings']['run_role'] == 'mvp_finetune'
    assert record['bindings']['recipe']['max_updates'] == 250
    for name, stage in r.EVIDENCE_RECEIPTS.items():
        value = dict(measurement(name), profile=DEV, seed=123, run_role='mvp_finetune')
        sha = put(tmp_path / (name + '.json'), {'pass': True, 'preflight': stage,
            'bindings_sha256': record['bindings_sha256'], 'exit_code': 0, 'signal': None,
            'timed_out': False, 'measurement': value})
        (tmp_path / (name + '.DONE')).write_text(sha)
    command = ['generate', *args, *pins, '--manifest-output', str(manifest), '--preflight-dir', str(tmp_path), '--output', str(ready)]
    assert r.main(command) == 0
    assert r.main(['check', *args, '--readiness', str(ready), '--readiness-sha256', r._sha256(ready)]) == 0
    legacy_args = args[:-2]
    legacy_args[-1] = '1'
    assert r.main(['generate', *legacy_args, *pins, '--manifest-output', str(manifest),
                   '--preflight-dir', str(tmp_path), '--output', str(tmp_path / 'legacy-ready.json')]) == 2
    assert not (tmp_path / 'legacy-ready.json').exists()
    assert r.main(['check', *legacy_args, '--readiness', str(ready), '--readiness-sha256', r._sha256(ready)]) == 2


@pytest.mark.parametrize('damage', ['missing_profile', 'wrong_profile', 'float_seed', 'role'])
def test_resource_receipt_dev_identity_is_required(tmp_path, monkeypatch, damage):
    import resource_probe as r
    import real_probes
    import transgenic.training.mvp_cli_runtime as runtime
    from test_mvp_readiness_tools import put
    monkeypatch.setattr(real_probes, 'validate_identity', lambda value: None)
    monkeypatch.setattr(runtime, '_validate_public_hardware', lambda value: value)
    budget, ranks = dev_budget()
    path = tmp_path / 'budget.json'
    sha = put(path, budget)
    value = dict(mode='resource_budget', profiling_complete=True, bindings_sha256='a'*64,
                 authorization_path=str(path), authorization_sha256=sha, ranks=ranks,
                 profile=DEV, seed=123, run_role='mvp_finetune', **r.project_budget(budget, ranks))
    r.validate_resource(value)
    if damage == 'missing_profile': del value['profile']
    if damage == 'wrong_profile': value['profile'] = LEGACY
    if damage == 'float_seed': value['seed'] = 123.0
    if damage == 'role': value['run_role'] = 'b5_baseline'
    with pytest.raises(ValueError): r.validate_resource(value)


@pytest.mark.parametrize('tamper', [False, True])
def test_final_publication_retains_profile_and_rechecks_role(tmp_path, tamper):
    from test_mvp_finalize import metadata, summary, LocalAccelerator, SaveToy, OneDataset
    from torch.utils.data import DataLoader
    from transgenic.training.mvp_finalize import evaluate_and_publish_final
    m = metadata()
    m.update(profile=DEV, run_role='mvp_finetune')
    m['seed_augmentation'] = dict(seed=123, aug_seed=20261037)
    def mutate():
        if tamper: m['run_role'] = 'b5_baseline'
    def run():
        evaluate_and_publish_final(LocalAccelerator(), SaveToy(), DataLoader(OneDataset(), batch_size=1),
            [11], tmp_path / 'final', metadata=m, training_summary=summary(), prepublication_prepare=mutate)
    if tamper:
        with pytest.raises(ValueError, match='run_role'): run()
        assert not (tmp_path / 'final').exists()
    else:
        run()
        result = json.loads((tmp_path / 'final' / 'metadata.json').read_text())
        assert result['profile'] == DEV and result['run_role'] == 'mvp_finetune'
        assert result['seed_augmentation'] == dict(seed=123, aug_seed=20261037)


def test_source_seed_float_rejected(tmp_path):
    from make_launch_readiness import _source_checkpoint_binding
    checkpoint = tmp_path / 'epoch_05'
    checkpoint.mkdir()
    (checkpoint / 'model.safetensors').write_bytes(b'x')
    (tmp_path / 'run_config.json').write_text('{"seed":123.0}')
    (tmp_path / 'runtime_identity.txt').write_text('runtime')
    with pytest.raises((ValueError, SystemExit)):
        _source_checkpoint_binding(str(checkpoint))


def test_probe_setup_rejects_invalid_profile_before_cuda(tmp_path, monkeypatch):
    import real_probes
    def forbidden_cuda():
        raise AssertionError('invalid profile reached CUDA')
    monkeypatch.setattr(real_probes, 'require_cuda', forbidden_cuda)
    from real_probes import setup
    bindings = dict(profile=DEV, seed=1)
    path = tmp_path / 'bindings.json'
    path.write_text(json.dumps(dict(bindings=bindings)))
    with pytest.raises(ValueError, match='profile'):
        setup(SimpleNamespace(bindings=str(path), bindings_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
