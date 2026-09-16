"""CPU-only fixtures; none of these receipts certify a real runtime."""
import hashlib
import importlib.util
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'revision/scripts/mvp'
sys.path.insert(0, str(TOOLS))
import make_launch_readiness as readiness
import stage_checkpoint as staging


def put(path, value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_receipt_rejects_unbound_pass(tmp_path):
    put(tmp_path / 'real_tokenizer.json', {'pass': True, 'preflight': 'p1_identity'})
    (tmp_path / 'real_tokenizer.DONE').write_text('complete\n')
    with pytest.raises((ValueError, SystemExit), match='binding|marker'):
        readiness._load_receipt(tmp_path, 'real_tokenizer', 'a' * 64)


def test_staging_requires_all_pins_before_copy(tmp_path):
    source = tmp_path / 'epoch_05'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'weights')
    put(tmp_path / 'run_config.json', {'seed': 123})
    (tmp_path / 'runtime_identity.txt').write_text('runtime')
    with pytest.raises(SystemExit):
        staging.main(['--source-epoch', str(source), '--dest', str(tmp_path / 'out')])
    assert not (tmp_path / 'out').exists()


def test_compute_guard_before_hashing(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('MVP_DATA_MOVER_ALLOCATION_ID', raising=False)
    assert hasattr(readiness, 'require_allocation'), 'large hashing lacks allocation guard'
    with pytest.raises(ValueError, match='allocation'):
        readiness.require_allocation()


def test_candidate_input_identity_rejected_during_prepare(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'CPU_UNIT_FIXTURE')
    args, pins = input_args(tmp_path)
    (tmp_path / 'db').write_text('not the accepted corpus')
    assert readiness.main(['prepare', *args, *pins, '--manifest-output', str(tmp_path / 'code'),
                           '--output', str(tmp_path / 'bindings')]) == 2
    assert not (tmp_path / 'bindings').exists()


def test_training_wrapper_rejects_validate_only(tmp_path):
    assert (TOOLS / 'run_training.py').exists(), 'bounded training wrapper missing'
    import run_training
    argv = tmp_path / 'train.json'
    sha = put(argv, ['--validate-only'])
    with pytest.raises(ValueError, match='validate-only'):
        run_training.training_arguments(argv, sha)


def test_step0_gate_rejects_tolerance_and_missing_tiers():
    assert (TOOLS / 'run_bounded.py').exists(), 'bounded preflight runner missing'
    import run_bounded
    with pytest.raises(ValueError):
        run_bounded.validate_step0({'pass': True})


def test_bounded_timeout_and_sigterm(tmp_path):
    assert (TOOLS / 'run_bounded.py').exists(), 'bounded runner missing'
    import run_bounded
    result = run_bounded.run_command([sys.executable, '-c', 'import time; time.sleep(60)'],
                                     tmp_path / 'timeout.log', 0.1)
    assert result['exit_code'] != 0 and result['timed_out']


def measurement(name):
    value = {'device': 'cuda', 'real_runtime': True}
    if name == 'real_hyena_step0_three_tiers':
        value.update(deterministic_algorithms=True, cublas_workspace_config=':4096:8',
                     comparison='bitwise', eval_mode=True,
                     tiers={tier: dict(input_sha256='b'*64, finite=True,
                         b5_logits_sha256='a'*64, mvp_off_logits_sha256='a'*64,
                         mvp_zero_projection_logits_sha256='a'*64, repeat_logits_sha256='a'*64)
                         for tier in ('30720', '61440', '129024')})
    elif name == 'four_rank_smoke':
        from test_mvp_cli import _gpu_fact
        value.update(hardware=[_gpu_fact(i) for i in range(4)], optimizer_steps=1, finite=True)
    elif name == 'resource_budget':
        value.update(measured_step_seconds=1, peak_memory_bytes=10, memory_limit_bytes=20,
                     authorized_wall_seconds=1000, projected_run_seconds=500,
                     authorization_sha256='c'*64)
    else:
        value.update({key: True for key in {
            'real_tokenizer': ('tokenizer_roundtrip', 'hyena_loaded'),
            'generation_cache_parity': ('cached_uncached_equal', 'evidence_off_equal'),
            'bf16_compile': ('native_bf16', 'inductor', 'finite', 'no_eager_fallback')}[name]})
    return value


def input_args(tmp_path, sidecars=None):
    # Standalone synthetic acceptance fixture: never a production receipt.
    from transgenic.training.mvp_cli import _ACCEPTANCE_CHECKS
    files = {}
    for name in ('db', 'evidence-db', 'panel', 'sif', 'config'):
        files[name] = tmp_path / name
        files[name].write_text(name)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    for name, value in (sidecars or {}).items():
        put(checkpoint / name, value)
    (checkpoint / 'model.safetensors').write_bytes(b'weights')
    put(tmp_path / 'run_config.json', {'seed': 123})
    (tmp_path / 'runtime_identity.txt').write_text('runtime')
    inventory = {'schema_version': 1, 'scope': 'exploratory_mvp_candidate',
                 'records': [{'rn': 1, 'chromosome': 'Chr1', 'start0': 0, 'end0': 2,
                              'split': 'valid', 'is_rc': False}],
                 'comparison': {'exact_ordered_identity_match': True, 'expected_count': 1, 'source_count': 1},
                 'counts': {'total': 1, 'train': 0, 'valid': 1,
                            'evidence_corpus_tile_all_splits': {'train': 0, 'valid': 1, 'test': 0}}}
    inv = tmp_path / 'inventory.json'
    rec = tmp_path / 'receipts.json'
    inv_sha = put(inv, inventory)
    rec_sha = put(rec, {'schema_version': 1, 'scope': 'exploratory_mvp_candidate',
                         'receipt_count': 0, 'receipts': []})
    acceptance = {'schema_version': 1, 'scope': 'exploratory_mvp_candidate',
        'decision': dict(technical_acceptance=True, production_training_accepted=False,
                        production_authorized=False, author_protocol_freeze=False, scientific_approval=False),
        'checks': dict.fromkeys(_ACCEPTANCE_CHECKS, True),
        'qc': dict(valid_tiles_total=1, valid_tiles_with_evidence=1,
                   retained_overlapping_panel=0, retained_overlapping_test=0),
        'source_counts': dict(members=1, collapsed_rule_a=0, collapsed_rule_b=0,
            molecules_after_dedup=1, retained=1, excluded_heldout_locus=0,
            excluded_test_block=0, excluded_unmapped=0),
        'companions': {'inventory': {'sha256': inv_sha}, 'receipts': {'sha256': rec_sha}},
        'inventory_summary': dict(companion_sha256=inv_sha, exact_match_to_evidence_corpus_tile=True,
                                 total=1, train=0, valid=1),
        'artifact': {'sha256': readiness._sha256(files['evidence-db'])},
        'inputs': {name: {'sha256': readiness._sha256(files[key])}
                   for name, key in [('corpus', 'db'), ('panel', 'panel')]}}
    acc = tmp_path / 'acceptance.json'
    acc_sha = put(acc, acceptance)
    args = [item for key, path in files.items() for item in ('--' + key, str(path))]
    args += ['--init-from', str(checkpoint), '--acceptance-record', str(acc),
             '--acceptance-sha256', acc_sha, '--inventory-record', str(inv),
             '--receipts-record', str(rec), '--seed', '1']
    pins = []
    for name, path in [('weights', checkpoint / 'model.safetensors'),
                       ('run-config', tmp_path / 'run_config.json'),
                       ('runtime-identity', tmp_path / 'runtime_identity.txt')]:
        pins += ['--expected-' + name + '-sha256', readiness._sha256(path)]
    return args, pins


def prepared(tmp_path, monkeypatch, sidecars=None):
    # These launcher/identity unit fixtures do not execute a CUDA resource probe.
    # Mock the resource-certification boundary explicitly rather than claiming
    # synthetic runtime flags meet it. Real schema negatives live separately in
    # test_mvp_resource_probe.py; no production bypass exists.
    import resource_probe
    monkeypatch.setattr(resource_probe, 'validate_resource', lambda value: None)
    monkeypatch.setenv('SLURM_JOB_ID', 'CPU_UNIT_FIXTURE')
    args, pins = input_args(tmp_path, sidecars)
    manifest = tmp_path / 'code.json'
    bindings = tmp_path / 'bindings.json'
    assert readiness.main(['prepare', *args, *pins, '--manifest-output', str(manifest),
                           '--output', str(bindings)]) == 0
    return args, pins, manifest, bindings


def test_generate_then_direct_consumer_and_changed_inputs(tmp_path, monkeypatch):
    args, pins, manifest, bindings = prepared(tmp_path, monkeypatch)
    binding_sha = json.loads(bindings.read_text())['bindings_sha256']
    receipts = tmp_path / 'preflight'
    receipts.mkdir()
    for name, stage in readiness.EVIDENCE_RECEIPTS.items():
        sha = put(
            receipts / (name + '.json'), {'pass': True, 'preflight': stage,
            'bindings_sha256': binding_sha, 'exit_code': 0, 'signal': None,
            'timed_out': False, 'measurement': measurement(name)})
        (receipts / (name + '.DONE')).write_text(sha + '\n')
    out = tmp_path / 'ready.json'
    command = ['generate', *args, *pins, '--manifest-output', str(manifest),
               '--preflight-dir', str(receipts), '--output', str(out)]
    assert readiness.main(command) == 0
    assert readiness.main(['check', *args, '--readiness', str(out),
                           '--readiness-sha256', readiness._sha256(out)]) == 0
    (tmp_path / 'sif').write_text('CHANGED')
    assert readiness.main(['check', *args, '--readiness', str(out),
                           '--readiness-sha256', readiness._sha256(out)]) == 2
    out.unlink()
    assert readiness.main(command) == 2
    assert not out.exists()


@pytest.mark.parametrize('mutation', ['missing', 'unequal', 'nonfinite', 'nondeterministic'])
def test_step0_measurements_fail_closed(mutation):
    import run_bounded
    value = measurement('real_hyena_step0_three_tiers')
    run_bounded.validate_step0(value)
    if mutation == 'missing':
        del value['tiers']['129024']
    elif mutation == 'unequal':
        value['tiers']['30720']['repeat_logits_sha256'] = 'd'*64
    elif mutation == 'nonfinite':
        value['tiers']['61440']['finite'] = False
    else:
        value['deterministic_algorithms'] = False
    with pytest.raises(ValueError):
        run_bounded.validate_step0(value)


def stage_args(tmp_path, monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'CPU_UNIT_FIXTURE')
    source = tmp_path / 'epoch_05'
    source.mkdir()
    (source / 'model.safetensors').write_bytes(b'weights')
    put(tmp_path / 'run_config.json', {'seed': 123})
    (tmp_path / 'runtime_identity.txt').write_text('runtime')
    quota = tmp_path / 'quota.json'
    sha = put(quota, {'path': str(tmp_path), 'available_bytes': 100000,
                      'measured_at_unix': time.time()})
    args = ['--source-epoch', str(source), '--dest', str(tmp_path / 'staged'),
            '--quota-record', str(quota), '--quota-sha256', sha,
            '--max-copy-bytes', '1000', '--reserve-bytes', '1000']
    for prefix, path in [('weights', source / 'model.safetensors'),
                         ('run-config', tmp_path / 'run_config.json'),
                         ('runtime-identity', tmp_path / 'runtime_identity.txt')]:
        args += ['--expected-' + prefix + '-sha256', readiness._sha256(path)]
    return args


def test_staged_checkpoint_layout_and_collision(tmp_path, monkeypatch):
    from transgenic.training.mvp_cli_runtime import _source_paths
    args = stage_args(tmp_path, monkeypatch)
    assert staging.main(args) == 0
    root = tmp_path / 'staged'
    weights, config = _source_paths(root / 'checkpoint')
    assert weights.read_bytes() == b'weights'
    assert json.loads(config.read_text())['seed'] == 123
    assert (root / 'runtime_identity.txt').read_text() == 'runtime'
    assert (root / 'STAGING_DONE').read_text().strip() == readiness._sha256(root / 'staging_receipt.json')
    with pytest.raises(SystemExit):
        staging.main(args)


@pytest.mark.parametrize('mutation', ['quota', 'pin', 'alias', 'budget'])
def test_staging_rejects_unapproved_copy(tmp_path, monkeypatch, mutation):
    args = stage_args(tmp_path, monkeypatch)
    if mutation == 'quota':
        args[args.index('--reserve-bytes') + 1] = '100000000'
    elif mutation == 'budget':
        args[args.index('--max-copy-bytes') + 1] = '1'
    elif mutation == 'pin':
        args[args.index('--expected-weights-sha256') + 1] = '0'*64
    else:
        (tmp_path / 'alias').symlink_to(tmp_path / 'epoch_05')
        args[args.index('--source-epoch') + 1] = str(tmp_path / 'alias')
    with pytest.raises((ValueError, SystemExit)):
        staging.main(args)
    assert not (tmp_path / 'staged').exists()


@pytest.mark.parametrize('mutation', ['wrong_binding', 'bad_marker', 'failed', 'nonzero', 'timeout'])
def test_receipt_execution_and_identity_rejected(tmp_path, mutation):
    name = 'real_tokenizer'
    value = {'pass': True, 'preflight': 'p1_identity', 'bindings_sha256': 'a'*64,
             'exit_code': 0, 'signal': None, 'timed_out': False, 'measurement': measurement(name)}
    if mutation == 'wrong_binding':
        value['bindings_sha256'] = 'b'*64
    elif mutation == 'nonzero':
        value['exit_code'] = 1
    elif mutation == 'timeout':
        value['timed_out'] = True
    sha = put(tmp_path / (name + '.json'), value)
    (tmp_path / (name + '.DONE')).write_text('0'*64 if mutation == 'bad_marker' else sha)
    if mutation == 'failed':
        put(tmp_path / (name + '.FAILED.json'), {'pass': False})
    with pytest.raises(ValueError):
        readiness._load_receipt(tmp_path, name, 'a'*64)


def test_probe_retries_only_warmup_then_publishes_hash_marker(tmp_path, monkeypatch):
    import run_bounded
    _, _, _, bindings = prepared(tmp_path, monkeypatch)
    counter = tmp_path / 'counter'
    warmup = tmp_path / 'warmup.json'
    warm_sha = put(warmup, [sys.executable, '-c',
        f'from pathlib import Path; p=Path({str(counter)!r}); n=int(p.read_text()) if p.exists() else 0; p.write_text(str(n+1)); raise SystemExit(75 if n==0 else 0)'])
    command = tmp_path / 'probe.json'
    value = measurement('real_tokenizer')
    sha = put(command, [sys.executable, '-c',
        'import json,os; from pathlib import Path; '
        f'v={value!r}; v["bindings_sha256"]=os.environ["MVP_BINDINGS_SHA256"]; '
        'Path(os.environ["MVP_MEASUREMENT_OUTPUT"]).write_text(json.dumps(v))'])
    out = tmp_path / 'run'
    assert run_bounded.main(['--name', 'real_tokenizer', '--bindings', str(bindings),
        '--bindings-file-sha256', readiness._sha256(bindings), '--command-file', str(command),
        '--command-sha256', sha, '--output-dir', str(out), '--timeout-seconds', '5',
        '--warmup-command-file', str(warmup), '--warmup-sha256', warm_sha, '--warmup-attempts', '2']) == 0
    assert counter.read_text() == '2'
    assert (out / 'real_tokenizer.DONE').read_text().strip() == readiness._sha256(out / 'real_tokenizer.json')


def launch_fixture(tmp_path, monkeypatch, sidecars=None):
    args, pins, manifest, bindings = prepared(tmp_path, monkeypatch, sidecars)
    budget = tmp_path / 'budget.json'
    budget_sha = put(budget, {'authorized_scope': 'exploratory_mvp', 'seeds': [1], 'max_wall_seconds': 1000})
    for name, stage in readiness.EVIDENCE_RECEIPTS.items():
        value = measurement(name)
        if name == 'resource_budget':
            value['authorization_sha256'] = budget_sha
        sha = put(tmp_path / (name + '.json'), {'pass': True, 'preflight': stage,
            'bindings_sha256': json.loads(bindings.read_text())['bindings_sha256'],
            'exit_code': 0, 'signal': None, 'timed_out': False, 'measurement': value})
        (tmp_path / (name + '.DONE')).write_text(sha)
    ready = tmp_path / 'ready.json'
    assert readiness.main(['generate', *args, *pins, '--manifest-output', str(manifest),
                           '--preflight-dir', str(tmp_path), '--output', str(ready)]) == 0
    return args, ready, budget, budget_sha


@pytest.mark.parametrize('mutation', ['seed', 'recipe', 'evidence', 'source', 'corpus', 'sif'])
def test_direct_validator_rejects_modified_generated_record(tmp_path, monkeypatch, mutation):
    from transgenic.training.mvp_cli import validate_candidate_records
    from transgenic.training.mvp_cli_runtime import _source_paths, validate_launch_readiness
    from transgenic.training.mvp_runtime import MVPRecipe
    args, ready, _, _ = launch_fixture(tmp_path, monkeypatch)
    parsed = readiness.build_parser().parse_args(['check', *args, '--readiness', str(ready),
                                                 '--readiness-sha256', readiness._sha256(ready)])
    candidate = validate_candidate_records(parsed.acceptance_record, parsed.acceptance_sha256,
                                           parsed.inventory_record, parsed.receipts_record)
    paths = tuple(Path(getattr(parsed, key)) for key in ('db', 'evidence_db', 'panel', 'sif', 'config')) + _source_paths(parsed.init_from)
    record = json.loads(ready.read_text())
    if mutation == 'seed':
        record['bindings']['seed'] = 2
    elif mutation == 'recipe':
        record['bindings']['recipe'] = {}
    elif mutation == 'evidence':
        del record['evidence']['bf16_compile']
    elif mutation == 'source':
        record['bindings']['source_checkpoint']['weights_sha256'] = '0'*64
    else:
        record['bindings'][mutation + '_sha256'] = '0'*64
    sha = put(ready, record)
    with pytest.raises(ValueError):
        validate_launch_readiness(ready, sha, candidate, paths, seed=1, recipe=MVPRecipe())


@pytest.mark.parametrize('outcome', ['success', 'sigterm', 'warmup-failure', 'transient', 'exhausted'])
def test_training_wrapper_single_attempt_and_failure_semantics(tmp_path, monkeypatch, outcome):
    import run_training
    args, ready, budget, budget_sha = launch_fixture(tmp_path, monkeypatch)
    argv_path = tmp_path / 'training-argv.json'
    sha = put(argv_path, [*args, '--output-dir', str(tmp_path / 'model'),
                         '--launch-readiness', str(ready), '--launch-readiness-sha256', readiness._sha256(ready)])
    commands = []
    # Substitute only process execution (would otherwise require real GPUs).
    def child(argv, log, timeout):
        commands.append(argv)
        if outcome == 'warmup-failure':
            return dict(exit_code=1, signal=None, timed_out=False)
        if outcome == 'exhausted' or (outcome == 'transient' and len(commands) <= 2):
            return dict(exit_code=75, signal=None, timed_out=False)
        if outcome == 'sigterm' and len(commands) == 2:
            return dict(exit_code=-15, signal=15, timed_out=False)
        return dict(exit_code=0, signal=None, timed_out=False)
    monkeypatch.setattr(run_training, 'run_command', child)
    status = tmp_path / 'status'
    result = run_training.main(['--arguments', str(argv_path), '--arguments-sha256', sha,
        '--budget-authorization', str(budget), '--budget-sha256', budget_sha,
        '--timeout-seconds', '600', '--status-dir', str(status)])
    assert len(commands) == {'warmup-failure': 1, 'exhausted': 3, 'transient': 4}.get(outcome, 2)
    if outcome not in {'warmup-failure', 'exhausted'}:
        command = commands[-1]
        assert command[:10] == [sys.executable, '-m', 'accelerate.commands.launch', '--multi_gpu',
                               '--num_machines', '1', '--num_processes', '4', '--mixed_precision', 'bf16']
    if outcome in {'success', 'transient'}:
        assert result == 0 and (status / 'PROCESS_EXIT.json').exists()
    else:
        assert result == 2 and not (status / 'PROCESS_EXIT.json').exists()
        assert json.loads((status / 'FAILED.json').read_text())['resume_supported'] is False
    assert not (status / 'DONE').exists()


def test_probe_sigterm_publishes_only_failure(tmp_path, monkeypatch):
    _, _, _, bindings = prepared(tmp_path, monkeypatch)
    command = tmp_path / 'command.json'
    sha = put(command, [sys.executable, '-c', 'import time; print("started", flush=True); time.sleep(60)'])
    out = tmp_path / 'probe'
    child = subprocess.Popen([sys.executable, str(TOOLS / 'run_bounded.py'), '--name', 'real_tokenizer',
        '--bindings', str(bindings), '--bindings-file-sha256', readiness._sha256(bindings),
        '--command-file', str(command), '--command-sha256', sha, '--output-dir', str(out),
        '--timeout-seconds', '20'], env=dict(os.environ, PYTHONPATH=str(ROOT / '.testdeps') + ':' + str(ROOT / 'src')))
    try:
        deadline = time.monotonic() + 10
        while not (out / 'probe.log').exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (out / 'probe.log').exists()
        time.sleep(0.15)
        child.send_signal(signal.SIGTERM)
        assert child.wait(timeout=15) == 2
        failure = json.loads((out / 'real_tokenizer.FAILED.json').read_text())
        assert failure['pass'] is False and failure['resume_supported'] is False
        assert failure['signal'] == signal.SIGTERM
        assert not (out / 'real_tokenizer.DONE').exists()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
