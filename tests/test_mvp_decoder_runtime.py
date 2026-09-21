"""CPU fixtures only: never historical CUDA receipts or SIF attestation."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision/scripts/mvp'))
import mvp_decode_runtime as r


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def bounded(tmp_path, monkeypatch):
    """Current pinned runner/tooling; only measurement/allocation are synthetic."""
    modules = {}
    authority = tmp_path / 'authority/revision/scripts/mvp'
    authority.mkdir(parents=True)
    for filename in ('make_launch_readiness.py', 'real_probes.py', 'run_bounded.py',
                     'resource_probe.py', 'stage_checkpoint.py', 'run_training.py', 'accelerate_mvp.json'):
        (authority / filename).write_bytes((r.ROOT / 'revision/scripts/mvp' / filename).read_bytes())
    for name in ('make_launch_readiness', 'real_probes', 'resource_probe', 'run_bounded'):
        path = authority / (name + '.py')
        if path.name in r.PROBE_CODE:
            assert sha(path) == r.PROBE_CODE[path.name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    runner = modules['run_bounded']
    # Allocation and measurement are CPU substitutions, NOT operational evidence.
    # Tooling validation is genuine, including resource_probe and all seven files.
    monkeypatch.setattr(runner, 'require_allocation', lambda: 'CPU_FIXTURE_NO_ALLOCATION')
    bindings = {'tooling': runner.tooling_identity()}
    assert len(bindings['tooling']) == 7
    assert str(authority / 'resource_probe.py') in bindings['tooling']
    binding_sha = hashlib.sha256(runner._canonical_bytes(bindings)).hexdigest()
    bindings_path = dump(tmp_path / 'bindings.json', dict(bindings=bindings, bindings_sha256=binding_sha))
    assets = tmp_path / 'assets'; assets.mkdir(); (assets / 'fixture.txt').write_text('CPU FIXTURE')
    measurement = dict(mode='generation_cache_parity', device='cuda', real_runtime=True,
        fixture='SYNTHETIC_CPU_NOT_GPU_EVIDENCE', cached_uncached_equal=True, evidence_off_equal=True,
        nonzero_projection=True, reorder_indices_sha256=['fixture'], runtime={'versions': r.VERSIONS},
        probe_identity=dict(schema_version=1, probe_path=modules['real_probes'].__file__,
            imports={m.__file__: sha(m.__file__) for n,m in modules.items() if n != 'run_bounded'},
            asset_roots={str(assets): r.tree_identity(assets)}))
    script = tmp_path / 'synthetic_measurement.py'
    script.write_text('import json, os\nv = ' + repr(measurement) + '\n'
        "v['bindings_sha256'] = os.environ['MVP_BINDINGS_SHA256']\n"
        "open(os.environ['MVP_MEASUREMENT_OUTPUT'], 'w').write(json.dumps(v))\n")
    command = dump(tmp_path / 'command.json', [sys.executable, str(script)])
    out = tmp_path / 'publication'
    assert runner.main(['--name', 'generation_cache_parity', '--bindings', str(bindings_path),
        '--bindings-file-sha256', sha(bindings_path), '--command-file', str(command),
        '--command-sha256', sha(command), '--output-dir', str(out), '--timeout-seconds', '20']) == 0
    return out / 'generation_cache_parity.json'


def spec_for(path):
    return {'receipt': {'path': str(path), 'sha256': sha(path)}}


def test_real_bounded_publication(bounded):
    measurement, records = r.completed_measurement(spec_for(bounded))
    assert measurement['fixture'] == 'SYNTHETIC_CPU_NOT_GPU_EVIDENCE'
    assert len(records) == 2
    # Official wrapper now gets through completion to the actual runtime gate.
    with pytest.raises(ValueError, match='runtime version pin'):
        r.verify_runtime(spec_for(bounded), {})


@pytest.mark.parametrize('mutation', ['missing_done', 'bad_done', 'failed', 'raw', 'binding',
    'exit_code', 'timed_out', 'signal', 'pass', 'schema_version', 'preflight', 'receipt_sha'])
def test_completed_negatives(bounded, mutation):
    spec = spec_for(bounded)
    wrapper = json.loads(bounded.read_text())
    if mutation == 'missing_done': bounded.with_suffix('.DONE').unlink()
    elif mutation == 'bad_done': bounded.with_suffix('.DONE').write_text(sha(bounded))
    elif mutation == 'failed': bounded.with_suffix('.FAILED.json').write_text('{}')
    elif mutation == 'receipt_sha': spec['receipt']['sha256'] = '0' * 64
    else:
        if mutation == 'raw': wrapper = wrapper['measurement']
        elif mutation == 'binding': wrapper['measurement']['bindings_sha256'] = '0' * 64
        else: wrapper[mutation] = {'exit_code': 1, 'timed_out': True, 'signal': 15,
            'pass': False, 'schema_version': 2, 'preflight': 'wrong'}[mutation]
        dump(bounded, wrapper)
        spec = spec_for(bounded)
        bounded.with_suffix('.DONE').write_text(sha(bounded) + '\n')
    with pytest.raises(ValueError): r.completed_measurement(spec)


@pytest.mark.parametrize('filename', ['modeling_HyenaTransgenic.py', 'configuration_transgenic.py',
                                     'tokenization_transgenic.py', 'encoder_init.py'])
def test_relocated_project_not_root_exempt(tmp_path, monkeypatch, filename):
    current = tmp_path / 'current'; old = tmp_path / 'probed'
    relative = Path('src/transgenic/model') / filename
    for root in (current, old):
        (root / relative).parent.mkdir(parents=True)
        (root / relative).write_text('original = True\n')
    identity = {'imports': {str(old / relative): sha(old / relative)}}
    monkeypatch.setattr(r, 'ROOT', current)
    monkeypatch.setattr(r, 'sys', SimpleNamespace(modules={'model': SimpleNamespace(__file__=str(current / relative))}))
    assert r.verify_loaded_modules(identity, set(), project_root=old)
    (current / relative).write_text('changed = True\n')
    with pytest.raises(ValueError, match='project implementation'):
        r.verify_loaded_modules(identity, set(), project_root=old)
    # A decoder audit cannot override an identity already consumed by A.
    with pytest.raises(ValueError, match='project implementation'):
        r.verify_loaded_modules(identity, set(), project_root=old, decoder_code={str(relative): sha(current / relative)})


@pytest.mark.parametrize('stale_name,stale_hash', [
    (None, None),
    ('real_probes.py', 'a40b46b3e2df1f5a80a4357d06c0e77bf1cdedd66c0fd22582df7de0541bbc58'),
    ('make_launch_readiness.py', '73728aee67611575448722cb4180062917ee0c86330749ff1ca648346e75a6d5'),
])
def test_probe_handshake_and_operator_code_pin(tmp_path, stale_name, stale_hash):
    # Independent TRAIN_REVIEW.md 2026-09-17 authority, not receipt-selected pins.
    approved = {
        'real_probes.py': '7f9507f4fced687e1886c659224d4b778b3486a2b390dac1e8f3c6954e46c7ae',
        'run_bounded.py': '8b2f4042e03732c16d633fb66444b1b91e87cef54d3c696104775beebf19a932',
        'make_launch_readiness.py': 'a5b2c895bd76699718655993d67430b9459c5db07d5f47636feb6acc77ce6de2',
    }
    probe = tmp_path / 'revision/scripts/mvp/real_probes.py'
    identity = {'probe_path': str(probe), 'imports': {str(probe.with_name(n)): h for n,h in approved.items()}}
    closure = {p: sha(r.ROOT / p) for p in r.DECODER_ONLY}
    pin = hashlib.sha256((json.dumps(closure, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
    if stale_name is not None:
        identity['imports'][str(probe.with_name(stale_name))] = stale_hash
        with pytest.raises(ValueError, match='unreviewed probe/handshake'):
            r.project_closure(identity, {'decoder_code_sha256': pin})
        return
    assert r.project_closure(identity, {'decoder_code_sha256': pin})[0] == tmp_path
    without_runner = dict(identity, imports={p:h for p,h in identity['imports'].items() if not p.endswith('/run_bounded.py')})
    assert r.project_closure(without_runner, {'decoder_code_sha256': pin})[0] == tmp_path
    with pytest.raises(ValueError, match='operator decoder code pin'):
        r.project_closure(identity, {'decoder_code_sha256': '0' * 64})
    for name in approved:
        changed = dict(identity, imports=dict(identity['imports']))
        changed['imports'][str(probe.with_name(name))] = '0' * 64
        with pytest.raises(ValueError, match='unreviewed probe/handshake'):
            r.project_closure(changed, {'decoder_code_sha256': pin})


@pytest.mark.parametrize('mutation', ['valid', 'nonempty_init', 'unbound', 'relocation', 'implementation', 'wrong_name'])
def test_hf_scaffolding(tmp_path, monkeypatch, mutation):
    cache = tmp_path / 'modules'; root = cache / 'transformers_modules'; root.mkdir(parents=True)
    path = root / '__init__.py'; path.write_bytes(b'')
    name = 'transformers_modules'
    identity = {'imports': {str(path): sha(path)}}
    if mutation == 'nonempty_init': path.write_text('x=1'); identity['imports'][str(path)] = sha(path)
    if mutation == 'unbound': identity['imports'] = {}
    if mutation == 'relocation': cache = tmp_path / 'other'
    if mutation == 'implementation':
        path = root / 'impl.py'; path.write_text('x=1'); name += '.impl'; identity['imports'][str(path)] = sha(path)
    if mutation == 'wrong_name': name += '.wrong'
    monkeypatch.setattr(r, 'sys', SimpleNamespace(modules={name: SimpleNamespace(__file__=str(path))}))
    if mutation == 'valid': assert r.verify_loaded_modules(identity, set(), module_cache=cache)
    else:
        with pytest.raises(ValueError): r.verify_loaded_modules(identity, set(), module_cache=cache)


def test_installed_transformers_dynamic_import(tmp_path, monkeypatch):
    import transformers
    from transformers import dynamic_module_utils as dynamic
    cache = tmp_path / 'hf_modules'
    monkeypatch.setattr(dynamic, 'HF_MODULES_CACHE', str(cache))
    # Restore process-global imports/sys.path after this real CPU loader exercise.
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for name in list(sys.modules):
        if name == 'transformers_modules' or name.startswith('transformers_modules.'):
            monkeypatch.delitem(sys.modules, name)
    source = tmp_path / 'fixture_model'; source.mkdir()
    (source / 'modeling_fixture.py').write_text('class FixtureModel:\n    pass\n')
    cls = dynamic.get_class_from_dynamic_module('modeling_fixture.FixtureModel', str(source), local_files_only=True)
    modules = {n: m for n,m in sys.modules.items() if n == 'transformers_modules' or n.startswith('transformers_modules.')}
    try:
        assert cls.__module__ in modules
        identity = {'imports': {str(Path(m.__file__).resolve()): sha(m.__file__) for m in modules.values()}}
        monkeypatch.setattr(r, 'sys', SimpleNamespace(modules=modules))
        assert r.verify_loaded_modules(identity, {sha(source / 'modeling_fixture.py')}, module_cache=cache)
        assert (cache / 'transformers_modules/__init__.py').read_bytes() == b''
        print('CPU dynamic import transformers=' + transformers.__version__ + '; not SIF evidence')
    finally:
        for name in modules: sys.modules.pop(name, None)


def test_pinned_transformers_4576_loader_contract(tmp_path, monkeypatch):
    import transformers
    if transformers.__version__ != '4.57.6':
        pytest.skip('pinned 4.57.6 not installed; CPU 4.46.3 test is not SIF validation')
    test_installed_transformers_dynamic_import(tmp_path, monkeypatch)


@pytest.mark.parametrize('name', list(r.PROBE_CODE))
def test_old_d99_handshake_rejected(tmp_path, name):
    probe = tmp_path / 'revision/scripts/mvp/real_probes.py'
    imports = {str(probe.with_name(n)): h for n,h in r.PROBE_CODE.items()}
    imports[str(probe.with_name(name))] = hashlib.sha256(subprocess.check_output(
        ['git', 'show', 'd99b1f8:revision/scripts/mvp/' + name], cwd=r.ROOT)).hexdigest()
    with pytest.raises(ValueError, match='unreviewed probe/handshake'):
        r.project_closure({'probe_path': str(probe), 'imports': imports}, {})


def test_bootstrap_is_explicit_decoder_dependency():
    # Producer imports this new numerical-count validator before runtime preflight.
    import mvp_produce_counts as producer
    module = sys.modules[producer.validate_paired_record.__module__]
    assert str(Path(module.__file__).resolve().relative_to(r.ROOT)) in r.DECODER_ONLY


def test_panel_compatibility_live_import_requires_operator_closure_and_probe_precedence(tmp_path, monkeypatch):
    import mvp_panel_compatibility as compat
    relative = str(Path(compat.__file__).resolve().relative_to(r.ROOT))
    probe = tmp_path / 'revision/scripts/mvp/real_probes.py'
    identity = {'probe_path': str(probe), 'imports': {str(probe.with_name(n)): h for n, h in r.PROBE_CODE.items()}}
    closure = {p: sha(r.ROOT / p) for p in r.DECODER_ONLY}
    pin = hashlib.sha256((json.dumps(closure, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
    old, audited = r.project_closure(identity, {'decoder_code_sha256': pin})
    monkeypatch.setattr(r, 'sys', SimpleNamespace(modules={'mvp_panel_compatibility': compat}))
    assert r.verify_loaded_modules(identity, set(), project_root=old, decoder_code=audited)
    bad = dict(audited, **{relative: '0'*64})
    with pytest.raises(ValueError, match='project implementation'):
        r.verify_loaded_modules(identity, set(), project_root=old, decoder_code=bad)
    identity['imports'][str(old / relative)] = '0'*64
    with pytest.raises(ValueError, match='project implementation'):
        r.verify_loaded_modules(identity, set(), project_root=old, decoder_code=audited)


def test_cpu_device_refused():
    with pytest.raises(ValueError, match='operational CUDA'): r.require_cuda_device('cpu')


def test_unknown_project_file_not_decoder_audited(tmp_path, monkeypatch):
    file = tmp_path / 'unknown.py'; file.write_text('x=1')
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'sys', SimpleNamespace(modules={'unknown': SimpleNamespace(__file__=str(file))}))
    with pytest.raises(ValueError, match='project implementation'):
        r.verify_loaded_modules({'imports': {}}, set(), project_root=tmp_path / 'old', decoder_code={})


def test_all_declared_bytes_and_handshake_checked(bounded, monkeypatch, tmp_path):
    import torch
    import transformers
    import accelerate
    # Explicit CPU-only runtime-gate substitutions: never a genuine CUDA PASS.
    versions = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                    transformers=transformers.__version__, accelerate=accelerate.__version__)
    monkeypatch.setattr(r, 'VERSIONS', versions)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1'); monkeypatch.setenv('TRANSFORMERS_OFFLINE', '1')
    sif = tmp_path / 'fixture.sif'; sif.write_text('NOT A REAL SIF')
    monkeypatch.setattr(r, 'SIF', sha(sif)); monkeypatch.setenv('APPTAINER_CONTAINER', str(sif))
    unused = tmp_path / 'declared_but_not_loaded.py'; unused.write_text('original = 1')
    wrapper = json.loads(bounded.read_text())
    measurement = wrapper['measurement']; measurement['runtime']['versions'] = versions
    measurement['probe_identity']['imports'][str(unused)] = sha(unused)
    dump(bounded, wrapper); bounded.with_suffix('.DONE').write_text(sha(bounded) + '\n')
    closure = {p: sha(r.ROOT / p) for p in r.DECODER_ONLY}
    pin = hashlib.sha256((json.dumps(closure, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
    spec = dict(spec_for(bounded), sif={'path': str(sif), 'sha256': sha(sif)}, decoder_code_sha256=pin)
    with pytest.raises(ValueError, match='actual model/tokenizer caches unbound'):
        r.verify_runtime(spec, {})
    unused.write_text('changed = 2')
    with pytest.raises(ValueError, match='SHA256'):
        r.verify_runtime(spec, {})
    unused.write_text('original = 1')
    runner = Path(measurement['probe_identity']['probe_path']).with_name('run_bounded.py')
    runner.write_text('changed handshake')
    with pytest.raises(ValueError, match='SHA256'):
        r.verify_runtime(spec, {})
