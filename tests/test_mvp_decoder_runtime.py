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
    """Execute real d99b1f8 runner publication with openly synthetic measurement."""
    modules = {}
    authority = tmp_path / 'authority/revision/scripts/mvp'
    authority.mkdir(parents=True)
    for name in ('make_launch_readiness', 'real_probes', 'run_bounded'):
        path = authority / (name + '.py')
        path.write_bytes(subprocess.check_output(['git', 'show',
            'd99b1f8:revision/scripts/mvp/' + path.name], cwd=r.ROOT))
        assert sha(path) == r.PROBE_CODE[path.name]
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules[name] = module
    runner = modules['run_bounded']
    # Allocation/tooling are CPU fixture substitutions, not operational evidence.
    monkeypatch.setattr(runner, 'require_allocation', lambda: 'CPU_FIXTURE_NO_ALLOCATION')
    monkeypatch.setattr(runner, 'tooling_identity', lambda: {'fixture': True})
    bindings = {'tooling': {'fixture': True}}
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


def test_probe_handshake_and_operator_code_pin(tmp_path):
    probe = tmp_path / 'revision/scripts/mvp/real_probes.py'
    identity = {'probe_path': str(probe), 'imports': {str(probe.with_name(n)): h for n,h in r.PROBE_CODE.items()}}
    closure = {p: sha(r.ROOT / p) for p in r.DECODER_ONLY}
    pin = hashlib.sha256((json.dumps(closure, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
    assert r.project_closure(identity, {'decoder_code_sha256': pin})[0] == tmp_path
    without_runner = dict(identity, imports={p:h for p,h in identity['imports'].items() if not p.endswith('/run_bounded.py')})
    assert r.project_closure(without_runner, {'decoder_code_sha256': pin})[0] == tmp_path
    with pytest.raises(ValueError, match='operator decoder code pin'):
        r.project_closure(identity, {'decoder_code_sha256': '0' * 64})
    for name in r.PROBE_CODE:
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
