"""Mandatory operator-pinned real generation preflight and live runtime closure.

A receipt is trusted operator input, not cryptographic proof of execution. CPU
fixtures cannot satisfy the real runtime checks. No unverified override exists.
"""
import hashlib
import json
import os
from pathlib import Path
import sys

from transgenic.evidence import mvp_decode as d

COMMIT = '0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446'
SIF = '1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6'
VERSIONS = dict(torch='2.14.0a0+4fdf77b940.nv26.08', cuda='13.4', transformers='4.57.6', accelerate='1.15.0')
ROOT = Path(__file__).resolve().parents[3]
# Reviewed fixed-A d99b1f8 Git bytes, NOT receipt-selected executable hashes.
PROBE_CODE = {
    'real_probes.py': '2c75dc4b31752fe39d4fa4ad7550b53ac735df1aafac89e6d307e94810ba5f3c',
    'run_bounded.py': '307f2ce9df0c7d4fb0c06b1ffe3149da77ae26c987c4b4061e8e63b4bb7ba867',
    'make_launch_readiness.py': 'ee303a72a079b02ce1d1b3fe1d24a2cb32f51f214baea129bc19f504c484a9b3',
}
# Decoder orchestration/grammar/transport, not A's numerical model implementation.
# Exact closed boundary: no directory exemptions; independently operator-pinned.
DECODER_ONLY = (
    'revision/scripts/mvp/mvp_decode_panel.py',
    'revision/scripts/mvp/mvp_decode_runtime.py',
    'revision/scripts/mvp/mvp_verify_panel_inputs.py',
    'revision/scripts/mvp/mvp_prepare_panel_inputs.py',
    'revision/scripts/mvp/mvp_produce_counts.py',
    'src/transgenic/evidence/mvp_decode.py',
    'src/transgenic/evidence/mvp_windows.py',
    'src/transgenic/evidence/mvp_tracks.py',
    'src/transgenic/model/constrained_decoding.py',
    'src/transgenic/utils/gsf.py',
    'src/transgenic/utils/gsf_contract.py',
    'src/transgenic/utils/gsf_grammar.py',
)


def completed_measurement(spec):
    record = d.checked_file(spec['receipt'])
    path = Path(record['path'])
    d.require(path.name == 'generation_cache_parity.json', 'completed bounded receipt required')
    d.require(not path.with_suffix('.FAILED.json').exists() and
              not path.with_suffix('.FAILED.json').is_symlink(), 'failed bounded receipt')
    done = path.with_suffix('.DONE')
    d.require(done.is_file() and done.read_bytes() == (record['sha256'] + '\n').encode(),
              'completed bounded DONE digest required')
    wrapper = d.read_json(path)
    d.require(type(wrapper.get('schema_version')) is int and wrapper['schema_version'] == 1 and
              wrapper.get('preflight') == 'p3_generation_parity' and
              type(wrapper.get('exit_code')) is int and wrapper['exit_code'] == 0 and
              wrapper.get('timed_out') is False and 'signal' in wrapper and
              wrapper['signal'] is None and wrapper.get('pass') is True,
              'successful bounded wrapper required')
    measurement = wrapper.get('measurement')
    binding = wrapper.get('bindings_sha256')
    d.require(isinstance(binding, str) and len(binding) == 64 and
              all(c in '0123456789abcdef' for c in binding) and
              isinstance(measurement, dict) and measurement.get('bindings_sha256') == binding,
              'wrapper measurement binding mismatch')
    return measurement, [record, d.identity(done)]


def project_closure(identity, spec):
    probe = Path(identity['probe_path'])
    d.require(probe.is_absolute() and probe.parts[-4:] ==
              ('revision', 'scripts', 'mvp', 'real_probes.py'), 'unsupported probe checkout')
    old_root = probe.parents[3]
    for name, digest in PROBE_CODE.items():
        # A snapshots imports BEFORE its final run_bounded import; do not invent
        # that historical import. Its sibling bytes are checked independently.
        if name != 'run_bounded.py' or str(probe.with_name(name)) in identity['imports']:
            d.require(identity['imports'].get(str(probe.with_name(name))) == digest,
                      'unreviewed probe/handshake implementation')
    # Explicit relative-root relocation only. Original imports still rehashed.
    audited = {relative: d.identity(ROOT / relative)['sha256'] for relative in DECODER_ONLY}
    digest = hashlib.sha256((json.dumps(audited, sort_keys=True, separators=(',', ':')) + '\n').encode()).hexdigest()
    d.require(spec.get('decoder_code_sha256') == digest, 'operator decoder code pin mismatch')
    return old_root, audited


def require_cuda_device(value):
    import torch
    device = torch.device(value)
    d.require(device.type == 'cuda' and torch.cuda.is_available(), 'operational CUDA device required')
    index = torch.cuda.current_device() if device.index is None else device.index
    d.require(0 <= index < torch.cuda.device_count(), 'CUDA device index unavailable')
    torch.empty(0, device=device)  # Exercise allocation, not just a device string.
    return str(device)


def tree_identity(root):
    # Same canonical tree contract as frozen real_probes.py at 3b62306.
    import json
    root = Path(root).absolute()
    rows = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            rows[str(path.relative_to(root))] = dict(resolved=str(path.resolve(strict=True)), sha256=d.identity(path)['sha256'])
        elif path.is_symlink():
            raise ValueError('broken or directory asset symlink')
    d.require(rows, 'empty asset root')
    return hashlib.sha256((json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()).hexdigest()


def verify_loaded_modules(identity, remote_hashes, *, project_root=None, decoder_code=None, module_cache=None):
    """Actual Python/extension closure, not only the receipt's declared imports."""
    records = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        d.require(Path(file).is_file(), 'loaded module file missing: ' + name)
        actual = d.identity(file)
        path = Path(actual['path'])
        if name == 'transformers_modules' or name.startswith('transformers_modules.'):
            # No module-cache relocation: exact genuine probe path+bytes required.
            d.require(module_cache is not None and path.is_relative_to(module_cache) and
                      identity['imports'].get(actual['path']) == actual['sha256'],
                      'remote cache relocation/unbound import unsupported')
            scaffold = (path.name == '__init__.py' and path.read_bytes() == b'' and
                        path.relative_to(module_cache).parts == (*name.split('.'), '__init__.py'))
            d.require(scaffold or actual['sha256'] in remote_hashes,
                      'loaded remote-code asset mismatch')
        elif path.is_relative_to(ROOT):
            relative = str(path.relative_to(ROOT))
            expected = identity['imports'].get(str(project_root / relative)) if project_root else None
            # Probe-consumed code always wins, even within decoder boundary.
            if expected is None and decoder_code is not None:
                expected = decoder_code.get(relative)
            d.require(expected == actual['sha256'], 'project implementation differs from probe/audit: ' + relative)
        else:
            d.require(identity['imports'].get(actual['path']) == actual['sha256'],
                      'loaded module outside preflight closure: ' + name)
        records.append(actual)
    return records


def verify_runtime(spec, tokenizer_spec, *, model=None, tokenizer=None):
    """Re-run before loading, after each model load and immediately before publish.

    spec: {receipt:{path,sha256}, sif:{path,sha256}, decoder_code_sha256}.
    Receipt is A's completed bounded wrapper, not measurement.json. Its expected digest is supplied
    independently by the operator, never inferred from the receipt itself.
    """
    import torch
    import transformers
    import accelerate
    from huggingface_hub.constants import HF_HUB_CACHE
    receipt, records = completed_measurement(spec)
    d.require(receipt.get('mode') == 'generation_cache_parity' and receipt.get('device') == 'cuda'
              and receipt.get('real_runtime') is True, 'genuine generation preflight required')
    d.require(receipt.get('cached_uncached_equal') is True and receipt.get('evidence_off_equal') is True
              and receipt.get('nonzero_projection') is True and receipt.get('reorder_indices_sha256'),
              'generation preflight evidence missing')
    actual = dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                  transformers=transformers.__version__, accelerate=accelerate.__version__)
    d.require(actual == VERSIONS and receipt['runtime']['versions'] == VERSIONS, 'runtime version pin mismatch')
    d.require(torch.cuda.is_available(), 'actual CUDA runtime required')
    d.require(os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline runtime required')
    d.require(spec['sif']['sha256'] == SIF, 'SIF deployment pin mismatch')
    sif = d.checked_file(spec['sif'])
    container = os.environ.get('APPTAINER_CONTAINER') or os.environ.get('SINGULARITY_CONTAINER')
    d.require(container and Path(container).resolve() == Path(sif['path']), 'active container path mismatch')
    identity = receipt['probe_identity']
    d.require(identity['schema_version'] == 1 and identity['imports'] and identity['asset_roots'], 'runtime closure missing')
    project_root, decoder_code = project_closure(identity, spec)
    records.append(sif)
    records.extend(d.checked_file(dict(path=str(Path(identity['probe_path']).with_name(name)), sha256=digest))
                   for name, digest in PROBE_CODE.items())
    for path, digest in identity['imports'].items():
        records.append(d.checked_file(dict(path=path, sha256=digest)))
    roots = identity['asset_roots']
    for root, digest in roots.items():
        d.require(Path(root).is_absolute() and tree_identity(root) == digest, 'offline asset tree mismatch')
    cache = Path(HF_HUB_CACHE).absolute()
    d.require(str(cache) in roots and str((Path.cwd()/'HFmodels').absolute()) in roots, 'actual model/tokenizer caches unbound')
    repo = cache / 'models--LongSafari--hyenadna-large-1m-seqlen-hf'
    d.require((repo/'refs/main').read_text().strip() == COMMIT, 'offline Hyena revision mismatch')
    snapshot = repo/'snapshots'/COMMIT
    d.require((snapshot/'config.json').is_file(), 'immutable model config missing')
    tokenroot = Path(tokenizer_spec['path']).resolve(strict=True)
    tokenrepo = Path.cwd()/'HFmodels/models--LongSafari--hyenadna-large-1m-seqlen-hf'
    d.require((tokenrepo/'refs/main').read_text().strip() == COMMIT, 'offline tokenizer revision mismatch')
    d.require(tokenroot in {snapshot.resolve(), (tokenrepo/'snapshots'/COMMIT).resolve()}, 'tokenizer must be the pinned Hyena snapshot')
    remote_hashes = {d.identity(p)['sha256'] for root in (snapshot, tokenroot)
                     for p in root.rglob('*.py') if p.is_file()}
    d.require(remote_hashes, 'pinned remote Python closure missing')
    from transformers.utils.hub import HF_MODULES_CACHE
    module_cache = Path(HF_MODULES_CACHE).resolve(strict=True)
    # Generated package root must itself have been consumed at this exact cache.
    scaffold = module_cache / 'transformers_modules/__init__.py'
    d.require(str(scaffold) in identity['imports'], 'module-cache relocation unsupported')
    # Load the concrete project factories before construction, so a relocated
    # current model/config/tokenizer cannot evade the pre-load comparison.
    import importlib
    for name in ('modeling_HyenaTransgenic', 'configuration_transgenic',
                 'tokenization_transgenic', 'encoder_init'):
        importlib.import_module('transgenic.model.' + name)
    records.extend(verify_loaded_modules(identity, remote_hashes, project_root=project_root,
                                        decoder_code=decoder_code, module_cache=module_cache))
    if model is not None:
        from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
        from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
        d.require(type(model) is transgenicForConditionalGeneration and type(model.config) is HyenaTransgenicConfig,
                  'actual project model/config classes required')
        hyena = model.get_encoder().hyena
        d.require(model.config.encoder_model == 'LongSafari/hyenadna-large-1m-seqlen-hf', 'unbound encoder model')
        d.require(getattr(hyena.config, '_commit_hash', None) == COMMIT, 'loaded model config revision mismatch')
        raw_config = d.read_json(snapshot/'config.json')
        overrides = dict(max_seq_len=model.config.max_encoder_seqlen,
                         d_model=model.config.encoder_d_model, n_layer=model.config.encoder_n_layer)
        for key, value in dict(raw_config, **overrides).items():
            if key not in {'auto_map', 'architectures', 'torch_dtype', 'transformers_version', '_name_or_path', '_commit_hash'}:
                d.require(getattr(hyena.config, key, None) == value, 'loaded config differs from pinned asset: ' + key)
        for obj in (hyena, hyena.config, tokenizer):
            d.require(obj is not None and type(obj).__module__.startswith('transformers_modules.'), 'actual remote class required')
            source = sys.modules[type(obj).__module__].__file__
            d.require(d.identity(source)['sha256'] in remote_hashes, 'loaded class outside immutable asset closure')
    return records
