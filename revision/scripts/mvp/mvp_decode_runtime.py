"""Mandatory operator-pinned real generation preflight and live runtime closure.

A receipt is trusted operator input, not cryptographic proof of execution. CPU
fixtures cannot satisfy the real runtime checks. No unverified override exists.
"""
import hashlib
import os
from pathlib import Path
import sys

from transgenic.evidence import mvp_decode as d

COMMIT = '0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446'
SIF = '1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6'
VERSIONS = dict(torch='2.14.0a0+4fdf77b940.nv26.08', cuda='13.4', transformers='4.57.6', accelerate='1.15.0')
ROOT = Path(__file__).resolve().parents[3]


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


def verify_loaded_modules(identity, remote_hashes):
    """Actual Python/extension closure, not only the receipt's declared imports."""
    records = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, '__file__', None)
        if not file:
            continue
        d.require(Path(file).is_file(), 'loaded module file missing: ' + name)
        actual = d.identity(file)
        if name.startswith('transformers_modules.'):
            d.require(actual['sha256'] in remote_hashes and actual['sha256'] in identity['imports'].values(),
                      'loaded remote-code asset mismatch')
        elif not Path(actual['path']).is_relative_to(ROOT):
            d.require(identity['imports'].get(actual['path']) == actual['sha256'],
                      'loaded module outside preflight closure: ' + name)
        records.append(actual)
    return records


def verify_runtime(spec, tokenizer_spec, *, model=None, tokenizer=None):
    """Re-run before loading, after each model load and immediately before publish.

    spec: {receipt:{path,sha256}, sif:{path,sha256}}. Receipt uses A's
    generation_cache_parity/probe_identity schema. Its expected digest is supplied
    independently by the operator, never inferred from the receipt itself.
    """
    import torch
    import transformers
    import accelerate
    from huggingface_hub.constants import HF_HUB_CACHE
    receipt_identity = d.checked_file(spec['receipt'])
    receipt = d.read_json(receipt_identity['path'])
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
    d.require(identity['probe_path'] in identity['imports'], 'probe executable not bound')
    records = [receipt_identity, sif]
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
    records.extend(verify_loaded_modules(identity, remote_hashes))
    if model is not None:
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
