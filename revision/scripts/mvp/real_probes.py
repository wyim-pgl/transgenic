#!/usr/bin/env python3
"""CUDA-only real Hyena readiness measurements; never a trainer or CPU fallback.

resource_budget is deliberately NOT implemented: see MVP_REAL_PROBES_REPORT.md.
Run only through run_bounded in the pinned offline SIF/allocation. The five
implemented modes emit nothing until every check and identity recheck succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

from make_launch_readiness import (_atomic_write, _canonical_bytes, _sha256,
    json_snapshot, optional_sidecars, require_allocation, tooling_identity)

TIERS = (30720, 61440, 129024)
MODES = ('real_tokenizer', 'real_hyena_step0_three_tiers',
         'generation_cache_parity', 'bf16_compile', 'four_rank_smoke')
PROJECTION = 'transgenic.encoder.evidence_proj.weight'


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA-only real probe: CPU execution cannot certify readiness')
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError('native bf16 is required; emulation is forbidden')


def runtime_contract(bindings):
    """P1 imports and the actual installed Accelerate parser, without launching."""
    import torch
    import transformers
    import accelerate
    from accelerate.commands import launch
    versions = {'torch': str(torch.__version__), 'cuda': torch.version.cuda,
                'transformers': transformers.__version__, 'accelerate': accelerate.__version__}
    expected = {'torch': '2.14.0a0+4fdf77b940.nv26.08', 'cuda': '13.4',
                'transformers': '4.57.6', 'accelerate': '1.15.0'}
    if versions != expected:
        raise ValueError(f'actual SIF import versions disagree with deployment pin: {versions}')
    bound = bindings['accelerate_config']
    if _sha256(Path(bound['path'])) != bound['sha256']:
        raise ValueError('Accelerate config identity mismatch')
    argv = ['--multi_gpu', '--num_machines', '1', '--num_processes', '4',
            '--mixed_precision', 'bf16', '--config_file', bound['path'], '--max_restarts', '0',
            str(Path(__file__).resolve()), '--mode', 'four_rank_smoke']
    parsed, _, _ = launch._validate_launch_command(launch.launch_command_parser().parse_args(argv))
    observed = {key: getattr(parsed, key) for key in ('gpu_ids', 'dynamo_backend',
        'num_processes', 'num_machines', 'machine_rank', 'mixed_precision',
        'main_process_ip', 'main_process_port', 'rdzv_backend', 'max_restarts')}
    expected_config = dict(gpu_ids='0,1,2,3', dynamo_backend='NO', num_processes=4,
        num_machines=1, machine_rank=0, mixed_precision='bf16', main_process_ip='127.0.0.1',
        main_process_port=29500, rdzv_backend='static', max_restarts=0)
    if observed != expected_config:
        raise ValueError(f'actual Accelerate parser changed pinned launch semantics: {observed}')
    return {'versions': versions, 'accelerate_parsed': observed}


def tree_identity(root):
    """Hash cache bytes AND relative link destinations; include additions/removals."""
    root = Path(root).absolute()
    rows = {}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            rows[str(p.relative_to(root))] = {
                'resolved': str(p.resolve(strict=True)), 'sha256': _sha256(p)}
        elif p.is_symlink():
            raise ValueError(f'broken or directory symlink in asset root: {p}')
    if not rows:
        raise ValueError(f'empty/missing asset root: {root}')
    return hashlib.sha256(_canonical_bytes(rows)).hexdigest()


def validate_identity(value):
    """Recheck actual executable/import bytes and offline assets, not just argv."""
    identity = value.get('probe_identity', {})
    if (identity.get('schema_version') != 1 or not identity.get('imports') or
            not identity.get('asset_roots') or identity.get('probe_path') != str(Path(__file__).resolve())):
        raise ValueError('missing or malformed real probe identity')
    for path, expected in identity['imports'].items():
        if _sha256(Path(path)) != expected:
            raise ValueError(f'probe import identity changed: {path}')
    if identity['imports'].get(str(Path(__file__).resolve())) != _sha256(Path(__file__)):
        raise ValueError('probe identity does not bind executable bytes')
    for path, expected in identity['asset_roots'].items():
        if tree_identity(path) != expected:
            raise ValueError(f'offline asset identity changed: {path}')


def tensor_digest(tensor):
    import torch
    t = tensor.detach().cpu().contiguous()
    if t.is_floating_point() and not torch.isfinite(t).all():
        raise FloatingPointError('nonfinite tensor in real probe')
    h = hashlib.sha256(_canonical_bytes({'shape': list(t.shape), 'dtype': str(t.dtype)}))
    h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def equal_tensors(a, b):
    if tensor_digest(a) != tensor_digest(b):
        raise ValueError('required bitwise equality failed (no tolerance policy)')


def check_tracks(sequence, tracks):
    import torch
    if tracks.shape != (len(sequence), 8) or tracks.dtype != torch.float16:
        raise ValueError('EST ordering/alignment requires float16 [L,8]')
    mask = torch.tensor([base in 'Nn' for base in sequence])
    if torch.count_nonzero(tracks[mask]):
        raise ValueError('EST N mask must zero all eight channels')
    if not torch.equal(tracks[~mask, 7], torch.ones_like(tracks[~mask, 7])):
        raise ValueError('EST channel 7 must be source availability')
    if not torch.isfinite(tracks).all() or (tracks < 0).any():
        raise ValueError('invalid EST tracks')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=MODES, required=True)
    for flag in ('bindings', 'bindings-file-sha256', 'assets-manifest', 'assets-sha256',
                 'db', 'evidence-db', 'config', 'tile-rns', 'tile-rns-sha256', 'scratch-root'):
        p.add_argument('--' + flag, required=True)
    return p


def setup(args):
    require_cuda()  # Before DB/cache hashing or construction; no CPU fixtures.
    require_allocation()
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('runner must set CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup')
    forbidden = ('TRANSGENIC_RUN_DIR', 'TRANSGENIC_ALLOW_ENCODER_DRIFT', 'TRANSGENIC_NO_GRAD_CKPT',
                 'TRANSGENIC_CKPT_REENTRANT', 'TRANSGENIC_NO_COMPILE', 'TRANSGENIC_COMPILE_SCOPE')
    if any(os.environ.get(key) for key in forbidden):
        raise ValueError('probe refuses model/compile/checkpoint override environment')
    # Must precede all HF imports. Factories retain the exact training cache paths.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    record = json_snapshot(args.bindings, args.bindings_file_sha256)
    bindings = record['bindings']
    binding_sha = hashlib.sha256(_canonical_bytes(bindings)).hexdigest()
    if record['bindings_sha256'] != binding_sha or os.environ.get('MVP_BINDINGS_SHA256') != binding_sha:
        raise ValueError('prepared bindings/runner identity mismatch')
    if bindings['tooling'] != tooling_identity():
        raise ValueError('probe tooling changed since prepare')
    runtime = runtime_contract(bindings)  # Fail before expensive asset/DB hashing.
    from dataclasses import asdict
    from transgenic.training.mvp_runtime import MVPRecipe
    if bindings['recipe'] != asdict(MVPRecipe()) or bindings['seed'] not in (1, 2, 3):
        raise ValueError('only the approved 250/12 recipe and seeds 1/2/3 are supported')
    for path, key in ((args.db, 'corpus_sha256'), (args.evidence_db, 'evidence_sha256'),
                      (args.config, 'config_sha256')):
        if _sha256(Path(path)) != bindings[key]:
            raise ValueError(f'input binding mismatch: {key}')
    from transgenic.training.mvp_cli_runtime import _manifest_files
    manifest = bindings['consumed_code_manifest']
    if _sha256(Path(manifest['path'])) != manifest['sha256']:
        raise ValueError('consumed code manifest mismatch')
    _manifest_files(Path(manifest['path']))
    assets = json_snapshot(args.assets_manifest, args.assets_sha256)
    if not isinstance(assets, dict) or set(assets) != {'roots'} or not assets['roots']:
        raise ValueError('asset manifest requires nonempty roots mapping')
    for root, expected in assets['roots'].items():
        if not Path(root).is_absolute() or tree_identity(root) != expected:
            raise ValueError(f'asset manifest mismatch: {root}')
    # Both cache locations used by the unchanged runtime must be explicitly pinned.
    from huggingface_hub.constants import HF_HUB_CACHE
    from transformers.utils.hub import HF_MODULES_CACHE
    required = (Path.cwd() / 'HFmodels', Path(HF_HUB_CACHE))
    if any(str(p.absolute()) not in assets['roots'] for p in required):
        raise ValueError('pin both ./HFmodels and HF_HUB_CACHE roots explicitly')
    for root in required:
        ref = root / 'models--LongSafari--hyenadna-large-1m-seqlen-hf/refs/main'
        if ref.read_text().strip() != '0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446':
            raise ValueError('encoder/tokenizer offline HF revision differs from deployment pin')
    # Dynamic module/compiler caches are writable in an independent namespace,
    # not immutable input roots. Their consumed import bytes are recorded later.
    scratch = Path(args.scratch_root).resolve(strict=True)
    for key, value in (('HF_MODULES_CACHE', HF_MODULES_CACHE),
                       ('TORCHINDUCTOR_CACHE_DIR', os.environ.get('TORCHINDUCTOR_CACHE_DIR', '')),
                       ('TRITON_CACHE_DIR', os.environ.get('TRITON_CACHE_DIR', ''))):
        p = Path(value).resolve()
        if not value or not p.is_relative_to(scratch) or any(p.is_relative_to(Path(r).resolve()) for r in assets['roots']):
            raise ValueError(f'{key} must be in independent writable scratch, not B5/input caches')
    import numpy as np
    import torch
    seed = bindings['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return bindings, assets['roots'], runtime


def load_model(args, bindings):
    import torch
    from transgenic.training.b5_runtime import load_b5_config, model_kwargs
    from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
    from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
    from transgenic.training.weights_init import load_weights_only
    from transgenic.training.mvp_cli_runtime import _validate_initialization_readiness
    cfg = load_b5_config(args.config)
    if (cfg['encoder_model'] != 'LongSafari/hyenadna-large-1m-seqlen-hf' or
            (cfg['d_model_encoder'], cfg['encoder_n_layer'], cfg['max_encoder_seqlen']) != (768, 12, 129024)):
        raise ValueError('real probes require the approved 400M window-v3 encoder')
    config = HyenaTransgenicConfig(**model_kwargs(cfg), evidence_channels=8)
    if not config.load_pretrained_encoder:
        raise ValueError('real pretrained Hyena is mandatory')
    model = transgenicForConditionalGeneration(config)
    source = bindings['source_checkpoint']
    weights = Path(source['weights_path'])
    if optional_sidecars(weights.parent) != source['optional_sidecars']:
        raise ValueError('optional initialization sidecars changed')
    audit = load_weights_only(model, weights.parent)
    _validate_initialization_readiness(audit, {'bindings': bindings}, weights,
                                       Path(source['run_config_path']))
    if torch.count_nonzero(model.get_parameter(PROJECTION)):
        raise ValueError('step0 projection is not zero')
    return model, cfg, audit


def datasets(args, bindings, cfg, model_config):
    from transgenic.datasets.datasets import isoformDataHyena
    common = dict(mode='train', split='train', strict=True, species='Athaliana',
                  evidence_db=args.evidence_db, evidence_k=8,
                  aug_seed=20260914 + bindings['seed'], return_tile_ids=True,
                  gff_vocab_version=cfg['gff_vocab_version'])
    ds = isoformDataHyena(args.db, eval_fraction=1, **common)
    from transgenic.training.mvp_cli_runtime import validate_decoder_dataset, _close_dataset
    try:
        validate_decoder_dataset(ds, cfg, model_config)
        return ds
    except Exception:
        _close_dataset(ds)
        raise


def measured_batches(args, ds):
    """Require real forward/RC evidence tiles at each full, uncropped tier."""
    import torch
    from transgenic.datasets.datasets import hyena_collate_fn
    chosen = json_snapshot(args.tile_rns, args.tile_rns_sha256)
    if set(chosen) != {str(t) for t in TIERS}:
        raise ValueError('tile-rns must supply every tier')
    batches, any_n = {}, False
    lookup = {rn: i for i, rn in enumerate(ds._index_map)}
    for tier in TIERS:
        rns = chosen[str(tier)]
        if (not isinstance(rns, list) or len(rns) != 2 or
                any(type(rn) is not int or rn not in lookup for rn in rns) or rns[0] == rns[1]):
            raise ValueError('each tier needs two distinct accepted train RNs, forward then RC')
        samples, sequences, rows = [], [], []
        for rn in rns:
            sample = ds[lookup[rn]]
            row = ds._get_connection().execute(
                'SELECT sequence,is_rc,chromosome,start,fin FROM geneList WHERE rn=?', [rn]).fetchone()
            seq, rc, chrom, start, end = row
            if len(seq) != tier or sample[0].shape != (1, tier) or not sample[9]['has_evidence']:
                raise ValueError('selected tile is wrong tier or has no real EST evidence')
            check_tracks(seq, sample[8])
            ids = ds.encoder_tokenizer(seq, return_tensors='pt')['input_ids'][:, :-1]
            equal_tensors(ids, sample[0])
            if not torch.count_nonzero(sample[8][:, :7]):
                raise ValueError('tile has no nonzero EST signal after masking')
            any_n |= 'N' in seq.upper()
            samples.append(sample)
            sequences.append(seq)
            rows.append((rc, chrom, start, end))
        if rows[0][0] is not False or rows[1][0] is not True or rows[0][1:] != rows[1][1:]:
            raise ValueError('tier pair must be forward/RC twins of the same interval')
        complement = str.maketrans('ACGTNacgtn', 'TGCANtgcan')
        if sequences[0].translate(complement)[::-1] != sequences[1]:
            raise ValueError('DNA reverse complement mismatch')
        equal_tensors(samples[0][8].flip(0)[:, [1, 0, 2, 5, 6, 3, 4, 7]], samples[1][8])
        batches[tier] = [hyena_collate_fn([sample]) for sample in samples]
    if not any_n:
        raise ValueError('selected actual tiles must exercise N masking')
    return batches


def inputs(batch, device):
    from transgenic.training.mvp_runtime import _batch_parts, MVPRecipe
    ii, am, labels, tracks = (x.to(device) for x in _batch_parts(batch, MVPRecipe()))
    return dict(input_ids=ii, attention_mask=am, labels=labels, evidence_tracks=tracks,
                use_cache=False, return_dict=True)


def projection_gradients(model, batch, device):
    """Real CE autograd: nonzero finite evidence gradient; zeros contribute nothing."""
    import torch
    model.eval()
    kw = inputs(batch, device)
    projection = model.get_parameter(PROJECTION)
    result = {}
    for on in (False, True):
        model.zero_grad(set_to_none=True)
        tr = kw['evidence_tracks'] if on else torch.zeros_like(kw['evidence_tracks'])
        output = model(**dict(kw, evidence_tracks=tr))
        if not torch.isfinite(output.loss):
            raise FloatingPointError('nonfinite projection probe loss')
        output.loss.backward()
        grad = projection.grad
        if grad is None or not torch.isfinite(grad).all():
            raise FloatingPointError('missing/nonfinite evidence projection gradient')
        norm = float(grad.float().norm().item())
        if (on and norm <= 0) or (not on and torch.count_nonzero(grad)):
            raise ValueError('EST on must reach projection; all-zero off must contribute zero')
        result['on' if on else 'off'] = {'gradient_norm': norm, 'gradient_sha256': tensor_digest(grad)}
    model.zero_grad(set_to_none=True)
    return result


def step0(model, cfg, bindings, batches, device):
    import torch
    from safetensors.torch import load_model as load_safetensors
    from transgenic.training.b5_runtime import model_kwargs
    from transgenic.model.configuration_transgenic import HyenaTransgenicConfig
    from transgenic.model.modeling_HyenaTransgenic import transgenicForConditionalGeneration
    # An independent channels=0 architecture loaded strictly from the same verified B5.
    b5 = transgenicForConditionalGeneration(HyenaTransgenicConfig(**model_kwargs(cfg), evidence_channels=0))
    load_safetensors(b5, bindings['source_checkpoint']['weights_path'], strict=True)
    b5.to(device).eval()
    model.to(device).eval()
    tiers = {}
    with torch.no_grad():
        for tier, pair in batches.items():
            hashes = {key: [] for key in ('b5', 'off', 'zero', 'repeat')}
            batch_hashes = []
            for batch in pair:
                kw = inputs(batch, device)
                batch_hashes.append([tensor_digest(kw[k]) for k in ('input_ids', 'attention_mask', 'labels', 'evidence_tracks')])
                base = b5(**dict(kw, evidence_tracks=None)).logits
                off = model(**dict(kw, evidence_tracks=torch.zeros_like(kw['evidence_tracks']))).logits
                zero = model(**kw).logits
                repeat = model(**kw).logits
                for name, value in (('b5', base), ('off', off), ('zero', zero), ('repeat', repeat)):
                    equal_tensors(base, value)
                    hashes[name].append(tensor_digest(value))
            row = {'input_sha256': hashlib.sha256(_canonical_bytes({'batches': batch_hashes})).hexdigest(), 'finite': True}
            for name, field in (('b5', 'b5_logits_sha256'), ('off', 'mvp_off_logits_sha256'),
                                ('zero', 'mvp_zero_projection_logits_sha256'), ('repeat', 'repeat_logits_sha256')):
                row[field] = hashes[name][0]
            row['rc_logits_sha256'] = {name: values[1] for name, values in hashes.items()}
            tiers[str(tier)] = row
    del b5
    torch.cuda.empty_cache()
    gradients = projection_gradients(model, batches[30720][0], device)
    return dict(tiers=tiers, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                eval_mode=not model.training, comparison='bitwise',
                cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'], est_gradients=gradients,
                rc_n_alignment=True)


def generation(model, batch, device):
    """Nonzero EST, beam2 expansion/reorder and raw-logit same-path comparison.

    Cached/full-prefix arithmetic is NOT silently assigned a tolerance: tokens
    must match; raw logits must match bitwise against direct-encoder generation
    with the SAME cache/beam setting. Cross-path logit equality is not claimed.
    This process owns a dedicated probe model; its projection is restored even
    when a hook or a comparison fails.
    """
    import torch
    from transgenic.model.modeling_HyenaTransgenic import HyenaModelOutput
    model.to(device).eval()
    kw = inputs(batch, device)
    projection = model.get_parameter(PROJECTION)
    original = projection.detach().clone()
    encoder = model.get_encoder()
    records, activations = {}, {}
    old_reorder = model._reorder_cache
    previous_override = model.__dict__.get('_reorder_cache')
    reorder_calls = []
    handles = []

    def observed_reorder(past, beam_idx):
        reference = tuple(tuple(t.index_select(0, beam_idx.to(t.device)) for t in layer[:2])
                          + tuple(layer[2:]) for layer in past)
        actual = old_reorder(past, beam_idx)
        for got_layer, expected_layer in zip(actual, reference, strict=True):
            for got, expected in zip(got_layer, expected_layer, strict=True):
                equal_tensors(got, expected)
        reorder_calls.append(tensor_digest(beam_idx))
        return actual

    try:
        with torch.no_grad():
            # Deterministic nonzero diagonal signal in all eight channels, not
            # learned weights, and never used by any training/evaluation output.
            projection.zero_()
            projection[:8, :8].copy_(torch.eye(8, device=device, dtype=projection.dtype))
            model._reorder_cache = observed_reorder
            for on in (False, True):
                tr = kw['evidence_tracks'] if on else torch.zeros_like(kw['evidence_tracks'])
                raw = encoder(kw['input_ids'], evidence_tracks=tr).last_hidden_state
                activations['on' if on else 'off'] = tensor_digest(raw)
                if not on:
                    equal_tensors(raw, encoder(kw['input_ids']).last_hidden_state)
                for beams in (1, 2):
                    outputs = {}
                    for cached in (False, True):
                        received, expanded = [], []

                        def capture_encoder(module, positional, keywords, output):
                            evidence = keywords.get('evidence_tracks')
                            if evidence is None:
                                raise ValueError('generation dropped EST tracks at encoder hook')
                            equal_tensors(evidence, tr)
                            equal_tensors(output.last_hidden_state, raw)
                            received.append(True)

                        def capture_forward(module, positional, keywords):
                            hidden = keywords.get('encoder_outputs')
                            if hidden is None:
                                raise ValueError('generation did not forward encoder outputs')
                            equal_tensors(hidden, raw.repeat_interleave(beams, dim=0))
                            expanded.append(int(hidden.shape[0]))

                        handles = [encoder.register_forward_hook(capture_encoder, with_kwargs=True),
                                   model.register_forward_pre_hook(capture_forward, with_kwargs=True)]
                        common = dict(input_ids=kw['input_ids'], attention_mask=kw['attention_mask'],
                            do_sample=False, num_beams=beams, min_new_tokens=16, max_new_tokens=16,
                            use_cache=cached, return_dict_in_generate=True, output_logits=True)
                        before_reorder = len(reorder_calls)
                        actual = model.generate(**common, evidence_tracks=tr)
                        for handle in handles:
                            handle.remove()
                        handles = []
                        if len(received) != 1 or len(expanded) != 16 or set(expanded) != {beams}:
                            raise ValueError('generation encoder-once/beam expansion contract failed')
                        if beams == 2 and cached and len(reorder_calls) <= before_reorder:
                            raise ValueError('beam2 did not exercise observable cache reordering')
                        # Bypass generation's encoder/evidence forwarding, but keep
                        # exact eager decoder/cache/beam numerical path unchanged.
                        reference = model.generate(**common,
                            encoder_outputs=HyenaModelOutput(last_hidden_state=raw.clone()))
                        equal_tensors(actual.sequences, reference.sequences)
                        if len(actual.logits) != 16 or len(reference.logits) != 16:
                            raise ValueError('missing raw generation logits')
                        for got, expected in zip(actual.logits, reference.logits, strict=True):
                            equal_tensors(got, expected)
                        label = f'{"on" if on else "off"}/beam{beams}/cache{int(cached)}'
                        records[label] = {'sequences_sha256': tensor_digest(actual.sequences),
                            'logits_sha256': [tensor_digest(x) for x in actual.logits],
                            'encoder_calls': len(received), 'expanded_batches': expanded}
                        outputs[cached] = actual.sequences
                    equal_tensors(outputs[False], outputs[True])
            if activations['on'] == activations['off']:
                raise ValueError('nonzero EST projection did not change encoder activations')
    finally:
        for handle in handles:
            handle.remove()
        if previous_override is None:
            model.__dict__.pop('_reorder_cache', None)
        else:
            model._reorder_cache = previous_override
        with torch.no_grad():
            projection.copy_(original)
        equal_tensors(projection, original)
    return dict(cached_uncached_equal=True, evidence_off_equal=True,
                generated=records, encoder_activation_sha256=activations,
                reorder_indices_sha256=reorder_calls, new_tokens=16, nonzero_projection=True,
                projection_restored_sha256=tensor_digest(projection),
                comparison='same-cache-beam-path raw logits bitwise; cache/full-prefix tokens bitwise')


def one_update(accelerator, model, optimizer, batches):
    """Probe-local one update; fixed recipe, same training primitives, no trainer override."""
    import torch
    from transgenic.training.mvp_runtime import (MVPRecipe, _collective_true,
        _set_lrs, _step_was_skipped)
    recipe = MVPRecipe()
    model.train()
    optimizer.zero_grad()
    _set_lrs(optimizer, recipe, 0)
    start = time.monotonic()
    torch.cuda.reset_peak_memory_stats(accelerator.device)
    syncs = 0
    for batch in batches:
        with accelerator.accumulate(model):
            from transgenic.training.mvp_runtime import _batch_parts
            try:
                _batch_parts(batch, recipe)
                valid = True
            except (ValueError, TypeError):
                valid = False
            if not _collective_true(accelerator, valid):
                raise ValueError('invalid batch contract on at least one rank')
            kw = inputs(batch, accelerator.device)
            # Same forward signature as run_fixed_updates (no alternate loss).
            kw.pop('use_cache')
            kw.pop('return_dict')
            output = model(**kw)
            if not _collective_true(accelerator, bool(torch.isfinite(output.loss).item())):
                raise FloatingPointError('nonfinite loss on a rank')
            accelerator.backward(output.loss)
            if accelerator.sync_gradients:
                finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                if not _collective_true(accelerator, finite):
                    raise FloatingPointError('nonfinite gradient on a rank')
                accelerator.clip_grad_norm_(model.parameters(), recipe.clip_grad_norm)
                optimizer.step()
                if not _collective_true(accelerator, not _step_was_skipped(optimizer)):
                    raise RuntimeError('optimizer update skipped')
                optimizer.zero_grad()
                syncs += 1
    if not _collective_true(accelerator, syncs == 1 and len(batches) == recipe.accumulation_steps):
        raise RuntimeError('probe must observe exactly one update after 24 microbatches')
    torch.cuda.synchronize(accelerator.device)
    finite = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    if not _collective_true(accelerator, finite):
        raise FloatingPointError('nonfinite parameters after update')
    return dict(rank=accelerator.process_index, wall_seconds=time.monotonic() - start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(accelerator.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(accelerator.device),
                microbatches=24, optimizer_steps=syncs)


def collective_batches(accelerator, loader):
    from transgenic.training.mvp_finalize import _collective_preflight
    import itertools
    result = []
    def retrieve():
        result.extend(itertools.islice(loader, 24))
        if len(result) != 24:
            raise ValueError('insufficient real per-rank training samples for accumulation24')
    _collective_preflight(accelerator, retrieve)
    return result


def collective_compiler_check(accelerator, before):
    import torch
    from torch._inductor import metrics
    from transgenic.training.mvp_finalize import _collective_preflight
    def check():
        if torch._dynamo.config.suppress_errors or metrics.generated_kernel_count <= before:
            raise RuntimeError('no observed Inductor kernel generation or eager fallback enabled')
    _collective_preflight(accelerator, check)


def compiled_update(args, bindings, model, ds, accelerator):
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from transgenic.datasets.datasets import hyena_collate_fn
    from transgenic.training.mvp_runtime import MVPRecipe, build_optimizer
    from transgenic.training.mvp_cli_runtime import (compile_bound_forward,
        _collect_runtime_hardware, _validate_public_hardware)
    from accelerate.utils import gather_object
    from transgenic.training.mvp_finalize import _collective_preflight
    state = {}
    hardware = _validate_public_hardware(_collect_runtime_hardware(accelerator))
    def configure():
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        state['optimizer'] = build_optimizer(model, MVPRecipe())
        compile_bound_forward(model)
    _collective_preflight(accelerator, configure)
    from torch._inductor import metrics
    before = metrics.generated_kernel_count
    # prepare contains collectives; never nest it inside a local-status phase.
    model, optimizer = accelerator.prepare(model, state['optimizer'])
    def prepare_loader():
        # Restore the real training augmentation (paired probes use fraction=1).
        ds.eval_fraction = None
        seed = bindings['seed']
        aug_seed = 20260914 + seed
        ds._aug_generator = torch.Generator().manual_seed(aug_seed + accelerator.process_index)
        ds._aug_key = (os.getpid(), None)
        loader = DataLoader(ds, batch_size=1, num_workers=0, collate_fn=hyena_collate_fn,
            sampler=DistributedSampler(ds, num_replicas=4, rank=accelerator.process_index,
                                       shuffle=True, seed=seed),
            generator=torch.Generator().manual_seed(aug_seed + 1_000_000 + accelerator.process_index),
            pin_memory=True)
        state['loader'] = loader
    _collective_preflight(accelerator, prepare_loader)
    # No short-tile replacement, no repeating the same toy batch.
    batches = collective_batches(accelerator, state['loader'])
    measured = one_update(accelerator, model, optimizer, batches)
    collective_compiler_check(accelerator, before)
    rank_rows = gather_object([measured])
    if len(rank_rows) != 4:
        raise RuntimeError('missing rank measurement')
    return dict(hardware=hardware, optimizer_steps=1, finite=True, native_bf16=True,
                inductor=True, no_eager_fallback=True, ranks=rank_rows,
                accumulation_steps=24, timing_scope='cold_compile_plus_one_update_not_resource_budget')


def main(argv=None):
    args = parser().parse_args(argv)
    # Accelerator may import HF modules: offline policy must precede it too.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    started = time.monotonic()
    require_cuda()
    import torch
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    accelerator = None
    if args.mode in ('bf16_compile', 'four_rank_smoke'):
        from transgenic.training.mvp_runtime import make_accelerator, MVPRecipe
        accelerator = make_accelerator(MVPRecipe(), mixed_precision='bf16')
    state = {}
    ds = None
    try:
        def prepare_inputs():
            state['inputs'] = setup(args)
        if accelerator is None:
            prepare_inputs()
        else:
            from transgenic.training.mvp_finalize import _collective_preflight
            _collective_preflight(accelerator, prepare_inputs)
        bindings, roots, runtime = state['inputs']
        def initialize():
            nonlocal ds
            state['model'], state['cfg'], state['audit'] = load_model(args, bindings)
            ds = datasets(args, bindings, state['cfg'], state['model'].config)
            state['batches'] = measured_batches(args, ds)
        if accelerator is None:
            initialize()
        else:
            from transgenic.training.mvp_finalize import _collective_preflight
            _collective_preflight(accelerator, initialize)
        model, cfg, audit = state['model'], state['cfg'], state['audit']
        batches = state['batches']
        if args.mode == 'real_tokenizer':
            seq = 'ACGTN'
            ids = ds.encoder_tokenizer(seq, return_tensors='pt')['input_ids'][0, :-1]
            if ds.encoder_tokenizer.decode(ids, skip_special_tokens=True).replace(' ', '') != seq:
                raise ValueError('real tokenizer ACGTN roundtrip failed')
            model.to(device).eval()
            with torch.no_grad():
                logits = model(**inputs(batches[30720][0], device)).logits
            result = dict(tokenizer_roundtrip=True, hyena_loaded=True,
                          logits_sha256=tensor_digest(logits), rc_n_alignment=True)
        elif args.mode == 'real_hyena_step0_three_tiers':
            result = step0(model, cfg, bindings, batches, device)
        elif args.mode == 'generation_cache_parity':
            result = generation(model, batches[30720][0], device)
        else:
            result = compiled_update(args, bindings, model, ds, accelerator)
        # Snapshot actual loaded Python and extension modules, including HF remote code.
        imports = {str(Path(__file__).resolve()): _sha256(Path(__file__))}
        for module in list(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and Path(path).is_file():
                imports[str(Path(path).resolve())] = _sha256(Path(path))
        from transgenic.training.mvp_cli_runtime import _manifest_files
        _manifest_files(Path(bindings['consumed_code_manifest']['path']))
        result.update(device='cuda', real_runtime=True, mode=args.mode, runtime=runtime,
            bindings_sha256=os.environ['MVP_BINDINGS_SHA256'],
            initialization_audit_sha256=hashlib.sha256(_canonical_bytes(audit)).hexdigest(),
            tile_rns_sha256=args.tile_rns_sha256, assets_manifest_sha256=args.assets_sha256,
            wall_seconds=time.monotonic() - started,
            probe_identity={'schema_version': 1, 'probe_path': str(Path(__file__).resolve()),
                            'imports': imports, 'asset_roots': roots})
        validate_identity(result)
        # Recheck inputs/loader sidecars after executing all kernels.
        for path, key in ((args.db, 'corpus_sha256'), (args.evidence_db, 'evidence_sha256'),
                          (args.config, 'config_sha256')):
            if _sha256(Path(path)) != bindings[key]:
                raise ValueError(f'input changed during probe: {key}')
        source = bindings['source_checkpoint']
        for prefix in ('weights', 'run_config', 'runtime_identity'):
            if _sha256(Path(source[prefix + '_path'])) != source[prefix + '_sha256']:
                raise ValueError('source changed during probe')
        if optional_sidecars(Path(source['weights_path']).parent) != source['optional_sidecars']:
            raise ValueError('source optional sidecars changed during probe')
        from run_bounded import validate_measurement
        validate_measurement(args.mode, result)
        if int(os.environ.get('RANK', '0')) == 0:
            _atomic_write(Path(os.environ['MVP_MEASUREMENT_OUTPUT']), _canonical_bytes(result))
        return 0
    finally:
        from transgenic.training.mvp_cli_runtime import _close_dataset
        try:
            _close_dataset(ds)
        finally:
            if accelerator is not None:
                accelerator.end_training()


if __name__ == '__main__':
    sys.exit(main())
