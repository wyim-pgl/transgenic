"""Reconstruct actual prepared-loader companions and prime compiler state only.

Cache directory paths are bound literally: this deliberately requires the same
operational cache configuration on all ranks/links, including shared defaults.
No node-local path normalization or cache disabling is performed.
"""
from contextlib import ExitStack, contextmanager
import copy
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path, PurePosixPath
import re

import torch
from accelerate.utils import send_to_device

from . import b6_runtime as rt
from .b6_compile_history import CompileHistory, check_inputs, decode_rng
from .b6_replay_state import capture_rng, preserve_training_state, restore_rng

HISTORY_FILE = 'compile_history.json'
LEGACY_SLOTS = list(range(1, 16)) + [3915]
LEGACY_CHECKPOINT = dict(epoch=0, step=4800, global_step=200, pending_micro=0)
LEGACY_PROVENANCE = 'metadata-derived candidate; not an observed compiler ledger'


def checkpoint_position(meta):
    return {key: meta[key] for key in ('epoch', 'step', 'global_step', 'pending_micro')}


def runtime_fingerprint(spec, original_model, accelerator, checkpointing):
    """Bind effective runtime settings, including config overrides independent of env."""
    import torch._inductor.config as inductor
    import torch._functorch.config as functorch
    dynamo = torch._dynamo.config
    from torch._functorch._aot_autograd import autograd_cache as aot_cache

    def settings(module, names):
        return {name: getattr(module, name, '<unavailable>') for name in names}

    try:
        triton = importlib.metadata.version('triton')
    except importlib.metadata.PackageNotFoundError:
        triton = None
    backend = spec.compile_backend
    if not isinstance(backend, str):
        backend = f'{backend.__module__}.{backend.__qualname__}'
    gpu = None
    if accelerator.device.type == 'cuda':
        gpu = dict(name=torch.cuda.get_device_name(accelerator.device),
                   capability=list(torch.cuda.get_device_capability(accelerator.device)))
    # PretrainedConfig.to_dict includes non-model serialization provenance (e.g.
    # transformers version); retaining it is conservative and deterministic.
    config = json.loads(original_model.config.to_json_string(use_diff=False))
    return dict(torch=str(torch.__version__), torch_git=torch.version.git_version,
                cuda=torch.version.cuda, triton=triton, gpu=gpu, backend=backend,
                world_size=accelerator.num_processes, mixed_precision=spec.mixed_precision,
                checkpointing=checkpointing, model_config=config, recipe=spec.cfg,
                deterministic=torch.are_deterministic_algorithms_enabled(),
                deterministic_warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),
                float32_matmul_precision=torch.get_float32_matmul_precision(),
                cuda_matmul=settings(torch.backends.cuda.matmul, ['allow_tf32', 'allow_fp16_reduced_precision_reduction',
                                                                  'allow_bf16_reduced_precision_reduction']),
                cudnn=settings(torch.backends.cudnn, ['allow_tf32', 'benchmark', 'deterministic', 'enabled']),
                dynamo=settings(dynamo, ['suppress_errors', 'optimize_ddp', 'automatic_dynamic_shapes',
                                         'recompile_limit', 'cache_size_limit', 'accumulated_cache_size_limit',
                                         'fail_on_recompile_limit_hit', 'fail_on_cache_limit_hit']),
                inductor=settings(inductor, ['fx_graph_cache', 'fx_graph_remote_cache', 'force_disable_caches',
                                             'autotune_local_cache', 'autotune_remote_cache',
                                             'bundled_autotune_remote_cache', 'bundle_triton_into_fx_graph_cache',
                                             'global_cache_dir', 'max_autotune', 'max_autotune_gemm',
                                             'triton.cudagraphs']),
                effective_caches=dict(fx_local=bool(inductor.fx_graph_cache and not inductor.force_disable_caches),
                                      fx_remote=bool(aot_cache.should_use_remote_fx_graph_cache() and not inductor.force_disable_caches),
                                      aot_local=bool(aot_cache.should_use_local_autograd_cache()),
                                      aot_remote=bool(aot_cache.should_use_remote_autograd_cache())),
                aot=settings(functorch, ['enable_autograd_cache', 'enable_remote_autograd_cache',
                                        'strict_autograd_cache']),
                cache_env={key: value for key, value in sorted(os.environ.items())
                           if key.startswith(('TORCHINDUCTOR_', 'TORCHDYNAMO_', 'TORCH_COMPILE_', 'TRITON_'))
                           or key in ('TORCH_HOME', 'XDG_CACHE_HOME', 'CUBLAS_WORKSPACE_CONFIG',
                                      'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE', 'NVIDIA_TF32_OVERRIDE')})


def require_rank_agreement(accelerator, value):
    """Use the journal's bounded byte transport, after coordinated local encoding."""
    collector = CompileHistory({}, {}, accelerator)
    values = collector._gather(lambda: value)
    if any(item != values[0] for item in values):
        raise RuntimeError('compiler history binding mismatch across ranks')


@contextmanager
def preserve_loader_state(loader):
    """Restore sampler state and generator objects, without copying the corpus.

    Reconstruction iterates only batch samplers (never DataLoader iterators), so
    worker queues and active loader iterators are untouched. Preserve attributes
    of the sampler chain and the loader/dataset epoch fields touched by set_epoch.
    """
    objects, seen = [], set()

    def visit(obj, sampler=False):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if hasattr(obj, '__dict__'):
            attrs = vars(obj)
            names = (set(attrs) - {'sampler', 'batch_sampler', 'data_source', 'dataset'} if sampler else
                     {'epoch', '_epoch', 'iteration', 'generator', 'synchronized_generator'} & set(attrs))
            values = {}
            for key in names:
                value = attrs[key]
                values[key] = (value, value.get_state().clone()) if isinstance(value, torch.Generator) else copy.deepcopy(value)
            objects.append((obj, names, values, sampler, set(attrs)))
        for name in ('sampler', 'batch_sampler'):
            visit(getattr(obj, name, None), True)
    visit(loader)
    visit(getattr(loader, 'dataset', None))
    rng = capture_rng()
    try:
        yield
    finally:
        for obj, names, values, sampler, original_keys in reversed(objects):
            newly_added = set(vars(obj)) - original_keys
            if not sampler:
                newly_added &= {'epoch', '_epoch', 'iteration', 'generator', 'synchronized_generator'}
            for name in newly_added:
                delattr(obj, name)
            for name in names:
                value = values[name]
                if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], torch.Generator):
                    generator, state = value
                    generator.set_state(state)
                    setattr(obj, name, generator)
                else:
                    setattr(obj, name, value)
        restore_rng(rng)


def reconstruct_batch(loader, epoch, batch_index, device):
    """Fetch only one batch's real sample keys from the prepared shard/padding policy."""
    if type(epoch) is not int or epoch < 0 or type(batch_index) is not int or batch_index < 0:
        raise ValueError('invalid replay batch position')
    with preserve_loader_state(loader):
        if hasattr(loader, 'set_epoch'):
            loader.set_epoch(epoch)
        elif hasattr(loader.sampler, 'set_epoch'):
            loader.sampler.set_epoch(epoch)
        keys = next(itertools.islice(iter(loader.batch_sampler), batch_index, batch_index + 1), None)
        if keys is None:
            raise ValueError('replay batch index outside prepared loader')
        batch = loader.collate_fn([loader.dataset[key] for key in keys])
        return send_to_device(batch, device)


def replay_inputs(batch, arm, view):
    rt.validate_b6_batch(batch)
    inputs = rt.model_inputs(rt.de_novo_view(batch) if view == 'de_novo' else rt.apply_arm(batch, arm))
    torch._dynamo.maybe_mark_dynamic(inputs['labels'], 1)
    return inputs


def forward_position(phase, epoch, batch_index, view, optimizer_step):
    return dict(phase=phase, epoch=epoch, batch_index=batch_index, view=view,
                training=phase == 'train', grad_enabled=phase == 'train', optimizer_step=optimizer_step)


def check_replay_memory(accelerator, memory_probe=None):
    device = accelerator.device
    with rt.checkpoint_io(accelerator):
        reserved = int(memory_probe()) if memory_probe else (max(torch.cuda.memory_reserved(device),
                    torch.cuda.max_memory_reserved(device)) if device.type == 'cuda' else 0)
    peaks = accelerator.gather(torch.tensor([reserved], device=device)).tolist()
    if rt.memory_abort(peaks, None, write=False):
        raise RuntimeError(f'compiler replay exceeds B6 memory cap: {peaks}')


@contextmanager
def _preserve_modes(model):
    modes = [(module, module.training) for module in model.modules()]
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training


def execute_replay(history, model, original_model, loaders, arm, accelerator, per_rank_acc, guard,
                   *, memory_probe=None, legacy_rows=None):
    """Shared priming boundary for recorded history and explicit legacy reconstruction.

    Proof harnesses may wrap this callable to observe the before/after boundary.
    No parameter, gradient, mode, RNG, optimizer or scheduler progress is retained.
    Compiled graphs and guard observations deliberately survive the context.
    """
    legacy = legacy_rows is not None
    entries = ([dict(position=forward_position('train', 0, slot - 1, 'arm', (slot - 1) // per_rank_acc))
                for slot in LEGACY_SLOTS] if legacy else history.entries)
    with ExitStack() as protected:
        # Snapshot failures are rank-local too: coordinate before any prepared forward.
        with rt.checkpoint_io(accelerator):
            protected.enter_context(preserve_training_state(original_model))
            protected.enter_context(_preserve_modes(model))
        for entry in entries:
            position = entry['position']
            with rt.checkpoint_io(accelerator):
                batch = reconstruct_batch(loaders[position['phase']], position['epoch'], position['batch_index'],
                                          accelerator.device)
                inputs = replay_inputs(batch, arm, position['view'])
                if legacy:
                    expected = legacy_rows[str(accelerator.process_index)][str(position['batch_index'] + 1)]
                    actual = [[str(row), int(epoch), sha] for row, epoch, sha in
                              zip(batch['meta']['row'], batch['meta']['epoch'], batch['meta']['uniforms_sha256'])]
                    if actual != [expected]:
                        raise ValueError('legacy reconstructed batch differs from original draw prefix')
                else:
                    check_inputs(entry, inputs, batch['meta'], accelerator.process_index)
            with rt.checkpoint_io(accelerator):
                if not legacy:
                    restore_rng(decode_rng(entry['ranks'][accelerator.process_index]['rng_before']))
                model.train(position['training'])
                rng_before = capture_rng()
                graphs_before, activity_before = rt.compiled_graph_count(), rt.inductor_activity()
            # Prepared model/backward may enter DDP collectives: never enclose in checkpoint_io.
            with torch.set_grad_enabled(position['grad_enabled']):
                out = model(**inputs)
                if position['training']:
                    accelerator.backward(out.loss / per_rank_acc)
            graphs_after, activity_after = rt.compiled_graph_count(), rt.inductor_activity()
            with rt.checkpoint_io(accelerator):
                if not legacy and graphs_after - graphs_before != entry['ranks'][accelerator.process_index]['delta']:
                    raise RuntimeError('compiler replay graph delta differs from recorded history')
                shape = tuple(inputs['input_ids'].shape)
                guard.observe(position['optimizer_step'] + (1 if position['training'] else 0),
                              shape if position['training'] else ('eval',) + shape,
                              graphs_after, bool(torch._dynamo.config.suppress_errors))
            if legacy:
                history.observe(position, inputs, batch['meta'], rng_before, graphs_before, graphs_after,
                                origin='reconstructed_legacy', force=True,
                                activity_before=activity_before, activity_after=activity_after)
            check_replay_memory(accelerator, memory_probe)


def replay_history(history, model, original_model, loaders, arm, accelerator, per_rank_acc, guard,
                   *, memory_probe=None):
    with rt.checkpoint_io(accelerator):
        payload = history.to_payload()
    require_rank_agreement(accelerator, payload)
    return execute_replay(history, model, original_model, loaders, arm, accelerator, per_rank_acc, guard,
                          memory_probe=memory_probe)


def validate_legacy_manifest(path, checkpoint_dir, run_dir, checkpoint, *, arm, world_size, batch_size):
    """Local-only heavy validation; the caller runs this exactly once on rank 0."""
    def require(value, reason):
        if not value:
            raise ValueError(f'legacy compile bootstrap: {reason}')

    require(arm == 'C2' and world_size == 4 and batch_size == 1, 'requires C2/world4/batch1')
    require(checkpoint == LEGACY_CHECKPOINT and all(type(v) is int for v in checkpoint.values()),
            'requires original checkpoint200 position')
    raw = Path(path).read_bytes()
    def unique_fields(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    manifest = json.loads(raw, object_pairs_hook=unique_fields)
    require(type(manifest) is dict and set(manifest) == {'version', 'kind', 'checkpoint', 'world_size',
            'checkpoint_files', 'draw_prefix_lines', 'draw_prefix_sha256', 'slots', 'provenance'}, 'invalid schema')
    require(type(manifest['version']) is int and manifest['version'] == 1, 'invalid version')
    require(manifest['kind'] == 'reconstructed_legacy_compile_bootstrap', 'invalid kind')
    require(manifest['checkpoint'] == LEGACY_CHECKPOINT and
            all(type(v) is int for v in manifest['checkpoint'].values()), 'wrong checkpoint')
    require(type(manifest['world_size']) is int and manifest['world_size'] == 4, 'wrong world')
    require(type(manifest['draw_prefix_lines']) is int and manifest['draw_prefix_lines'] == 4800, 'wrong prefix count')
    require(manifest['slots'] == LEGACY_SLOTS and all(type(x) is int for x in manifest['slots']), 'wrong slots')
    require(manifest['provenance'] == LEGACY_PROVENANCE, 'wrong provenance')
    files = manifest['checkpoint_files']
    require(type(files) is dict and bool(files), 'missing checkpoint file declarations')
    sha_pattern = re.compile(r'[0-9a-f]{64}')
    for name, sha in files.items():
        p = PurePosixPath(name)
        require(type(name) is str and not p.is_absolute() and '..' not in p.parts and
                str(p) == name and '\\' not in name and name != '.', 'unsafe checkpoint path')
        require(type(sha) is str and sha_pattern.fullmatch(sha), 'invalid checkpoint SHA256')
    root = Path(checkpoint_dir)
    require(not root.is_symlink(), 'symlink checkpoint root')
    actual = set()
    for p in root.rglob('*'):
        require(not p.is_symlink(), 'symlink checkpoint file/directory')
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
        else:
            require(p.is_dir(), 'non-regular checkpoint entry')
    required = {'meta.json', rt.CHECKPOINT_IDENTITY, 'accelerate_state/model.safetensors',
                'accelerate_state/optimizer.bin', 'accelerate_state/scheduler.bin'} | {
                    f'accelerate_state/random_states_{rank}.pkl' for rank in range(4)}
    require(required <= actual, 'checkpoint missing model/optimizer/scheduler/identity/RNG files')
    require(set(files) == actual, 'checkpoint file declarations differ from complete file set')
    for name, sha in files.items():
        require(rt.file_sha256(root / name) == sha, f'checkpoint bytes changed: {name}')
    prefixes = manifest['draw_prefix_sha256']
    require(type(prefixes) is dict and set(prefixes) == {'0', '1', '2', '3'}, 'wrong draw prefix ranks')
    rows = {}
    for rank in range(4):
        require(type(prefixes[str(rank)]) is str and sha_pattern.fullmatch(prefixes[str(rank)]), 'bad prefix SHA256')
        log_path = Path(run_dir) / 'draw_log' / f'rank{rank:02d}.tsv'
        require(not log_path.is_symlink(), 'symlink draw prefix')
        with log_path.open('rb') as fh:
            lines = list(itertools.islice(fh, 4800))
        require(len(lines) == 4800 and all(line.endswith(b'\n') for line in lines), 'short/incomplete draw prefix')
        require(hashlib.sha256(b''.join(lines)).hexdigest() == prefixes[str(rank)], 'draw prefix SHA256 mismatch')
        rows[str(rank)] = {}
        for slot in LEGACY_SLOTS:
            fields = lines[slot - 1].decode('utf-8').rstrip('\n').split('\t')
            require(len(fields) == 3 and fields[1] == '0' and sha_pattern.fullmatch(fields[2]), 'invalid draw row')
            rows[str(rank)][str(slot)] = [fields[0], 0, fields[2]]
    return dict(manifest=manifest, sha256=hashlib.sha256(raw).hexdigest()), rows


def bootstrap_legacy(history, manifest_path, checkpoint_dir, run_dir, checkpoint, model, original_model,
                     loaders, arm, accelerator, per_rank_acc, guard, *, batch_size, memory_probe=None):
    with rt.checkpoint_io(accelerator):
        if accelerator.device.type != 'cuda':
            raise ValueError('legacy compiler bootstrap execution requires CUDA')
        local = validate_legacy_manifest(manifest_path, checkpoint_dir, run_dir, checkpoint, arm=arm.name,
                                         world_size=accelerator.num_processes, batch_size=batch_size) if accelerator.is_main_process else None
    values = CompileHistory({}, {}, accelerator)._gather(lambda: local)
    history.bootstrap, rows = values[0]
    with rt.checkpoint_io(accelerator):
        payload = history.to_payload()
    require_rank_agreement(accelerator, payload)
    return execute_replay(history, model, original_model, loaders, arm, accelerator, per_rank_acc, guard,
                          memory_probe=memory_probe, legacy_rows=rows)
