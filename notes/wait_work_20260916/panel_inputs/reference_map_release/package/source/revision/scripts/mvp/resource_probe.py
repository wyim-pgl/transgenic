"""CUDA resource profiling only; no trainer override or production finalizer.

CPU-callable arithmetic below does not certify execution. The executable is
reachable only through real_probes' resource_budget mode and the bounded runner.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

from make_launch_readiness import _atomic_write, _canonical_bytes, _sha256, json_snapshot

TIERS = (30720, 61440, 129024)
STAGES = {'setup', 'inventory', 'validation', 'publication', 'revalidation'}


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def validate_budget(budget, binding_sha):
    from run_bounded import digest
    if (not isinstance(budget, dict) or type(budget.get('schema_version')) is not int or budget['schema_version'] != 1 or
            budget.get('authorized_scope') != 'exploratory_mvp' or
            budget.get('seeds') != [1, 2, 3] or
            any(type(seed) is not int for seed in budget.get('seeds', [])) or
            budget.get('resource_envelope') != dict(nodes=1, ranks=4, microbatches_per_rank=24) or
            any(type(n) is not int for n in budget['resource_envelope'].values()) or
            not digest(binding_sha) or budget.get('bindings_sha256') != binding_sha):
        raise ValueError('explicit bound three-seed controller resource envelope required')
    for key in ('controller_record_reference', 'intent', 'margin_rationale'):
        if not isinstance(budget.get(key), str) or not budget[key].strip():
            raise ValueError(f'explicit controller record requires {key}')
    for key in ('safety_margin', 'memory_headroom_fraction', 'max_wall_seconds',
                'max_three_seed_node_hours', 'max_probe_wall_seconds', 'launcher_overhead_seconds'):
        if not positive(budget.get(key)):
            raise ValueError(f'positive finite budget field required: {key}')
    spent = budget.get('prior_smoke_node_hours')
    if type(spent) not in (int, float) or not math.isfinite(spent) or spent < 0:
        raise ValueError('explicit already-spent smoke node-hours required')
    if (budget['safety_margin'] <= 1 or budget['memory_headroom_fraction'] >= 1 or
            budget['max_probe_wall_seconds'] + spent * 3600 > 5 * 3600):
        raise ValueError('prospective margin/headroom required; cannot enlarge 5 node-hour probe cap')
    return budget


def project_budget(budget, ranks):
    """Recompute, never trust a supplied projection. All probe elapsed is overhead.

    Deliberate overcount: nine probe updates, full eval, scan, input audits and
    serialization are added AGAIN to 250 worst steady updates, then margin.
    This avoids subtracting cold/setup costs or hiding long inventory audits.
    """
    from run_bounded import digest
    validate_budget(budget, budget.get('bindings_sha256'))
    if (not isinstance(ranks, list) or len(ranks) != 4 or
            any(not isinstance(r, dict) or type(r.get('rank')) is not int for r in ranks) or
            {r.get('rank') for r in ranks} != {0, 1, 2, 3}):
        raise ValueError('exactly four measured ranks required')
    worst, peak, limit, elapsed = 0., 0, float('inf'), 0.
    inventories = set()
    for rank in ranks:
        contract = rank.get('label_contract', {})
        if contract.get('gff_vocab_version') != 'v3' or type(contract.get('maxlength')) is not int or contract['maxlength'] != 8192:
            raise ValueError('measured v3/8192 label contract required')
        capacity = rank.get('capacity_bytes')
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('exact observed CUDA capacity required')
        limit = min(limit, capacity * (1 - budget['memory_headroom_fraction']))
        if set(rank.get('stages', {})) != STAGES or set(rank.get('tiers', {})) != {str(t) for t in TIERS}:
            raise ValueError('all measured stages and tiers required')
        if not positive(rank.get('elapsed_seconds')) or rank['elapsed_seconds'] >= budget['max_probe_wall_seconds']:
            raise ValueError('probe exceeds authorized cap; partial timings are not PASS')
        elapsed = max(elapsed, rank['elapsed_seconds'])
        if type(rank.get('valid_count')) is not int or rank['valid_count'] <= 0 or not digest(rank.get('valid_inventory_sha256')):
            raise ValueError('complete valid inventory measurement required')
        inventories.add((rank['valid_count'], rank['valid_inventory_sha256']))
        measurements = list(rank['stages'].values())
        for tier, coverage in rank['tiers'].items():
            updates = coverage.get('updates', [])
            if len(updates) != 3:
                raise ValueError('one cold and two steady updates per rank/tier required')
            for index, update in enumerate(updates):
                if (type(update.get('microbatches')) is not int or update['microbatches'] != 24 or
                        type(update.get('optimizer_steps')) is not int or update['optimizer_steps'] != 1 or
                        len(update.get('rns', [])) != 24 or
                        any(type(rn) is not int for rn in update['rns']) or len(set(update['rns'])) != 24 or
                        type(update.get('generated_kernels')) is not int or update['generated_kernels'] < 0 or
                        type(update.get('est_signal_microbatches')) is not int or
                        not 0 < update['est_signal_microbatches'] <= 24 or
                        update.get('input_lengths') != [int(tier)] * 24 or
                        any(type(n) is not int for n in update['input_lengths']) or
                        len(update.get('label_lengths', [])) != 24 or
                        any(type(n) is not int or n <= 0 for n in update['label_lengths'])):
                    raise ValueError('actual microbatch/update/length coverage incomplete')
                if index:
                    if update.get('generated_kernels') != 0:
                        raise ValueError('steady update recompiled; cannot extrapolate as steady')
                    if not positive(update.get('wall_seconds')):
                        raise ValueError('invalid steady timing')
                    worst = max(worst, update['wall_seconds'])
            measurements.extend(updates)
        for measured in measurements:
            if not positive(measured.get('wall_seconds')):
                raise ValueError('all stage times must be finite positive measurements')
            for key in ('peak_cuda_bytes', 'peak_reserved_bytes'):
                value = measured.get(key)
                if type(value) is not int or not 0 < value < capacity * (1 - budget['memory_headroom_fraction']):
                    raise ValueError('stage memory exceeds prospective headroom or is unmeasured')
                peak = max(peak, value)
        if sum(m['wall_seconds'] for m in measurements) > rank['elapsed_seconds']:
            raise ValueError('elapsed time cannot omit measured stages or updates')
    if len(inventories) != 1:
        raise ValueError('rank valid inventories disagree')
    for tier in map(str, TIERS):
        if any(type(r['tiers'][tier].get('inventory_max_label_length')) is not int for r in ranks):
            raise ValueError('integer inventory maximum label length required')
        maxima = {r['tiers'][tier].get('inventory_max_label_length') for r in ranks}
        observed = max(max(u['label_lengths']) for r in ranks for u in r['tiers'][tier]['updates'][1:])
        if maxima != {observed}:
            raise ValueError('steady sample must include inventory maximum label length at every tier')
        for index in range(3):
            rns = [rn for r in ranks for rn in r['tiers'][tier]['updates'][index]['rns']]
            if len(set(rns)) != 96:
                raise ValueError('rank training samples must be disjoint within each update')
    projected = (250 * worst + elapsed) * budget['safety_margin'] + budget['launcher_overhead_seconds']
    three = projected * 3 / 3600
    if peak >= limit:
        raise ValueError('conservative cross-rank memory headroom exceeded')
    if projected >= budget['max_wall_seconds'] or three >= budget['max_three_seed_node_hours']:
        raise ValueError('conservative 250-update/three-seed budget exceeded')
    return dict(measured_step_seconds=worst, peak_memory_bytes=peak, memory_limit_bytes=limit,
        authorized_wall_seconds=budget['max_wall_seconds'], projected_run_seconds=projected,
        projected_three_seed_node_hours=three, measured_probe_seconds=elapsed,
        total_smoke_node_hours=budget['prior_smoke_node_hours'] + elapsed / 3600,
        projected_campaign_node_hours=three + budget['prior_smoke_node_hours'] + elapsed / 3600,
        projection_formula='(250 * worst_rank_tier_steady + entire_probe_elapsed) * margin + launcher_allowance')


def validate_resource(value):
    """Receipt boundary: legacy CUDA booleans alone cannot certify this mode."""
    from real_probes import validate_identity
    if value.get('mode') != 'resource_budget' or value.get('profiling_complete') is not True:
        raise ValueError('complete real resource probe required, not synthetic runtime flags')
    validate_identity(value)
    from transgenic.training.mvp_cli_runtime import _validate_public_hardware
    _validate_public_hardware(value.get('hardware', []))
    budget = json_snapshot(value.get('authorization_path'), value.get('authorization_sha256'))
    validate_budget(budget, value.get('bindings_sha256'))
    expected = project_budget(budget, value.get('ranks'))
    if any(value.get(key) != item for key, item in expected.items()):
        raise ValueError('resource projection does not match measured stages')


def require_v3_dataset(dataset, cfg):
    if (cfg.get('gff_vocab_version') != 'v3' or cfg.get('max_decoder_position_embeddings') != 8192 or
            dataset.gff_vocab_version != 'v3' or dataset.maxlength != 8192 or
            len(dataset.dt.vocab) != cfg.get('vocab_size')):
        raise ValueError('resource certification requires reviewed v3/8192 dataset/config wiring')


def strict_compile_limits(config):
    # Torch renamed cache limits to recompile limits. Neither supported means
    # unsupported runtime, not permission to execute cached eager fallbacks.
    for key in ('fail_on_recompile_limit_hit', 'fail_on_cache_limit_hit'):
        if hasattr(config, key):
            setattr(config, key, True)
            config.suppress_errors = False
            return
    raise RuntimeError('unsupported CUDA compiler: cannot prohibit cache-limit eager fallback')


def paired_validation(accelerator, model, loader, expected):
    """Actual finalizer metric, byte transport and aggregation, without publication."""
    from transgenic.training import mvp_finalize as f
    from transgenic.training.mvp_metrics import paired_metric_record, aggregate_paired_records, validate_paired_batch
    lengths = f._all_ints(accelerator, len(loader))
    if len(set(lengths)) != 1:
        raise ValueError('unequal paired validation lengths')
    records, error = [], None
    times = dict(read_transfer_seconds=0., paired_metric_seconds=0., transport_reduce_seconds=0.)
    iterator = None
    def open_iterator():
        nonlocal iterator
        iterator = iter(loader)
    f._collective_preflight(accelerator, open_iterator)
    import torch
    for _ in range(len(loader)):
        start = time.monotonic()
        batch = None
        def retrieve():
            nonlocal batch
            batch = f._to_device(next(iterator), accelerator.device)
            validate_paired_batch(batch)
            torch.cuda.synchronize(accelerator.device)
        f._collective_preflight(accelerator, retrieve)
        times['read_transfer_seconds'] += time.monotonic() - start
        start = time.monotonic()
        # DDP forwards may contain collectives: do not nest them in preflight.
        # A backend/collective failure is terminated by the bounded launcher.
        try:
            records.append(paired_metric_record(model, batch))
            torch.cuda.synchronize(accelerator.device)
        except Exception as exc:
            error = exc
        def post_metric():
            if error is not None:
                raise error
        f._collective_preflight(accelerator, post_metric)
        times['paired_metric_seconds'] += time.monotonic() - start
    start = time.monotonic()
    payload = b''
    try:
        payload = f._validate_bounds(records)
    except Exception as exc:
        error = error or exc
    lengths = f._transport_lengths(accelerator, error, payload)
    payloads = f._gather_record_payload_bytes(accelerator, payload, lengths)
    status, summary_payload = None, b''
    if accelerator.is_main_process:
        try:
            gathered = [row for payload in payloads for row in json.loads(payload)]
            f._validate_bounds(gathered)
            summary_payload = f._canonical_json(aggregate_paired_records(gathered, expected))
            if len(summary_payload) > f.MVP_MAX_SERIALIZED_BYTES:
                raise ValueError('paired summary exceeds transport cap')
            status = f._status()
        except Exception as exc:
            status = f._status(exc)
    f._raise_status(f._broadcast_status(accelerator, status))
    summary = json.loads(f._broadcast_payload_bytes(accelerator, summary_payload))
    times['transport_reduce_seconds'] = time.monotonic() - start
    return summary, times


def run(args):
    """Nine actual updates plus full validation; interruptions leave profiling only."""
    import real_probes as p
    # No authorization inferred from smoke-cap notes, an approval flag, or this code.
    for name in ('budget_authorization', 'budget_sha256', 'profile_dir', 'acceptance_record',
                 'inventory_record', 'receipts_record', 'panel', 'sif'):
        if not getattr(args, name, None):
            raise ValueError(f'resource_budget requires --{name.replace("_", "-")}')
    started = time.monotonic()
    if 'TRANSGENIC_RUN_DIR' in os.environ:
        raise ValueError('TRANSGENIC_RUN_DIR is prohibited for resource profiling')
    budget = json_snapshot(args.budget_authorization, args.budget_sha256)
    validate_budget(budget, os.environ.get('MVP_BINDINGS_SHA256'))
    timeout = float(os.environ.get('MVP_PROBE_TIMEOUT_SECONDS', 'nan'))
    if not positive(timeout) or timeout > budget['max_probe_wall_seconds']:
        raise ValueError('bounded runner timeout must fit explicit probe cap')
    p.require_cuda()
    import torch
    from torch.utils.data import DataLoader, DistributedSampler
    from torch._inductor import metrics
    from accelerate.utils import gather_object
    from transgenic.training.mvp_runtime import MVPRecipe, make_accelerator, build_optimizer
    from transgenic.training.mvp_cli import validate_candidate_records
    from transgenic.training.mvp_cli_runtime import (_collect_runtime_hardware, _validate_public_hardware,
        compile_bound_forward, verify_candidate_artifacts, _close_dataset, require_matching_identity,
        validate_decoder_dataset)
    from transgenic.training import mvp_finalize as f
    from transgenic.datasets.datasets import isoformDataHyena, hyena_collate_fn
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.cuda.reset_peak_memory_stats()
    accelerator = make_accelerator(MVPRecipe(), mixed_precision='bf16')
    ds = valid = None
    profile = None
    row = dict(rank=accelerator.process_index, stages={}, tiers={})

    def local(operation):
        # Only rank-local work here: never nest DDP/transport collectives.
        result = None
        def checked():
            nonlocal result
            result = operation()
        f._collective_preflight(accelerator, checked)
        return result

    progress_sequence = 0

    def progress():
        nonlocal progress_sequence
        def publish():
            # Immutable snapshots, never MVP_MEASUREMENT_OUTPUT or receipt/DONE.
            try:
                _atomic_write(profile / f'rank{accelerator.process_index}.{progress_sequence:06d}.profiling.json',
                    _canonical_bytes(dict(output_use='INTERMEDIATE_PROFILING_NOT_READINESS', measurement=row)))
            except SystemExit as error:
                # The fresh-path writer uses SystemExit; preflight exchanges Exception.
                raise RuntimeError(str(error)) from error
            if time.monotonic() - started >= budget['max_probe_wall_seconds']:
                raise RuntimeError('authorized probe cap exhausted; retain partial profiling only')
        local(publish)
        progress_sequence += 1

    def stage(name, operation, *, collective=False):
        accelerator.wait_for_everyone()
        local(lambda: (torch.cuda.synchronize(accelerator.device),
                       torch.cuda.reset_peak_memory_stats(accelerator.device)))
        start = time.monotonic()
        result = operation() if collective else local(operation)
        def finish():
            torch.cuda.synchronize(accelerator.device)
            row['stages'][name] = dict(wall_seconds=time.monotonic() - start,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(accelerator.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(accelerator.device))
        local(finish)
        progress()
        return result

    def audit_inputs():
        candidate = validate_candidate_records(args.acceptance_record,
            bindings['technical_acceptance_sha256'], args.inventory_record, args.receipts_record)
        if candidate.identities != dict(acceptance=bindings['technical_acceptance_sha256'],
                inventory=bindings['inventory_sha256'], receipts=bindings['receipts_sha256']):
            raise ValueError('candidate identity differs from prepared bindings')
        artifacts = verify_candidate_artifacts(candidate, args.db, args.evidence_db, args.panel)
        if _sha256(Path(args.sif)) != bindings['sif_sha256']:
            raise ValueError('SIF bytes differ from prepared bindings')
        return artifacts

    def profile_path():
        path = Path(args.profile_dir).resolve()
        scratch = Path(args.scratch_root).resolve(strict=True)
        if not path.is_relative_to(scratch) or path == scratch:
            raise ValueError('fresh profile directory must be beneath independent scratch root')
        return path

    try:
        profile = local(profile_path)
        f._collective_preflight(accelerator, lambda: profile.mkdir() if accelerator.is_main_process else None)
        bindings, roots, runtime = local(lambda: p.setup(args))
        hardware = _validate_public_hardware(_collect_runtime_hardware(accelerator))
        require_matching_identity(accelerator, _canonical_bytes(budget))
        row['capacity_bytes'] = local(lambda: torch.cuda.get_device_properties(accelerator.device).total_memory)
        model, cfg, audit = local(lambda: p.load_model(args, bindings))
        model_config = model.config  # Preserve the bound config before compilation/DDP wrapping.
        ds = local(lambda: p.datasets(args, bindings, cfg, model.config))
        local(lambda: validate_decoder_dataset(ds, cfg, model_config))
        local(lambda: require_v3_dataset(ds, cfg))
        row['label_contract'] = dict(gff_vocab_version=ds.gff_vocab_version,
                                     maxlength=ds.maxlength, vocab_size=len(ds.dt.vocab))
        local(lambda: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}))
        optimizer = local(lambda: build_optimizer(model, MVPRecipe()))
        local(lambda: strict_compile_limits(torch._dynamo.config))
        local(lambda: compile_bound_forward(model))
        model, optimizer = accelerator.prepare(model, optimizer)
        def setup_finished():
            torch.cuda.synchronize(accelerator.device)
            row['stages']['setup'] = dict(wall_seconds=time.monotonic() - started,
                peak_cuda_bytes=torch.cuda.max_memory_allocated(accelerator.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(accelerator.device))
        local(setup_finished)
        progress()

        def inventory():
            nonlocal valid
            artifacts = audit_inputs()
            valid = isoformDataHyena(args.db, mode='train', split='valid', strict=True,
                species='Athaliana', evidence_db=args.evidence_db, evidence_k=8,
                aug_seed=20260914 + bindings['seed'], eval_fraction=1, return_tile_ids=True,
                gff_vocab_version=cfg['gff_vocab_version'])
            validate_decoder_dataset(valid, cfg, model_config)
            require_v3_dataset(valid, cfg)
            if ds._index_map != artifacts['train_rns'] or valid._index_map != artifacts['valid_rns']:
                raise ValueError('dataset differs from accepted full inventory')
            selections = {tier: [] for tier in TIERS}
            # Full scan validates actual labels/evidence; its cost is NOT excluded.
            for index, rn in enumerate(ds._index_map):
                sample = ds[index]
                tier, labels = sample[0].shape[-1], sample[2].shape[-1]
                if tier not in selections:
                    raise ValueError('uncovered train sequence tier')
                selections[tier].append((labels, rn, index))
            return artifacts, selections
        artifacts, selections = stage('inventory', inventory)
        expected = artifacts['valid_rns']
        row.update(valid_count=len(expected),
                   valid_inventory_sha256=hashlib.sha256(_canonical_bytes(expected)).hexdigest())
        require_matching_identity(accelerator, _canonical_bytes(selections))
        ds.eval_fraction = None
        ds._aug_generator = torch.Generator().manual_seed(20260914 + bindings['seed'] + accelerator.process_index)
        ds._aug_key = (os.getpid(), None)
        generated_before = metrics.generated_kernel_count
        for tier in TIERS:
            ordered = sorted(selections[tier], reverse=True)
            if len(ordered) < 96:
                raise ValueError('need 96 distinct real train rows at each tier; no fabricated repetition within update')
            # Longest real labels, deterministic disjoint rank shards. Reuse the
            # same RNs across updates, explicitly recorded, with fresh augmentation.
            chosen = ordered[:96][accelerator.process_index::4]
            row['tiers'][str(tier)] = dict(inventory_count=len(ordered),
                inventory_min_label_length=min(x[0] for x in ordered),
                inventory_max_label_length=ordered[0][0], updates=[])
            for update in range(3):
                accelerator.wait_for_everyone()
                local(lambda: torch.cuda.synchronize(accelerator.device))
                start = time.monotonic()
                kernels = metrics.generated_kernel_count
                batches = p.collective_batches(accelerator,
                    (hyena_collate_fn([ds[index]]) for _, _, index in chosen))
                # Retains shared probe LR semantics; does not override the trainer.
                with torch._dynamo.config.patch(error_on_recompile=bool(update)):
                    measured = p.one_update(accelerator, model, optimizer, batches)
                def finish_update():
                    measured.update(wall_seconds=time.monotonic() - start,
                        rns=[int(b[9]['rn'].item()) for b in batches],
                        label_lengths=[b[2].shape[-1] for b in batches],
                        input_lengths=[b[0].shape[-1] for b in batches],
                        est_signal_microbatches=sum(bool(torch.count_nonzero(b[8][..., :7])) for b in batches),
                        generated_kernels=metrics.generated_kernel_count - kernels,
                        phase='cold_compile_warmup' if update == 0 else 'steady')
                    if not measured['est_signal_microbatches']:
                        raise ValueError('no nonzero real EST observed in tier/rank update')
                local(finish_update)
                row['tiers'][str(tier)]['updates'].append(measured)
                del batches
                progress()
                def check_kernels():
                    if update and measured['generated_kernels']:
                        raise RuntimeError('steady update generated new kernels; partial profiling only')
                    if torch._dynamo.config.suppress_errors:
                        raise RuntimeError('eager fallback enabled; resource probe cannot pass')
                local(check_kernels)
        p.collective_compiler_check(accelerator, generated_before)
        loader = local(lambda: DataLoader(valid, batch_size=1, num_workers=0, collate_fn=hyena_collate_fn,
            sampler=DistributedSampler(valid, num_replicas=4, rank=accelerator.process_index,
                                       shuffle=False, drop_last=False), pin_memory=True))
        summary, evaluation_times = stage('validation', lambda: paired_validation(accelerator, model, loader, expected), collective=True)
        row['stages']['validation'].update(evaluation_times)

        def publication():
            if not accelerator.is_main_process:
                return
            # Low-level payload serializer only: truthful probe metadata, never
            # _publish_final / _write_done_marker / fabricated 250-update provenance.
            destination = profile / 'PROBE_ONLY_CHECKPOINT'
            destination.mkdir()
            names = f._write_payload_files(destination, f._save_target(accelerator, model), summary,
                {'identity_scope': 'resource_probe_only', 'output_use': 'DISPOSABLE_NOT_TRAINING_COMPLETION'},
                {'optimizer_steps': 9, 'microbatches_per_rank': 216, 'world_size': 4})
            f._write_manifest(destination, names)
            f._fsync_parent(destination / 'manifest.json')
            f._fsync_parent(destination)
            return dict(artifact_path=str(destination), filesystem_device=profile.stat().st_dev)
        publication_info = stage('publication', publication)
        if publication_info:
            row['stages']['publication'].update(publication_info)

        def revalidate():
            _close_dataset(ds)
            _close_dataset(valid)
            if audit_inputs() != artifacts:
                raise ValueError('accepted input changed during profiling')
            p.setup(args)  # Independent full input/cache/manifest revalidation, timed.
            source = bindings['source_checkpoint']
            for prefix in ('weights', 'run_config', 'runtime_identity'):
                if _sha256(Path(source[prefix + '_path'])) != source[prefix + '_sha256']:
                    raise ValueError('source checkpoint changed during profiling')
            if p.optional_sidecars(Path(source['weights_path']).parent) != source['optional_sidecars']:
                raise ValueError('source sidecars changed during profiling')
            if json_snapshot(args.budget_authorization, args.budget_sha256) != budget:
                raise ValueError('budget changed during profiling')
        stage('revalidation', revalidate)
        def imported_identity():
            imports = {str(Path(p.__file__).resolve()): _sha256(Path(p.__file__))}
            for module in list(sys.modules.values()):
                path = getattr(module, '__file__', None)
                if path and Path(path).is_file():
                    imports[str(Path(path).resolve())] = _sha256(Path(path))
            return imports
        imports = local(imported_identity)
        result = dict(device='cuda', real_runtime=True, mode='resource_budget', runtime=runtime,
            hardware=hardware, bindings_sha256=os.environ['MVP_BINDINGS_SHA256'],
            authorization_path=str(Path(args.budget_authorization).resolve()), authorization_sha256=args.budget_sha256,
            initialization_audit_sha256=hashlib.sha256(_canonical_bytes(audit)).hexdigest(),
            probe_identity=dict(schema_version=1, probe_path=str(Path(p.__file__).resolve()),
                                imports=imports, asset_roots=roots))
        local(lambda: p.validate_identity(result))  # Include import/cache rehash in elapsed.
        accelerator.wait_for_everyone()
        row['elapsed_seconds'] = time.monotonic() - started
        progress()
        result['ranks'] = gather_object([row])
        result.update(project_budget(budget, result['ranks']), profiling_complete=True)
        # The bounded runner independently revalidates identity and arithmetic.
        if accelerator.is_main_process:
            _atomic_write(Path(os.environ['MVP_MEASUREMENT_OUTPUT']), _canonical_bytes(result))
        return 0
    finally:
        for dataset in (ds, valid):
            if dataset is not None:
                _close_dataset(dataset)
        accelerator.end_training()
