"""Receipt-bound C2 allocation observer; never replaces the ordinary input verifier."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

PROFILE = dict(arm='C2', target=4099, seed=123, world_size=4, batch_size=1,
               mixed_precision='bf16', deterministic=True)
SOURCE = '188bf3e9f6afcc6389b54c123d5bf533a90384cf'
POSITION = ('epoch', 'step', 'global_step', 'pending_micro')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, 'duplicate JSON key: ' + key)
        value[key] = item
    return value


def read_json(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'nonregular JSON: ' + str(path))
    return json.loads(path.read_text(), object_pairs_hook=unique,
                      parse_constant=lambda x: require(False, 'nonfinite JSON: ' + x))


def beneath(root, relative):
    require(isinstance(relative, str) and bool(relative), 'empty path')
    parts = relative.split('/')
    require(not relative.startswith('/') and all(p not in ('', '.', '..') for p in parts), 'unsafe relative path')
    root = Path(root).absolute()
    path = Path(root.anchor)
    for part in root.parts[1:]:
        path = path / part
        require(not path.is_symlink(), 'symlink root ancestor')
    for part in parts:
        path = path / part
        require(not path.is_symlink(), 'symlink control path: ' + str(path))
    return path


def integer(value, name, minimum=0):
    require(type(value) is int and value >= minimum, 'invalid integer ' + name)
    return value


def hash_value(value, name, length=64):
    require(isinstance(value, str) and re.fullmatch('[0-9a-f]{%d}' % length, value) is not None, 'invalid hash ' + name)


def exclusive_json(path, value):
    path = Path(path)
    raw = canonical(value) + b'\n'
    # Link the fsynced temporary into place; publication never replaces old evidence.
    tmp = path.with_name(path.name + '.tmp.%d' % os.getpid())
    with open(tmp, 'xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    finally:
        tmp.unlink()
    fd = os.open(str(path.parent), os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def inventory(directory):
    result = {}
    for root, dirs, files in os.walk(directory, followlinks=False):
        for name in dirs + files:
            require(not (Path(root) / name).is_symlink(), 'symlink checkpoint entry')
        for name in files:
            path = Path(root) / name
            require(path.is_file(), 'nonregular checkpoint file')
            result[str(path.relative_to(directory))] = dict(size=path.stat().st_size, sha256=sha(path))
    return result


def finite(value):
    import numpy as np
    import torch
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), 'nonfinite tensor')
    elif isinstance(value, np.ndarray):
        require(bool(np.isfinite(value).all()), 'nonfinite array')
    elif isinstance(value, dict):
        for item in value.values(): finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value: finite(item)
    elif isinstance(value, (float, np.floating)):
        require(math.isfinite(value), 'nonfinite scalar')


def checkpoint_common(directory, identity, fingerprint):
    from transgenic.training import b6_runtime as rt
    from transgenic.training import b6_compile_replay as replay
    from transgenic.training.b6_compile_history import CompileHistory
    directory = beneath(Path(directory).parent, Path(directory).name)
    for name in ('meta.json', rt.CHECKPOINT_IDENTITY, replay.HISTORY_FILE):
        beneath(directory, name)
    require(directory.is_dir() and not directory.is_symlink(), 'invalid checkpoint directory')
    meta = read_json(directory / 'meta.json')
    for key in POSITION: integer(meta.get(key), key)
    require(meta.get('arm') == 'C2' and type(meta.get('seed')) is int and meta['seed'] == 123, 'checkpoint profile')
    stored = read_json(directory / rt.CHECKPOINT_IDENTITY)
    rt.check_run_config_b6(stored, identity)
    require(stored == identity, 'checkpoint must bind current identity exactly')
    if 'identity_sha256' in meta:
        require(meta['identity_sha256'] == rt.canonical_sha256(stored), 'identity digest mismatch')
    history = CompileHistory.load(directory / replay.HISTORY_FILE, meta.get('compile_history_sha256'),
                                  identity, fingerprint, SimpleNamespace(num_processes=4, process_index=0),
                                  checkpoint=replay.checkpoint_position(meta))
    require(bool(history.entries), 'empty trained compiler history')
    return meta, history


def load_weights(path):
    from safetensors.torch import load_file
    require(path.is_file() and not path.is_symlink(), 'missing model')
    weights = load_file(str(path), device='cpu')
    require(bool(weights), 'empty model')
    finite(weights)
    return weights


def check_checkpoint(directory, identity, fingerprint, *, capture_contract=None, expected_contract=None):
    import torch
    import probe
    from transgenic.training import b6_runtime as rt
    directory = Path(directory)
    meta, _ = checkpoint_common(directory, identity, fingerprint)
    require(meta['global_step'] < 4099, 'resume checkpoint is at or beyond final target')
    acc = integer(identity.get('accumulation_steps'), 'accumulation_steps', 4)
    require(acc % 4 == 0 and meta['pending_micro'] < acc // 4, 'invalid pending accumulation')
    weights = load_weights(beneath(directory, 'accelerate_state/model.safetensors'))
    state = {}
    for name in ['optimizer.bin', 'scheduler.bin'] + ['random_states_%d.pkl' % r for r in range(4)]:
        path = beneath(directory, 'accelerate_state/' + name)
        require(path.is_file(), 'missing state ' + name)
        value = torch.load(path, map_location='cpu', weights_only=False)
        require(type(value) is dict and bool(value), 'empty state ' + name)
        finite(value)
        state[name] = value
    optimizer = state['optimizer.bin']
    require(type(optimizer.get('state')) is dict and bool(optimizer['state']) and
            type(optimizer.get('param_groups')) is list and bool(optimizer['param_groups']), 'partial optimizer')
    parameters = [n for group in optimizer['param_groups'] for n in group.get('params', [])]
    require(parameters and len(set(parameters)) == len(parameters) and set(optimizer['state']) <= set(parameters), 'optimizer parameter groups')
    for value in optimizer['state'].values():
        require(type(value) is dict and {'step', 'exp_avg', 'exp_avg_sq'} <= set(value), 'partial AdamW state')
        require(isinstance(value['exp_avg'], torch.Tensor) and value['exp_avg'].shape == value['exp_avg_sq'].shape, 'optimizer moment shape')
    scheduler = state['scheduler.bin']
    require({'last_epoch', '_step_count', '_last_lr'} <= set(scheduler), 'partial scheduler')
    for rank in range(4):
        rng = state['random_states_%d.pkl' % rank]
        require({'random_state', 'numpy_random_seed', 'torch_manual_seed', 'torch_cuda_manual_seed'} <= set(rng), 'partial rank RNG')
        import random
        import numpy as np
        random.Random().setstate(rng['random_state'])
        np.random.RandomState().set_state(rng['numpy_random_seed'])
        torch.Generator().set_state(rng['torch_manual_seed'])
        require(isinstance(rng['torch_cuda_manual_seed'], (list, tuple)) and bool(rng['torch_cuda_manual_seed']), 'missing CUDA RNG')
        for seed in rng['torch_cuda_manual_seed']:
            require(isinstance(seed, torch.Tensor) and seed.dtype == torch.uint8 and seed.numel() > 0, 'invalid CUDA RNG')
    contract = dict(model={k: dict(shape=list(v.shape), dtype=str(v.dtype)) for k,v in weights.items()},
                    optimizer_groups=[group['params'] for group in optimizer['param_groups']],
                    optimizer_state={str(k): {n: dict(shape=list(v[n].shape),dtype=str(v[n].dtype))
                                             for n in ('exp_avg','exp_avg_sq')} for k,v in optimizer['state'].items()},
                    scheduler_keys=sorted(scheduler),
                    rng_keys={str(r): sorted(state['random_states_%d.pkl' % r]) for r in range(4)})
    if expected_contract is not None:
        for key in ('model','optimizer_groups','scheduler_keys','rng_keys'):
            require(contract[key] == expected_contract[key], 'checkpoint structure changed: ' + key)
        for key,value in expected_contract['optimizer_state'].items():
            require(contract['optimizer_state'].get(key) == value, 'initialized optimizer state disappeared or changed shape')
    if capture_contract is not None:
        capture_contract.update(contract)
    # Preserve probe.checkpoint's double tree canonicalization exactly.
    tree = dict(model={k: probe.tree(v) for k, v in weights.items()},
                state={k: probe.tree(v) for k, v in state.items()})
    if meta['pending_micro']:
        grads = {}
        for rank in range(4):
            path = beneath(directory, Path(rt.pending_grads_path(str(directory), rank)).name)
            require(path.is_file() and not path.is_symlink(), 'missing pending rank gradients')
            values = torch.load(path, map_location='cpu', weights_only=False)
            require(type(values) is dict and bool(values), 'empty pending gradients')
            for name, tensor in values.items():
                require(name in weights and isinstance(tensor, torch.Tensor) and tensor.shape == weights[name].shape and
                        tensor.dtype == weights[name].dtype, 'pending gradient model mismatch')
            finite(values)
            require(rank == 0 or set(values) == gradient_names, 'partial gradients across ranks')
            gradient_names = set(values)
            grads[str(rank)] = probe.tree(values)
        tree['pending_grads'] = grads
    return dict(position={key: meta[key] for key in POSITION}, relative_dir=directory.name,
                meta_sha256=sha(directory / 'meta.json'), identity_sha256=sha(directory / rt.CHECKPOINT_IDENTITY),
                history_sha256=meta['compile_history_sha256'], canonical_checkpoint_sha256=probe.digest(tree), files=inventory(directory))


def check_final(run, identity, fingerprint, *, expected_model=None):
    import probe
    from transgenic.training import b6_runtime as rt
    run = Path(run)
    directory = Path(rt.step_checkpoint_dir(str(run), 4099))
    meta, _ = checkpoint_common(directory, identity, fingerprint)
    require(meta['global_step'] == 4099 and meta['pending_micro'] == 0 and
            type(meta.get('stop_after_updates')) is int and meta['stop_after_updates'] == 4099 and
            meta.get('pilot_final') is True and meta.get('checkpoint_kind') == 'fixed_step' and
            meta.get('selection') == rt.FIXED_STEP_SELECTION, 'invalid final metadata')
    weights = load_weights(beneath(directory, 'model.safetensors'))
    if expected_model is not None:
        aliases = meta.get('tied_aliases', {})
        require(type(aliases) is dict, 'invalid tied aliases')
        for name, descriptor in expected_model.items():
            kept = name if name in weights else aliases.get(name)
            require(kept in weights and descriptor == dict(shape=list(weights[kept].shape), dtype=str(weights[kept].dtype)),
                    'final model tensor missing or changed shape: ' + name)
    require(sha(directory / 'model.safetensors') == meta.get('model_sha256'), 'final weights digest')
    markers = {rt.PILOT_FINAL_MARKER: 'WP5 pilot-final checkpoint at exactly 4099 optimizer updates (A46 rule 11); never a best-validation checkpoint\n',
               rt.PRUNE_EXEMPT_MARKER: 'A28/plan 9: step checkpoint, never pruned\n'}
    for name, content in markers.items():
        path = directory / name
        require(path.is_file() and not path.is_symlink() and path.read_text() == content, 'invalid final marker ' + name)
    done = run / 'TRAINING_DONE'
    require(done.is_file() and not done.is_symlink() and bool(done.read_text().strip()), 'missing DONE')
    return dict(position={key: meta[key] for key in POSITION}, relative_dir=directory.name,
                meta_sha256=sha(directory / 'meta.json'), identity_sha256=sha(directory / rt.CHECKPOINT_IDENTITY),
                history_sha256=meta['compile_history_sha256'], canonical_checkpoint_sha256=probe.digest({'model': {k: probe.tree(v) for k,v in weights.items()}}),
                files=inventory(directory), done_sha256=sha(done))


def validate_request(request):
    fields = {'version','operation_id','token','link_index','source_commit','source_snapshot_git','source_dir',
              'source_manifest_sha256','helper_dir','helper_manifest_sha256','run_dir','bindings','bindings_sha256',
              'config_relative','config_sha256','receipt_dir','profile','start_checkpoint','predecessor',
              'expected_runtime_fingerprint','completed_epoch_events'}
    require(type(request) is dict and set(request) == fields, 'request fields mismatch')
    require(type(request['version']) is int and request['version'] == 1, 'request version')
    integer(request['link_index'], 'link_index', 1)
    for key in ('operation_id', 'token'):
        require(isinstance(request[key], str) and re.fullmatch('[A-Za-z0-9_-]{1,128}', request[key]), 'invalid ' + key)
    hash_value(request['source_commit'], 'source_commit', 40)
    hash_value(request['source_snapshot_git'], 'snapshot', 40)
    for key in ('source_manifest_sha256','helper_manifest_sha256','bindings_sha256','config_sha256'):
        hash_value(request[key], key)
    for key in ('source_dir','helper_dir','run_dir','bindings','config_relative','receipt_dir'):
        beneath('/__request_schema_only__', request[key])
    require(type(request['profile']) is dict and canonical(request['profile']) == canonical(PROFILE), 'production profile mismatch')
    start = request['start_checkpoint']
    require(type(start) is dict and set(start) == set(POSITION) | {'relative_dir','meta_sha256','identity_sha256',
                       'history_sha256','canonical_checkpoint_sha256'}, 'start checkpoint fields')
    require(isinstance(start['relative_dir'], str) and re.fullmatch('latest_state|epoch_[0-9]{2,}',start['relative_dir']), 'invalid resume directory')
    for key in POSITION: integer(start[key], key)
    require(0 < start['global_step'] < 4099, 'invalid start step')
    for key in ('meta_sha256','identity_sha256','history_sha256','canonical_checkpoint_sha256'): hash_value(start[key], key)
    prior = request['predecessor']
    require(type(prior) is dict and set(prior) == {'relative_path','sha256'}, 'predecessor fields')
    beneath('/__request_schema_only__',prior['relative_path'])
    hash_value(prior['sha256'], 'predecessor')
    require(type(request['expected_runtime_fingerprint']) is dict and bool(request['expected_runtime_fingerprint']), 'missing fingerprint')
    require(type(request['completed_epoch_events']) is list, 'epoch events list required')


def check_manifest(root, name, expected):
    path = beneath(root, name)
    hash_value(expected, 'manifest')
    require(path.is_file() and sha(path) == expected, 'manifest pin mismatch')
    seen = set()
    for line in path.read_text().splitlines():
        match = re.fullmatch(r'([0-9a-f]{64}) [ *](.+)', line)
        require(match is not None, 'invalid manifest line')
        value, relative = match.groups()
        require(relative not in seen, 'duplicate manifest entry')
        seen.add(relative)
        item = beneath(root, relative)
        require(item.is_file() and sha(item) == value, 'manifest file changed: ' + relative)
    require(bool(seen), 'empty manifest')
    return seen


def pinned_request(work, request_path, request_sha256):
    hash_value(request_sha256, 'request')
    path = beneath(work, request_path)
    require(path.is_file() and sha(path) == request_sha256, 'request pin mismatch')
    request = read_json(path)
    validate_request(request)
    source = beneath(work, request['source_dir'])
    helpers = beneath(work, request['helper_dir'])
    files = check_manifest(source, 'validation_sources.sha256', request['source_manifest_sha256'])
    require({'train/train_HyenaTransgenic_B6.py','src/transgenic/training/b6_runtime.py',
             'src/transgenic/training/b6_compile_history.py','src/transgenic/training/b6_compile_replay.py'} <= files, 'source manifest incomplete')
    files = check_manifest(helpers, 'manifest.sha256', request['helper_manifest_sha256'])
    require({'production_link.py','production_link.sbatch','probe.py'} <= files, 'helper manifest incomplete')
    actual = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'], universal_newlines=True).strip()
    require(actual == request['source_snapshot_git'], 'source snapshot changed')
    for root, relative, expected in [(source, request['config_relative'], request['config_sha256']),
                                     (work, request['bindings'], request['bindings_sha256']),
                                     (work, request['predecessor']['relative_path'], request['predecessor']['sha256'])]:
        item = beneath(root, relative)
        require(item.is_file() and sha(item) == expected, 'control file changed: ' + relative)
        read_json(item)
    return request


def controls(work, request_path, request_sha256):
    request = pinned_request(work, request_path, request_sha256)
    run = beneath(work, request['run_dir'])
    require(run.is_dir(), 'missing canonical run')
    require(not any(re.fullmatch(r'(latest_state|epoch_[0-9]+|step_[0-9]+)\.tmp', name)
                    for name in os.listdir(run)), 'uncommitted checkpoint publication; audit required')
    from transgenic.training import b6_runtime as rt
    for name in ('TRAINING_FAILED', rt.MEMORY_ABORT_MARKER):
        require(not os.path.lexists(run / name), 'failure marker present: ' + name)
    return request


def execution_dir(work, request, job_id):
    require(isinstance(job_id, str) and re.fullmatch('[1-9][0-9]*',job_id), 'invalid allocation job ID')
    return beneath(work, request['receipt_dir'] + '/execution_' + job_id)


def check_draws(run, position, counts):
    """Reject all tails; never truncate or choose an older checkpoint here."""
    batches = integer(counts['rank_batches'], 'rank_batches', 1)
    total = position['epoch'] * batches + position['step']
    require(position['step'] <= batches, 'checkpoint beyond epoch loader')
    require(total == position['global_step'] * counts['per_rank_accumulation'] + position['pending_micro'], 'checkpoint sampler/accumulation mismatch')
    result = {}
    for rank in range(4):
        path = beneath(run, 'draw_log/rank%02d.tsv' % rank)
        n = 0
        with open(path, 'r') as handle:
            for n, line in enumerate(handle, 1):
                parts = line.rstrip('\n').split('\t')
                require(line.endswith('\n') and len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit(), 'invalid draw log')
                hash_value(parts[2], 'draw hash')
                require(int(parts[1]) == (n - 1) // batches, 'draw epoch disagreement')
        require(n == total, 'uncommitted or short draw log; audit required')
        result[str(rank)] = dict(rows=n, sha256=sha(path))
    return result


def validate_rank_records(records, request, job_id, request_sha256):
    require(type(records) is list and len(records) == 4, 'four rank observations required')
    require([r.get('rank') for r in records] == list(range(4)) and all(type(r.get('rank')) is int for r in records), 'rank identities')
    first = records[0]
    for row in records:
        require(row.get('job_id') == job_id and row.get('request_sha256') == request_sha256, 'stale rank receipt')
        verification = row.get('input_verification')
        require(type(verification) is dict and verification.get('job_id') == job_id and
                verification.get('bindings_sha256') == request['bindings_sha256'] and
                type(verification.get('record')) is dict and bool(verification['record']), 'missing fresh input verifier')
        require(verification == first.get('input_verification'), 'rank verifier identity mismatch')
        require(row.get('runtime_fingerprint') == request['expected_runtime_fingerprint'], 'effective runtime mismatch')
        for key in ('identity','counts','result'):
            require(type(row.get(key)) is dict and row[key] == first[key], 'rank disagreement: ' + key)
        result = row['result']
        step = integer(result.get('global_step'), 'result global step',1)
        require(request['start_checkpoint']['global_step'] < step <= 4099, 'no progress or target exceeded')
        require(type(result.get('history')) is list, 'actual trainer history required')
        if result.get('done') is True:
            require(step == 4099 and 'preempted' not in result and result.get('reason') == 'stop_after_updates', 'invalid final result')
        else:
            require(result.get('preempted') is True and 'done' not in result and step < 4099, 'invalid clean preemption')
        for key in ('train_rows','rank_batches','per_rank_accumulation'):
            integer(row['counts'].get(key), key,1)
    return first


def check_epochs(run, previous, records, completed, identity, fingerprint):
    from transgenic.training import b6_runtime as rt
    require(type(previous) is list, 'invalid prior epoch events')
    rows = records[0]['result']['history']
    for record in records:
        require(record['result']['history'] == rows, 'epoch history rank disagreement')
    observations = records[0]['epoch_end']
    require(len(observations) == len(rows), 'missing actual epoch_end observation')
    events = list(previous)
    for row, observation in zip(rows, observations):
        require(type(row) is dict and set(row) == {'epoch','global_step','data_order_sha256','train','de_novo','evidence'}, 'trainer epoch history fields')
        integer(row['epoch'],'epoch',1)
        integer(row['global_step'],'epoch global step',1)
        hash_value(row['data_order_sha256'],'epoch order')
        require(observation.get('epoch_end') == row['epoch'] and observation.get('arm') == 'C2' and
                observation.get('data_order_first1000_sha256') == row['data_order_sha256'], 'epoch_end identity')
        for key in ('train','de_novo','evidence'):
            require(type(row[key]) in (int,float) and math.isfinite(row[key]) and observation.get(key) == row[key], 'nonfinite or mismatched epoch loss')
        events.append(dict(row, epoch_end=observation))
    require([e.get('epoch') for e in events] == list(range(1, completed + 1)), 'missing/extra completed epoch evaluation')
    verified = []
    for index, event in enumerate(events):
        directory = beneath(run, 'epoch_%02d' % event['epoch'])
        meta, history = checkpoint_common(directory, identity, fingerprint)
        evaluation = read_json(directory / 'eval.json')
        require(meta['epoch'] == event['epoch'] and meta['step'] == 0 and meta['global_step'] == event['global_step'], 'epoch checkpoint position')
        for loss, field in [('train','train_loss'),('de_novo','eval_loss'),('evidence','eval_evidence_loss')]:
            value = evaluation.get(field)
            require(type(value) in (int,float) and math.isfinite(value) and value == event[loss], 'invalid committed epoch loss')
        require(evaluation.get('epoch') == event['epoch'] and evaluation.get('global_step') == event['global_step'], 'epoch eval position')
        for entry in history.entries:
            position = entry['position']
            if position['phase'] == 'eval':
                require(position['epoch'] < event['epoch'] and position['training'] is False and
                        position['grad_enabled'] is False and position['view'] in ('de_novo','evidence'), 'invalid history eval context')
        hashes = dict(meta_sha256=sha(directory / 'meta.json'), eval_sha256=sha(directory / 'eval.json'),
                      identity_sha256=sha(directory / rt.CHECKPOINT_IDENTITY), history_sha256=meta['compile_history_sha256'])
        if index < len(previous):
            require(all(event.get(k) == v for k,v in hashes.items()), 'prior epoch evidence changed')
        verified.append(dict(event, **hashes))
    return verified


def preflight(work, request_path, request_sha256, job_id):
    from transgenic.training import b6_runtime as rt
    request = controls(work, request_path, request_sha256)
    output = execution_dir(work, request, job_id)
    require(output.is_dir() and all(item.name == 'preflight.log' and item.is_file() and not item.is_symlink()
                                   for item in output.iterdir()), 'execution must be newly created and empty')
    run = beneath(work, request['run_dir'])
    require(not os.path.lexists(run / 'TRAINING_DONE'), 'stale DONE at link start')
    selected = rt.CheckpointLayout(str(run)).resume_dir('auto')
    require(selected is not None and Path(selected) == run / request['start_checkpoint']['relative_dir'], 'selected checkpoint mismatch')
    identity = read_json(Path(selected) / rt.CHECKPOINT_IDENTITY)
    rt.check_run_config_b6(read_json(run / 'run_config_b6.json'), identity)
    state_contract = {}
    checkpoint = check_checkpoint(selected, identity, request['expected_runtime_fingerprint'], capture_contract=state_contract)
    history = read_json(Path(selected) / 'compile_history.json')
    history_prefix = dict(entries=[digest(entry) for entry in history['entries']], bootstrap=digest(history['bootstrap']))
    expected = request['start_checkpoint']
    require(checkpoint['position'] == {k:expected[k] for k in POSITION} and
            all(checkpoint[k] == expected[k] for k in ('relative_dir','meta_sha256','identity_sha256','history_sha256','canonical_checkpoint_sha256')), 'start checkpoint pin mismatch')
    # Full count consistency is checked against the actual prepared loader in each rank.
    # At this boundary bind existing draw bytes, ensuring no mutation goes unrecorded.
    draw_files = {}
    for rank in range(4):
        path = beneath(run,'draw_log/rank%02d.tsv' % rank)
        require(path.is_file(), 'missing start draw log')
        draw_files[str(rank)] = dict(size=path.stat().st_size,sha256=sha(path))
    exclusive_json(output / 'preflight.json', dict(version=1,job_id=job_id,request_sha256=request_sha256,
                   checkpoint=checkpoint,identity=identity,draw_files=draw_files,
                   state_contract=state_contract,history_prefix=history_prefix))
    return request


def observe_training(trainer, request, work, job_id, request_sha256):
    """Call every original; hooks only record successful results and actual loader sizes."""
    from accelerate import Accelerator
    originals = []
    record = dict(rank=integer(int(os.environ['RANK']),'rank'), job_id=job_id,request_sha256=request_sha256,epoch_end=[])
    require(record['rank'] < 4 and os.environ.get('WORLD_SIZE') == '4', 'allocation rank topology')
    def replace(owner, name, function):
        originals.append((owner,name,getattr(owner,name)))
        setattr(owner,name,function)
    verify = trainer.verify_training_inputs
    def verified(*args, **kwargs):
        value = verify(*args, **kwargs)
        require('input_verification' not in record, 'verifier called unexpectedly twice')
        record['input_verification'] = dict(job_id=job_id,bindings_sha256=request['bindings_sha256'],record=json.loads(canonical(value)))
        return value
    replace(trainer,'verify_training_inputs',verified)
    build_identity = trainer.rt.build_run_identity
    def identity(*args, **kwargs):
        value = build_identity(*args,**kwargs)
        record['identity'] = json.loads(canonical(value))
        return value
    replace(trainer.rt,'build_run_identity',identity)
    fingerprint = trainer.replay.runtime_fingerprint
    def runtime(*args,**kwargs):
        value = fingerprint(*args,**kwargs)
        require(value == request['expected_runtime_fingerprint'], 'actual runtime fingerprint differs from proof')
        record['runtime_fingerprint'] = json.loads(canonical(value))
        return value
    replace(trainer.replay,'runtime_fingerprint',runtime)
    run_b6 = trainer.run_b6
    prepare = Accelerator.prepare
    def run(spec, model, train_ds, eval_ds, **kwargs):
        rows = len(kwargs['train_sampler'])
        accumulation = trainer.rt.accumulation_steps(spec.cfg,spec.batch_size)
        require(spec.batch_size == 1 and accumulation % 4 == 0, 'production batch/accumulation')
        record['counts'] = dict(train_rows=rows,per_rank_accumulation=accumulation//4)
        old_log = spec.log
        def log(message):
            old_log(message)
            try: payload = json.loads(message)
            except (TypeError,ValueError): return
            if isinstance(payload,dict) and 'epoch_end' in payload:
                record['epoch_end'].append(payload)
        spec.log = log
        def prepared(accelerator,*args,**kw):
            result = prepare(accelerator,*args,**kw)
            require(len(args) == 5 and accelerator.num_processes == 4, 'unexpected training prepare topology')
            record['counts']['rank_batches'] = len(result[2])
            require(len(result[2]) == math.ceil(rows/4), 'unexpected prepared sampler batching')
            check_draws(beneath(work,request['run_dir']),request['start_checkpoint'],record['counts'])
            return result
        replace(Accelerator,'prepare',prepared)
        try: return run_b6(spec,model,train_ds,eval_ds,**kwargs)
        finally: spec.log = old_log
    replace(trainer,'run_b6',run)
    try:
        result = trainer.main(['--config',str(beneath(Path('/work/repo') if str(work) == '/work' else beneath(work,request['source_dir']),request['config_relative'])),
            '--arm','C2','--seed','123','--bindings',str(beneath(work,request['bindings'])),
            '--output-dir',str(beneath(work,request['run_dir'])),'--resume','auto','--stop-after-updates','4099',
            '--save-every-n-steps','200','--batch-size','1','--mixed-precision','bf16','--deterministic'])
        record['result'] = result
        require('input_verification' in record and 'identity' in record and 'runtime_fingerprint' in record and
                'rank_batches' in record.get('counts',{}), 'trainer omitted required observations')
        return record
    finally:
        for owner,name,original in reversed(originals): setattr(owner,name,original)


def run_rank(work, request_path, request_sha256, job_id):
    request = controls(work,request_path,request_sha256)
    output = execution_dir(work,request,job_id)
    first = read_json(output/'preflight.json')
    require(first.get('request_sha256') == request_sha256 and first.get('job_id') == job_id, 'preflight binding mismatch')
    source = Path('/work/repo') if str(work) == '/work' else beneath(work,request['source_dir'])
    spec = importlib.util.spec_from_file_location('c2_production_actual_trainer',source/'train/train_HyenaTransgenic_B6.py')
    trainer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=trainer
    spec.loader.exec_module(trainer)
    record = observe_training(trainer,request,work,job_id,request_sha256)
    controls(work,request_path,request_sha256)
    exclusive_json(output/('rank%d.json'%record['rank']),record)


def postflight(work, request_path, request_sha256, job_id, trainer_exit):
    from transgenic.training import b6_runtime as rt
    require(type(trainer_exit) is int and trainer_exit == 0, 'nonzero trainer/launcher exit')
    request = controls(work,request_path,request_sha256)
    output = execution_dir(work,request,job_id)
    first = read_json(output/'preflight.json')
    require(first.get('request_sha256') == request_sha256 and first.get('job_id') == job_id, 'stale preflight')
    records = [read_json(output/('rank%d.json'%r)) for r in range(4)]
    record = validate_rank_records(records,request,job_id,request_sha256)
    identity = record['identity']
    rt.check_run_config_b6(first['identity'],identity)
    run = beneath(work,request['run_dir'])
    rt.check_run_config_b6(read_json(run/'run_config_b6.json'),identity)
    binding_identity = rt.identity_from_bindings(record['input_verification']['record'])
    require(all(identity.get(k) == v for k,v in binding_identity.items()), 'verified bindings do not match training identity')
    done = record['result'].get('done') is True
    if done:
        checkpoint = check_final(run,identity,request['expected_runtime_fingerprint'],expected_model=first['state_contract']['model'])
    else:
        require(not os.path.lexists(run/'TRAINING_DONE'),'DONE on preempted link')
        selected = rt.CheckpointLayout(str(run)).resume_dir('auto')
        require(selected is not None and re.fullmatch('latest_state|epoch_[0-9]{2,}',Path(selected).name), 'unsafe selected resume checkpoint')
        checkpoint = check_checkpoint(selected,identity,request['expected_runtime_fingerprint'],expected_contract=first['state_contract'])
    history = read_json(beneath(run, checkpoint['relative_dir'] + '/compile_history.json'))
    prefix = first['history_prefix']
    require(digest(history['bootstrap']) == prefix['bootstrap'] and
            [digest(entry) for entry in history['entries'][:len(prefix['entries'])]] == prefix['entries'],
            'compiler history prefix or bootstrap changed')
    if done:
        require(checkpoint['position']['step'] > 0 and checkpoint['position']['epoch'] ==
                (4099 * record['counts']['per_rank_accumulation'] - 1) // record['counts']['rank_batches'],
                'final completed epoch count mismatch')
    require(checkpoint['position']['global_step'] == record['result']['global_step'], 'result/checkpoint step mismatch')
    draws = check_draws(run,checkpoint['position'],record['counts'])
    # The unchanged old prefix is bound even when the log has grown during this link.
    for rank in range(4):
        path=beneath(run,'draw_log/rank%02d.tsv'%rank)
        prior=first['draw_files'][str(rank)]
        h=hashlib.sha256()
        with open(path,'rb') as handle:
            left=prior['size']
            while left:
                chunk=handle.read(min(left,8<<20));require(bool(chunk),'short prior draw prefix')
                h.update(chunk);left-=len(chunk)
        require(h.hexdigest()==prior['sha256'],'prior draw prefix changed')
    events = check_epochs(run,request['completed_epoch_events'],records,checkpoint['position']['epoch'],identity,request['expected_runtime_fingerprint'])
    final = None if not done else dict(relative_dir=checkpoint['relative_dir'],model_sha256=checkpoint['files']['model.safetensors']['sha256'],
                                      done_sha256=checkpoint['done_sha256'],expected_completed_epochs=len(events),epoch_events=events)
    receipt = dict(version=1,kind='c2_production_link_ready',operation_id=request['operation_id'],token=request['token'],
        link_index=request['link_index'],job_id=job_id,request_sha256=request_sha256,source_commit=request['source_commit'],
        predecessor_sha256=request['predecessor']['sha256'],start_global_step=request['start_checkpoint']['global_step'],
        result=record['result'],profile=request['profile'],runtime_fingerprint_sha256=digest(request['expected_runtime_fingerprint']),
        input_verification=dict(job_id=job_id,ranks=list(range(4)),bindings_sha256=request['bindings_sha256'],
                                record_sha256=digest(record['input_verification']['record']),identity_sha256=digest(identity)),
        checkpoint=checkpoint,epoch_events=events,final=final,counts=record['counts'],draw_logs=draws,
        source_snapshot_git=request['source_snapshot_git'],source_manifest_sha256=request['source_manifest_sha256'],
        helper_manifest_sha256=request['helper_manifest_sha256'],config_sha256=request['config_sha256'])
    controls(work,request_path,request_sha256)
    exclusive_json(output/'link_receipt.json',receipt)
    print('C2_PRODUCTION_FINAL_READY' if done else 'LINK_CHECKPOINT_READY',job_id,record['result']['global_step'],flush=True)
    return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['paths','preflight','run','postflight'])
    parser.add_argument('--work',required=True)
    parser.add_argument('--request',required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--job-id',default=os.environ.get('SLURM_JOB_ID'))
    parser.add_argument('--trainer-exit',type=int)
    args=parser.parse_args(argv)
    if args.command=='paths':
        request=pinned_request(args.work,args.request,args.request_sha256)
        for key in ('source_dir','helper_dir','run_dir'):
            print(beneath(args.work,request[key]))
        print(execution_dir(args.work,request,args.job_id))
        return
    require(not sys.flags.optimize,'optimized Python cannot verify checkpoints')
    require(bool(os.environ.get('SLURM_JOB_ID')) and args.job_id==os.environ['SLURM_JOB_ID'],'allocation required')
    if args.command=='preflight':preflight(args.work,args.request,args.request_sha256,args.job_id)
    elif args.command=='run':run_rank(args.work,args.request,args.request_sha256,args.job_id)
    else:postflight(args.work,args.request,args.request_sha256,args.job_id,args.trainer_exit)


if __name__=='__main__':main()
