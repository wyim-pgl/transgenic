"""Read-only comparison of the two preserved CP4 runs. CPU, one thread, mapped tensors."""
from pathlib import Path
import hashlib
import json
import torch
from safetensors import safe_open

root = Path('/work/runs/b6_int4/run_3207331')
torch.set_num_threads(1)
for sub in ['epoch_01', 'step_00003']:
    a = root / 'straight' / sub / 'model.safetensors'
    b = root / 'resumed' / sub / 'model.safetensors'
    diff, equal, maximum = [], 0, 0
    with safe_open(a, framework='pt', device='cpu') as fa, safe_open(b, framework='pt', device='cpu') as fb:
        print('KEYS', sub, fa.keys() == fb.keys(), len(fa.keys()))
        for key in fa.keys():
            x, y = fa.get_tensor(key), fb.get_tensor(key)
            if torch.equal(x, y):
                equal += 1
                continue
            count, delta = 0, 0.0
            for xx, yy in zip(x.reshape(-1).split(1000000), y.reshape(-1).split(1000000)):
                count += int((xx != yy).sum())
                delta = max(delta, float((xx.float() - yy.float()).abs().max()))
            diff.append(dict(key=key, count=count, numel=x.numel(), max_abs=delta))
            maximum = max(maximum, delta)
    print('MODEL_DIFF', sub, json.dumps(dict(equal=equal, different=len(diff), worst=maximum, examples=diff[:12])), flush=True)


def digest(value):
    if isinstance(value, torch.Tensor):
        return hashlib.sha256(value.numpy().tobytes()).hexdigest()
    if isinstance(value, (list, tuple)):
        return [digest(v) for v in value]
    return hashlib.sha256(repr(value).encode()).hexdigest()


def compare(x, y, prefix=''):
    if isinstance(x, dict):
        assert x.keys() == y.keys(), prefix
        return sum(compare(x[k], y[k], prefix + '/' + str(k)) for k in x)
    if isinstance(x, torch.Tensor):
        if not torch.equal(x, y):
            if '.alpha' in prefix or 'evidence_proj' in prefix:
                print('GRAD_DIFF_EXAMPLE', rank, prefix, float((x - y).abs().max()))
            return 1
        return 0
    if x != y:
        print('META_DIFF', prefix, x, y)
    return int(x != y)


for rank in range(2):
    for run in ['straight', 'resumed']:
        p = root / run / 'epoch_01' / 'accelerate_state' / f'random_states_{rank}.pkl'
        state = torch.load(p, map_location='cpu', weights_only=False)
        print('RNG', run, rank, {k: digest(v) for k, v in state.items()})
    a = torch.load(root / 'straight' / 'epoch_01' / f'pending_grads_rank{rank}.pt',
                   map_location='cpu', weights_only=False, mmap=True)
    b = torch.load(root / 'resumed' / 'epoch_01' / f'pending_grads_rank{rank}.pt',
                   map_location='cpu', weights_only=False, mmap=True)
    print('GRAD_PAYLOAD', rank, list(a)[:10])
    print('GRAD_DIFFERENT', rank, compare(a, b), flush=True)

for run in ['straight', 'resumed']:
    for name in ['data_order.jsonl', 'draw_log/rank00.tsv', 'epoch_01/meta.json', 'epoch_01/eval.json', 'step_00003/meta.json']:
        print('FILE', run, name, (root / run / name).read_text())
