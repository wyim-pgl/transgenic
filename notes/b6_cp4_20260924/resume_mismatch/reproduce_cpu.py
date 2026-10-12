"""Bounded CPU reproduction using the real trainer and existing tiny model fixture."""
import hashlib
import json
import sys
from pathlib import Path
import torch
from tests.test_b6_trainer import TrainerBase

backend = sys.argv[1]
backend = None if backend == 'none' else backend
torch.set_num_threads(1)
test = TrainerBase()
test.setUpClass()
test.setUp()
try:
    print('START', backend, test.tmp.name, flush=True)
    results = []
    for run, extra in [('straight', {}), ('resumed', {'stop_at_step': 1}), ('resumed', {'resume': 'auto'})]:
        torch._dynamo.reset()
        spec = test.spec('B6', run=run, n_pilot=3, accumulation_steps=4, compile_backend=backend, compile_warmup_steps=2,
                         deterministic=True, **extra)
        result = test.train_arm(spec, n_rows=6)
        print('LINK', run, extra, result, flush=True)
        results.append(result)
    root = Path(test.tmp.name)
    for part in ['epoch_01/model.safetensors', 'epoch_01/pending_grads_rank0.pt', 'step_00003/model.safetensors']:
        files = [root / run / part for run in ['straight', 'resumed']]
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
        print('COMPARE', part, hashes[0] == hashes[1], hashes, flush=True)
    print('LOGS', json.dumps(test.logs), flush=True)
finally:
    test.doCleanups()
    test.tearDownClass()
