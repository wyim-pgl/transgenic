"""Two real CPU ranks; fault injection only, never GPU readiness evidence."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'revision/scripts/mvp'))
from accelerate import Accelerator
import torch
from torch._inductor import metrics
from real_probes import collective_batches, collective_compiler_check

acc = Accelerator(cpu=True)
case, output = sys.argv[1], Path(sys.argv[2])
try:
    if case == 'loader':
        class Loader:
            def __iter__(self):
                if acc.process_index == 1:
                    raise ValueError('injected rank1 loader failure')
                return iter(range(24))
        collective_batches(acc, Loader())
    elif case == 'compiler':
        torch._dynamo.config.suppress_errors = False
        metrics.generated_kernel_count = 0 if acc.process_index == 1 else 1
        collective_compiler_check(acc, 0)
    else:
        raise ValueError(case)
except (ValueError, RuntimeError) as error:
    (output / f'rank{acc.process_index}.json').write_text(json.dumps({'error': str(error)}))
    acc.wait_for_everyone()
    acc.end_training()
    sys.exit(2)
raise RuntimeError('fault did not reject all ranks')
