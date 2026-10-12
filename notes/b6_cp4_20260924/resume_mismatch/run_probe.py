"""Diagnosis only: compare eager and unsplit compiled DDP; not a production gate."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--case', choices=['eager', 'unsplit'], required=True)
parser.add_argument('--out', required=True)
args = parser.parse_args()
if int(os.environ.get('WORLD_SIZE', '0')) != 2:
    raise SystemExit('diagnosis requires exactly 2 ranks')
import torch
if args.case == 'unsplit':
    torch._dynamo.config.optimize_ddp = False
root = Path('/work/repo')
backend = 'none' if args.case == 'eager' else 'inductor'
print(json.dumps(dict(diagnostic_only=True, case=args.case, rank=os.environ['RANK'],
                      optimize_ddp=torch._dynamo.config.optimize_ddp, backend=backend)), flush=True)
sys.path.insert(0, str(root / 'revision/scripts/b6'))
sys.argv = [str(root / 'revision/scripts/b6/integration_4rank.py'),
            '--model', 'real', '--config', 'configs/b6_400m_win_v3.json',
            '--data', 'wp3-fixtures', '--accumulation', '4', '--n-pilot', '3', '--stop-at', '1',
            '--backend', backend, '--mixed-precision', 'bf16', '--gen-tokens', '12', '--out', args.out]
runpy.run_path(sys.argv[0], run_name='__main__')
