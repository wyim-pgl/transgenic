#!/bin/bash
# Submit from a writable directory after Delta CPU allocation activation:
# sbatch --account="$DELTA_CPU_ACCOUNT" delta_cpu_preflight.sbatch
# No account name is guessed and no GPU resources are requested.
#SBATCH --job-name=transgenic-cpu-preflight
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --constraint=work
#SBATCH --no-requeue
#SBATCH --output=delta-cpu-preflight-%j.out
set -euo pipefail
umask 077
if [[ ! ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]]; then
    echo 'Run this metadata-only preflight through sbatch on Delta CPU.' >&2
    exit 64
fi
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python3 - <<'PY'
import hashlib
import json
import os
import platform
from pathlib import Path
import shlex
import subprocess
import sys

if sys.version_info < (3, 6):
    raise RuntimeError('Python3.6 or newer is required by the existing archive tools')
job = os.environ['SLURM_JOB_ID']
details = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], universal_newlines=True)
fields = dict(item.split('=', 1) for item in shlex.split(details) if '=' in item)
if fields.get('Partition') != 'cpu':
    raise RuntimeError('This preflight must use the normal Delta cpu partition')
for key in ('ReqTRES', 'AllocTRES'):
    if key not in fields:
        raise RuntimeError('Scheduler resource information is missing: ' + key)
    for item in fields[key].split(','):
        name, value = item.split('=', 1)
        if name.startswith('gres/gpu') and int(value) != 0:
            raise RuntimeError('A GPU was allocated to the zeroGPU preflight')

task_work = Path('/work/nvme/bilv/wcyim/transgenic')
task_archive = Path('/work/hdd/bilv/wcyim/transgenic/checkpoint_archive')
helper = task_work / 'ops/checkpoint_archive/3b4eb0d'
paths = []
for path in (task_work, task_archive, helper):
    if not path.is_dir() or not os.access(str(path), os.R_OK | os.X_OK):
        raise RuntimeError('Existing project directory is not readable/searchable: ' + str(path))
    paths.append(dict(path=str(path), uid=path.stat().st_uid, gid=path.stat().st_gid,
                      readable=True, writable=os.access(str(path), os.W_OK)))

expected = {
    'revision/scripts/b6/checkpoint_archive.py': '0c79dfddfec33695a60255b6dea55c459712241670924ed30366014c30e4fbf5',
    'revision/scripts/b6/checkpoint_archive_cli.py': '1a733e111d38bd5ae4da2e3370d7a9f0b3058a2a7852e797efcfbb1620b487d9',
    'revision/scripts/b6/storage_preflight.py': '03832311be8dfa8652383a34856d02b68a1b15f7ed1052dd8bdc7d4de28a1fb4',
}
source = helper / 'SOURCE_COMMIT'
if source.stat().st_size > 1024 or source.read_text().strip() != '3b4eb0dfe0e83463775ce8e1ab05fa2f62d1e20b':
    raise RuntimeError('Archive source revision differs from the reviewed version')
for relative, digest in expected.items():
    path = helper / relative
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise RuntimeError('Unsafe or unexpectedly large helper: ' + relative)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError('Archive helper digest differs: ' + relative)

print(json.dumps(dict(
    kind='delta_cpu_metadata_preflight', job_id=job,
    hostname=platform.node(), architecture=platform.machine(), python=sys.version,
    uid=os.getuid(), account=fields.get('Account'), partition=fields['Partition'],
    requested_tres=fields['ReqTRES'], allocated_tres=fields['AllocTRES'],
    paths=paths, reviewed_helper_hashes=expected,
    scope='metadata and small source files only; no checkpoint/data copy, payload hashing, deletion or write-access test',
), indent=2, sort_keys=True))
PY
