#!/bin/bash
#SBATCH --job-name=mvp-post-p1
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=00:30:00
set -euo pipefail
umask 077
# Controller supplies absolute exclusive release directory, current exclusions,
# output/error paths and a hash-reviewed container argv. This file never sbatch's.
: "${FOLLOWUP_RELEASE:?absolute controller release directory required}"
cd "$FOLLOWUP_RELEASE"
[[ ${SLURM_JOB_NUM_NODES:?} == 1 && ${SLURM_GPUS_ON_NODE:?} == 4 ]]
(set -o noclobber; printf '%s\n' "${SLURM_JOB_ID:?}" > STARTED)
/usr/bin/sha256sum -c release.sha256
# container.argv.json must run followup.py --execute; host mounts are reviewed
# against ACTUAL S0 job, never inferred from a predicted Slurm job directory.
# Preserve /scratch P1 cache and all earlier /followup paths at original names.
# Full 1800s allocation is reserved, including these checks and teardown.
export ALLOCATION_RESERVED_SECONDS=1800
/usr/bin/timeout --signal=TERM --kill-after=20s 1700s /usr/bin/python3 -c '
import json, os
from pathlib import Path
v=json.loads(Path("container.argv.json").read_text())
if not isinstance(v,list) or not v or not all(isinstance(x,str) and x for x in v) or not Path(v[0]).is_absolute():
    raise ValueError("reviewed absolute host argv required")
os.execv(v[0],v)
'
