#!/bin/bash
#SBATCH --job-name=est-real-sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60
#SBATCH --export=NIL
# Controller supplies account, partition and NEW exclusive stdout/stderr locations.
# Submit with --export=NIL (not NONE, which may reconstruct a login environment).
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?compute allocation required}"
[[ $# == 3 ]] || { printf 'Usage: sample.sbatch RELEASE LEDGER_SHA FRESH_OUT\n' >&2; exit 2; }
release=$1
sha=$2
out=$3
[[ $release == /* && $out == /* && $sha =~ ^[0-9a-f]{64}$ ]] || exit 2
[[ ! -e $out && ! -L $out ]] || { printf 'Output exists\n' >&2; exit 2; }
# env -i removes PYTHONHOME/PYTHONPATH, user-site switches, LD_PRELOAD,
# LD_LIBRARY_PATH, BASH_ENV, ENV, CONDA_* etc; no global environment activation.
/usr/bin/env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 \
  SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_JOB_ACCOUNT="${SLURM_JOB_ACCOUNT:-}" \
  SLURM_JOB_PARTITION="${SLURM_JOB_PARTITION:-}" SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-}" \
  SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-}" SLURM_MEM_PER_NODE="${SLURM_MEM_PER_NODE:-}" \
  /data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11 \
  -I -B "$release/run_compute.py" "$release" "$sha" "$out" &
pid=$!
# Do not exec: TERM is forwarded to the worker's own cleanup/report handler;
# interrupted wait is followed by another wait so the Slurm shell remains alive.
trap 'kill -TERM "$pid" 2>/dev/null || true' TERM INT
set +e
wait "$pid"
rc=$?
if kill -0 "$pid" 2>/dev/null; then wait "$pid"; rc=$?; fi
exit "$rc"
