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
# Only the reviewed prepare_controller.py argv is authorized: explicit distinct
# --output/--error, --open-mode=append, fresh private single-use namespace.
# Logs are reserved before Slurm starts this script. Never reuse or retry argv.
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
# Forward only the first interruption so repeated TERM cannot tear down the
# worker's scoped cleanup. Independent USR1 at 40s stops interruptible cleanup
# or postchecks, leaving 20s of the Slurm grace for the launcher receipt.
watchdog=
interrupted=0
interrupt() {
  (( interrupted == 0 )) || return 0
  interrupted=1
  kill -TERM "$pid" 2>/dev/null || true
  ( sleep 40; kill -USR1 "$pid" 2>/dev/null || true ) &
  watchdog=$!
}
trap interrupt TERM INT
set +e
while true; do
  wait "$pid"
  rc=$?
  kill -0 "$pid" 2>/dev/null || break
done
if [[ -n $watchdog ]]; then kill "$watchdog" 2>/dev/null || true; fi
# A graceful worker exit does not turn an interrupted allocation into success.
if (( interrupted != 0 && rc == 0 )); then rc=1; fi
exit "$rc"
