#!/bin/bash -p
#SBATCH --job-name=ont_f4_fixture
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:06:00
#SBATCH --no-requeue
# CANDIDATE ONLY. No array/retry/GPU/canonical reads. Review before single submission.
# Controller supplies separate scheduler logs and reviews their logical/allocated bytes.
# Privileged Bash ignores BASH_ENV and exported functions before this first command.
set -euo pipefail
unset LD_PRELOAD LD_LIBRARY_PATH LD_AUDIT BASH_ENV ENV
export PATH=/usr/bin:/bin
for name in ${!PYTHON@}; do unset "$name"; done
unset POSIXLY_CORRECT
set +o posix
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
: "${RELEASE:?reviewed immutable fixture bundle required}"
: "${OUT:?fresh private scratch output required}"
: "${MANIFEST_SHA256:?independently retained manifest identity required}"
: "${ONT_FIXTURE_APPROVAL:?controller-authored approval required}"
: "${ONT_FIXTURE_APPROVAL_SHA256:?independently retained approval identity required}"
cd "$RELEASE"
test "$(sha256sum MANIFEST.sha256 | cut -d ' ' -f 1)" = "$MANIFEST_SHA256"
sha256sum --strict -c MANIFEST.sha256
# finite_fixture_soft_aggregate: 256 MiB logical post-run review, NOT a hard quota.
# Controller verifies job cgroups; outer timeout cleanup is best effort.
# Bash non-POSIX units are 1024 bytes. Python verifies both actual limits.
ulimit -f 4096
# PYTHONSTARTUP, PYTHONPATH and PYTHONHOME were removed with all PYTHON* above.
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python3.11
/cm/shared/apps/slurm/20.11.7/bin/scontrol show job -o "$SLURM_JOB_ID"
# Explicit startup environment before interpreter exec. Preserve all allocation variables.
launch_env=("PATH=/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin"
  "HOME=/tmp" "TMPDIR=/tmp" "LANG=C" "LC_ALL=C" "OMP_NUM_THREADS=4" "GOMAXPROCS=4"
  "PYTHONNOUSERSITE=1" "PYTHONDONTWRITEBYTECODE=1"
  "ONT_FIXTURE_APPROVAL=$ONT_FIXTURE_APPROVAL"
  "ONT_FIXTURE_APPROVAL_SHA256=$ONT_FIXTURE_APPROVAL_SHA256")
for name in ${!SLURM_@}; do launch_env+=("$name=${!name}"); done
/usr/bin/env -i "${launch_env[@]}" /cm/shared/apps/slurm/20.11.7/bin/srun --ntasks=1 --cpu-bind=threads /usr/bin/timeout --signal=TERM --kill-after=10s 300s \
  "$PY" -I -B "$RELEASE/fixture.py" run "$RELEASE/package" "$OUT"
