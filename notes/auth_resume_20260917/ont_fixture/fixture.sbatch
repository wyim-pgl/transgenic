#!/bin/bash
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
# Controller supplies log paths outside the fresh OUT, under the same hard quota.
set -euo pipefail
: "${RELEASE:?reviewed immutable fixture bundle required}"
: "${OUT:?fresh output beneath dedicated hard-quota scratch required}"
: "${MANIFEST_SHA256:?independently retained manifest identity required}"
: "${ONT_FIXTURE_APPROVAL:?controller-authored approval required}"
: "${ONT_FIXTURE_APPROVAL_SHA256:?independently retained approval identity required}"
cd "$RELEASE"
test "$(sha256sum MANIFEST.sha256 | cut -d ' ' -f 1)" = "$MANIFEST_SHA256"
sha256sum --strict -c MANIFEST.sha256
# Hard filesystem quota (256 MiB including Slurm logs) must be independently verified.
# --tmp is NOT a hard disk limit. This template intentionally does not pretend it is.
# Controller must verify active job cgroup memory containment before approval.
ulimit -f 8192
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python3.11
scontrol show job -o "$SLURM_JOB_ID"
srun --ntasks=1 --cpu-bind=threads /usr/bin/timeout --signal=TERM --kill-after=10s 300s \
  "$PY" -I -B "$RELEASE/fixture.py" run "$RELEASE/package" "$OUT"
