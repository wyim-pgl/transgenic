#!/bin/bash
#SBATCH --job-name=est-real-sample
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --signal=B:TERM@60
# Proposal only. Controller supplies account/partition/log path and deploys after review.
# Usage after deployment: sbatch [site options] compute_proposal.sbatch CODE_ROOT MANIFEST MANIFEST_SHA NEW_OUTPUT_DIR
set -euo pipefail
: "${SLURM_JOB_ID:?compute allocation required}"
code_root=$1
manifest=$2
manifest_sha=$3
out=$4
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$code_root/revision/scripts"
exec /data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11 \
  -m evidence.est_real_sample --manifest "$manifest" --sha "$manifest_sha" --out "$out"
