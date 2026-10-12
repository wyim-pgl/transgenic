#!/bin/bash
#SBATCH --job-name=mvp50-input-audit
# Controller supplies approved CPU partition/account; observed default cpu-s3-sponsored-0.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
# Controller supplies --output/--error in an existing fresh staging directory.
# No submit/install/retry logic. Do not reuse a failed namespace.
set -euo pipefail
: "${SLURM_JOB_ID:?compute allocation required}"
: "${PANEL_RELEASE:?absolute reviewed release directory required}"
PANEL_OUTPUT="$PANEL_RELEASE/outputs/job-$SLURM_JOB_ID"
: "${PANEL_CPU_SITE:?reviewed extracted DuckDB1.5.5 cp310 site required}"
[[ "$PANEL_RELEASE" = /* && "$PANEL_OUTPUT" = /* && "$PANEL_CPU_SITE" = /* ]]
[[ ! -e "$PANEL_OUTPUT" && ! -L "$PANEL_OUTPUT" ]]
[[ "${SLURM_CPUS_PER_TASK}" -le 4 ]]
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONHOME PYTHONPATH
# Existing runtime was inspected via METADATA only, not certified as executable.
# Bind original databases/reference read-only; no --nv, GPU calls, cache or checkpoint.
SIF=/data/gpfs/assoc/pgl/singularity/micromamba.sif
PYTHON=/data/gpfs/home/wyim/micromamba/envs/signalp6/bin/python
# Source and dependencies must remain at this canonical path for later verification.
# Output parent is the sole writable project bind. It must be dedicated to this job.
PARENT=$(dirname "$PANEL_OUTPUT")
[[ -d "$PARENT" && "$PARENT" = "$PANEL_RELEASE/outputs" ]]
[[ "$PANEL_CPU_SITE" = "$PANEL_RELEASE/cpu_site" ]]
[[ "$PANEL_OUTPUT" = "$PANEL_RELEASE/outputs/job-$SLURM_JOB_ID" ]]
exec timeout --signal=TERM --kill-after=20s 3500s \
 /apps/singularity/3.6.1/bin/singularity exec --cleanenv \
 -B /data/gpfs:/data/gpfs:ro -B "$PARENT:$PARENT" \
 "$SIF" env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$PANEL_CPU_SITE" MVP_CPU_SITE="$PANEL_CPU_SITE" \
 SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_CPUS_PER_TASK="$SLURM_CPUS_PER_TASK" \
 "$PYTHON" -B -s "$PANEL_RELEASE/run_panel_cpu.py" \
 --root "$PANEL_RELEASE/source" --out "$PANEL_OUTPUT"
