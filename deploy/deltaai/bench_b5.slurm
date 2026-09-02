#!/bin/bash
#SBATCH --job-name=b5-bench
#SBATCH --account=CHANGE_ME            # ACCESS project on DeltaAI/Delta (e.g. bbxx-dtai-gh or bbxx-delta-gpu)
#SBATCH --partition=CHANGE_ME          # DeltaAI: ghx4 ; Delta: gpuA100x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=16
#SBATCH --mem=120g
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
# Issue #18: measure throughput of the frozen 400M recipe on this hardware.
set -euo pipefail
module load apptainer
WORK=$SCRATCH/transgenic; SIF=$SCRATCH/containers/transgenic-ngc.sif
GPUS=${GPUS:-1}; STEPS=${STEPS:-300}; DB=${DB:-/work/db/b5_smoke.duckdb}
export NCCL_DEBUG=WARN OMP_NUM_THREADS=8
apptainer exec --nv --bind "$WORK":/work "$SIF" bash -lc "
  cd /work/repo && pip install -q -e . --no-deps &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
    --config configs/b5_400m_ctx_v2.json --db $DB --seed 123 --benchmark-steps $STEPS --output-dir /work/runs/bench-g$GPUS
"
# Expected output: tokens/s, s/step, peak GB, hours/epoch (rows in DB / rows per step / 3600); paste into #18.
