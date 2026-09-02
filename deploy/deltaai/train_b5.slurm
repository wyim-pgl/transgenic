#!/bin/bash
#SBATCH --job-name=b5-seed
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=16
#SBATCH --mem=200g
#SBATCH --time=48:00:00
#SBATCH --output=%x-%j.out
# One B5 seed per job: sbatch --export=ALL,SEED=456 deploy/deltaai/train_b5.slurm
set -euo pipefail
module load apptainer
WORK=$SCRATCH/transgenic; SIF=$SCRATCH/containers/transgenic-ngc.sif
SEED=${SEED:?set SEED=123|456|789}; GPUS=${GPUS:-1}; RUN=/work/runs/seed$SEED
export NCCL_DEBUG=WARN OMP_NUM_THREADS=8 WANDB_MODE=${WANDB_MODE:-offline}
apptainer exec --nv --bind "$WORK":/work "$SIF" bash -lc "
  cd /work/repo && pip install -q -e . --no-deps &&
  sha256sum -c /work/db/b5_v1.duckdb.sha256 &&
  mkdir -p $RUN &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
    --config configs/b5_400m_v1.json --db /work/db/b5_v1.duckdb --seed $SEED --output-dir $RUN --resume auto \
    2>> $RUN/train.err
"
# After each epoch: python revision/scripts/60_epoch_budget.py --log $RUN/train.err --patience 3
