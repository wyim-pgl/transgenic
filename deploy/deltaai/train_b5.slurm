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
SEED=${SEED:?set SEED=123|456|789}; GPUS=${GPUS:-1}; RUN=/work/runs/seed$SEED; RUN_HOST=$WORK/runs/seed$SEED
export NCCL_DEBUG=WARN OMP_NUM_THREADS=8 WANDB_MODE=${WANDB_MODE:-offline}
# Per-epoch off-site copy (deploy/deltaai/sync_watch.sh). SYNC_DEST e.g. wyim@pronghorn:/data/gpfs/assoc/pgl/data/Transgenic/runs/seed$SEED
# Needs a passphrase-less key on this host (RSYNC_SSH="ssh -i ~/.ssh/id_ed25519_sync") authorised on the destination.
mkdir -p "$RUN_HOST"
if [ -n "${SYNC_DEST:-}" ]; then
  KEEP_LOCAL=${KEEP_LOCAL:-3} SYNC_OPTIMIZER=${SYNC_OPTIMIZER:-0} INTERVAL=${SYNC_INTERVAL:-300}     bash "$WORK/repo/deploy/deltaai/sync_watch.sh" "$RUN_HOST" "$SYNC_DEST" > "$RUN_HOST/sync.log" 2>&1 &
  SYNC_PID=$!
  # the trainer itself writes TRAINING_DONE on a clean finish; on failure we only stop the watcher (no marker)
  trap 'touch "$RUN_HOST/TRAINING_FAILED"; kill $SYNC_PID 2>/dev/null' ERR
  trap 'wait $SYNC_PID 2>/dev/null' EXIT
fi
apptainer exec --nv --bind "$WORK":/work "$SIF" bash -lc "
  cd /work/repo && pip install -q -e . --no-deps &&
  sha256sum -c /work/db/b5_v1.duckdb.sha256 &&
  mkdir -p $RUN &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
    --config configs/b5_400m_ctx_v2.json --db /work/db/b5_v1.duckdb --seed $SEED --output-dir $RUN --resume auto \
    2>> $RUN/train.err
"
[ -f "$RUN_HOST/TRAINING_DONE" ] || echo "WARNING: trainer exited without TRAINING_DONE" >&2
# After each epoch: python revision/scripts/60_epoch_budget.py --log $RUN/train.err --patience 3
