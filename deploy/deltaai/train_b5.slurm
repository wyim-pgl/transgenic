#!/bin/bash
#SBATCH --job-name=b5-seed
#SBATCH --account=CHANGE_ME
#SBATCH --partition=CHANGE_ME
#SBATCH --nodes=1
#SBATCH --gpus-per-node=${GPUS:-1}
#SBATCH --cpus-per-task=16
#SBATCH --mem=200g
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=%x-%j.out
# One B5 seed per job chain (protocol A28): sbatch --export=ALL,SEED=456 deploy/deltaai/train_b5.slurm
# The job saves a resumable latest_state every N optimizer steps and on SIGUSR1 (sent 15 min before the wall-clock
# limit), then resubmits itself with --dependency=afterany until TRAINING_DONE (or TRAINING_FAILED) exists.
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
  trap 'wait $SYNC_PID 2>/dev/null' EXIT
fi
CHAIN_MAX=${CHAIN_MAX:-8}; CHAIN_N=${CHAIN_N:-1}
apptainer exec --nv --bind "$WORK":/work "$SIF" bash -lc "
  cd /work/repo && pip install -q -e . --no-deps &&
  sha256sum -c /work/db/b5_v1.duckdb.sha256 &&
  mkdir -p $RUN &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
    --config configs/b5_400m_win_v3.json --db /work/db/b5_v1.duckdb --seed $SEED --output-dir $RUN --resume auto \
    --save-every-n-steps ${SAVE_EVERY:-200} 2>> $RUN/train.err
" &
TRAIN_PID=$!
# forward the pre-limit signal to the training process so it saves latest_state and exits cleanly
trap 'echo "USR1: forwarding to trainer"; pkill -USR1 -P $TRAIN_PID 2>/dev/null; kill -USR1 $TRAIN_PID 2>/dev/null' USR1
wait $TRAIN_PID; TRAIN_RC=$?
# chain: resubmit unless finished, failed, or the chain limit is reached
if [ -f "$RUN_HOST/TRAINING_DONE" ]; then echo "training finished"
elif [ -f "$RUN_HOST/TRAINING_FAILED" ] || [ "$TRAIN_RC" -ne 0 ]; then echo "training failed (rc=$TRAIN_RC); not resubmitting"; touch "$RUN_HOST/TRAINING_FAILED"
elif [ "$CHAIN_N" -ge "$CHAIN_MAX" ]; then echo "chain limit $CHAIN_MAX reached; not resubmitting"
else
  echo "resubmitting chain job $((CHAIN_N + 1))"
  sbatch --dependency=afterany:$SLURM_JOB_ID --export=ALL,SEED=$SEED,GPUS=$GPUS,CHAIN_N=$((CHAIN_N + 1)),CHAIN_MAX=$CHAIN_MAX "$WORK/repo/deploy/deltaai/train_b5.slurm"
fi

# After each epoch: python revision/scripts/60_epoch_budget.py --log $RUN/train.err --patience 3
