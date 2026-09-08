#!/bin/bash
#SBATCH --job-name=b5-seed
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
# SLURM does not expand shell variables in #SBATCH lines; ${GPUS:-1} here was rejected at submit
# with "Invalid Trackable RESource (TRES) specification". For DDP override on the command line:
#   sbatch --gpus-per-node=4 --export=ALL,SEED=123 deploy/deltaai/train_b5.slurm
#SBATCH --cpus-per-task=16
# No --mem. ghx4 bills with PriorityFlags=MAX_TRES over CPU=13.89, Mem=9.017/GB, GPU=1000:
# --mem=200g made a 1-GPU job bill as 1.80 GPUs (200*9.017=1803 > 1000), ~38.6 GPU-hours
# per 48 h link. DefMemPerGPU=113560 MB (110.9 GB) is tuned so memory never exceeds the
# GPU charge and it scales with --gpus-per-node; a 4-GPU job gets 443 GB, not a 200 GB cap.
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --output=%x-%j.out
# One B5 seed per job chain (protocol A28): sbatch --export=ALL,SEED=456 deploy/deltaai/train_b5.slurm
# The job saves a resumable latest_state every N optimizer steps and on SIGUSR1 (sent 15 min before the wall-clock
# limit), then resubmits itself with --dependency=afterany until TRAINING_DONE (or TRAINING_FAILED) exists.
set -euo pipefail
: "${SCRATCH:=/work/nvme/bilv/$USER}"   # DeltaAI has no /scratch and no $SCRATCH. Docs: /work/hdd for large job I/O, /work/nvme for many small I/Os; both Lustre, 1 TB default
command -v apptainer >/dev/null 2>&1 || module load apptainer   # DeltaAI ships /usr/bin/apptainer with no module
WORK=$SCRATCH/transgenic; SIF=$SCRATCH/containers/transgenic-ngc.sif
# SLURM_GPUS_ON_NODE is always a bare integer and is set for every request form.
# SLURM_GPUS_PER_NODE echoes the submitter's string (can be nvidia_gh200_120gb:4) and is
# unset for --gres=gpu:N or --gpus=N, which would silently run 1 process on 4 GPUs.
SEED=${SEED:?set SEED=123|456|789}; GPUS=${SLURM_GPUS_ON_NODE:-${GPUS:-1}}; RUN=/work/runs/seed$SEED; RUN_HOST=$WORK/runs/seed$SEED
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
# --writable-tmpfs: the SIF root is read-only and `pip install -e` writes to site-packages.
apptainer exec --nv --writable-tmpfs --bind "$WORK":/work "$SIF" bash -lc "
  cd /work/repo && pip install -q -e . --no-deps &&
  bash deploy/deltaai/preflight_b5.sh --bundle /work --repo /work/repo &&
  mkdir -p $RUN &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
    --config configs/b5_400m_win_v3.json --db /work/db/b5_full_a40_v1.db --seed $SEED --output-dir $RUN --resume auto \
    --save-every-n-steps ${SAVE_EVERY:-200} 2>> $RUN/train.err
" &
TRAIN_PID=$!
# forward the pre-limit signal to the training process so it saves latest_state and exits cleanly
trap 'echo "USR1: forwarding to trainer"; pkill -USR1 -P $TRAIN_PID 2>/dev/null; kill -USR1 $TRAIN_PID 2>/dev/null' USR1
# `wait` returns 138 the moment the trapped USR1 arrives, and under set -e that would end the
# script HERE, before the TRAINING_DONE/FAILED decision and the resubmit -- every seed would die
# at 48 h with no next link. Re-wait while the trainer is still alive; keep its real exit code.
TRAIN_RC=0
while :; do
  if wait "$TRAIN_PID"; then TRAIN_RC=0; break; fi
  TRAIN_RC=$?
  kill -0 "$TRAIN_PID" 2>/dev/null || break
done
# chain: resubmit unless finished, failed, or the chain limit is reached
if [ -f "$RUN_HOST/TRAINING_DONE" ]; then echo "training finished"
elif [ -f "$RUN_HOST/TRAINING_FAILED" ] || [ "$TRAIN_RC" -ne 0 ]; then echo "training failed (rc=$TRAIN_RC); not resubmitting"; touch "$RUN_HOST/TRAINING_FAILED"
elif [ "$CHAIN_N" -ge "$CHAIN_MAX" ]; then echo "chain limit $CHAIN_MAX reached; not resubmitting"
else
  echo "resubmitting chain job $((CHAIN_N + 1))"
  # --gpus-per-node must be repeated on the command line: the directive is fixed at 1 and an
  # --export=GPUS alone would silently drop a 4-GPU chain to one GPU on the next link.
  sbatch --gpus-per-node=$GPUS --dependency=afterany:$SLURM_JOB_ID --export=ALL,SEED=$SEED,GPUS=$GPUS,CHAIN_N=$((CHAIN_N + 1)),CHAIN_MAX=$CHAIN_MAX "$WORK/repo/deploy/deltaai/train_b5.slurm"
fi

# After each epoch: python revision/scripts/60_epoch_budget.py --log $RUN/train.err --patience 3
