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
WORK=$SCRATCH/transgenic
# SIF is an env override like STEPS/TIER (sbatch --export=ALL,SIF=...), so another NGC tag can be tried
# without editing this file. The job prints which image it ran in; #18 must cite that line.
SIF=${SIF:-$SCRATCH/containers/transgenic-ngc.sif}
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
echo "image: $SIF"; apptainer exec "$SIF" cat /opt/transgenic/IMAGE_INFO 2>/dev/null || echo "image: (no IMAGE_INFO)"
# SLURM_GPUS_ON_NODE is always a bare integer and is set for every request form.
# SLURM_GPUS_PER_NODE echoes the submitter's string (can be nvidia_gh200_120gb:4) and is
# unset for --gres=gpu:N or --gpus=N, which would silently run 1 process on 4 GPUs.
SEED=${SEED:?set SEED=123|456|789}; GPUS=${SLURM_GPUS_ON_NODE:-${GPUS:-1}}; RUN=/work/runs/seed$SEED; RUN_HOST=$WORK/runs/seed$SEED
# Triton must be told where apptainer --nv put the host driver. Inside the container the image's ldconfig
# points at /usr/local/cuda/compat/lib/libcuda.so.1, which does not exist, while the real library is
# bound at /.singularity.d/libs/libcuda.so.1; without this, torch.compile dies at the first forward with
# "libcuda.so cannot found!" (measured on the GB10 SIF, 2026-09-08; triton 3.3.0 honours the variable).
export TRITON_LIBCUDA_PATH=/.singularity.d/libs
# Encoder weights come from the pre-populated cache, never from the network. train_HyenaTransgenic.py:14 sets
# HF_HOME='./HFmodels' itself (so the image's HF_HOME=/work/hf is never used); the model lives under
# HFmodels/hub/models--... and the tokenizer under HFmodels/models--... (cache_dir=./HFmodels). Both were
# shipped from the GB10 test bed and load with HF_HUB_OFFLINE=1 on DeltaAI (6.6 M params, 18 s). Offline
# mode makes a missing file a fast failure instead of six jobs downloading into one shared directory.
export HF_HOME=/work/repo/HFmodels HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
# --writable-tmpfs: the SIF root is read-only; kept so library caches inside the image can write.
# No runtime `pip install -e`. Measured on GB10 in the same SIF recipe: setuptools egg_info wants to write
# src/transgenic.egg-info INTO the bind-mounted repo ("Cannot update time stamp of directory"), so
# concurrent jobs race on one checkout and the trainer died at its first import. The package is a
# src-layout namespace package; PYTHONPATH=/work/repo/src imports it with zero writes.
# First GPU touch of a fresh allocation, outside the trainer (2026-09-09): on GH200 nodes the very first CUDA
# kernel of a new allocation died with `unspecified launch failure` (gh062 x2, gh093, gh121, gh139) and the
# identical second launch on the same node ran clean; a 1-GPU job showed it too, so it is not NCCL. The cheapest
# mitigation is to let a throwaway process take that first touch. Its failure is logged and ignored.
for _g in $(seq 0 $((GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$_g apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize(); print('gpu warm-up ok: device $_g')" \
    || echo "gpu warm-up failed on device $_g (rc=$?); the trainer launch follows anyway"
done
_launch() {   # start the container; TRAIN_PID and ERR_OFF (train.err size before THIS attempt) are set for the caller
  ERR_OFF=$(stat -c %s "$RUN_HOST/train.err" 2>/dev/null || echo 0)
  apptainer exec --nv --writable-tmpfs --bind "$WORK":/work "$SIF" bash -lc "
    cd /work/repo && export PYTHONPATH=/work/repo/src &&
    bash deploy/deltaai/preflight_b5.sh --bundle /work --repo /work/repo &&
    mkdir -p $RUN &&
    accelerate launch --num_processes=$GPUS --mixed_precision=bf16 train/train_HyenaTransgenic.py \
      --config configs/b5_400m_win_v3.json --db /work/db/b5_full_a40_v1.db --seed $SEED --output-dir $RUN --resume auto \
      --save-every-n-steps ${SAVE_EVERY:-200} 2>> $RUN/train.err
  " &
  TRAIN_PID=$!
}
_launch
# The pre-limit USR1 must reach the trainer's Python handler and nothing else. Signalling the apptainer process
# (and its children) killed the container instead: rc 138, the trainer never saw the signal, the link died with
# nothing saved (job 3119780, 2026-09-09). The accelerate/torchrun launcher has no USR1 handler either. The rank
# processes are the descendants of $TRAIN_PID whose command line is the trainer script AND whose parent is the
# launcher; DataLoader workers share the command line but hang off a rank, so they are not signalled.
_descendants() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do echo "$p"; _descendants "$p"; done; }
_ranks() {
  local p pp
  for p in $(_descendants "$TRAIN_PID"); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$c" in *train/train_HyenaTransgenic.py*) ;; *) continue;; esac
    # the launcher's own command line also names the trainer script (rehearsal 3119819 signalled it and it
    # died with 138, taking the rank with it), and so does the `bash -lc` wrapper: a rank is a python process
    # whose command line does not contain "accelerate" and whose parent's does.
    case "$c" in *accelerate*|*"bash -lc"*) continue;; esac
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    case "$(tr '\0' ' ' < "/proc/$pp/cmdline" 2>/dev/null)" in *accelerate*) echo "$p";; esac
  done
}
# If the trainer does not exit within USR1_GRACE seconds of the signal (rehearsals 3119848/3119779: a rank under
# torch.compile/dynamo never ran its Python handler), the watchdog kills the ranks and marks FORCED_PREEMPT, and the
# decision below resubmits the next link from the last published latest_state (at most save_every_n_steps lost).
# The pre-limit margin is 900 s (#SBATCH --signal), so 600 s leaves time for the kill and the sbatch.
# The ranks get SIGTERM, not SIGUSR1: the trainer handles both identically (signal.signal for USR1 and TERM),
# but a rank with torch.compile/dynamo active never ran its USR1 handler (rehearsals 3119848, 3119779,
# probe 3119984) while the same rank saved and exited within a step on TERM (probe 3119984, 'signal 15 received').
# A second USR1 is ignored (trap "" USR1 at entry), every kill is guarded (a rank that exits between discovery
# and the kill must not end the script under errexit), and the marker is scoped to this job id.
FORCED=$RUN_HOST/FORCED_PREEMPT.$SLURM_JOB_ID
trap 'trap "" USR1; R=$(_ranks); echo "USR1: forwarding as SIGTERM to trainer ranks: ${R:-NONE FOUND}"; for p in $R; do kill -TERM "$p" 2>/dev/null || true; done;
      ( sleep "${USR1_GRACE:-600}"; if kill -0 "$TRAIN_PID" 2>/dev/null; then
          echo "trainer still running ${USR1_GRACE:-600}s after USR1; forcing preemption, the last latest_state resumes";
          touch "$FORCED"; for p in $(_ranks); do kill -KILL "$p" 2>/dev/null || true; done; fi ) & WATCHDOG=$!' USR1
# `wait` returns 138 the moment the trapped USR1 arrives, and under set -e that would end the
# script HERE, before the TRAINING_DONE/FAILED decision and the resubmit -- every seed would die
# at 48 h with no next link. Re-wait while the trainer is still alive; keep its real exit code.
# The exit code must be read in the `else` branch of the same `if`: after a failed condition the `if`
# statement's own status is 0, so a `TRAIN_RC=$?` on the next line recorded 0 for every crash and the
# chain resubmitted a dead trainer instead of marking TRAINING_FAILED (reproduced 2026-09-09, child exit 17).
# In-allocation relaunch: a step-0 `unspecified launch failure` ran clean on the identical second launch on the
# same node every time (2026-09-09, gh121/gh139/gh012...), so the first retry happens here, without a new
# allocation; only if that fails too does the decision below resubmit elsewhere (node excluded).
ATTEMPT=1
while :; do
  TRAIN_RC=0
  while :; do
    if wait "$TRAIN_PID"; then TRAIN_RC=0; break; else TRAIN_RC=$?; fi
    kill -0 "$TRAIN_PID" 2>/dev/null || break
  done
  if [ "$TRAIN_RC" -ne 0 ] && [ ! -f "$FORCED" ] && [ "$ATTEMPT" -lt "${LAUNCH_TRIES_MAX:-2}" ] \
     && tail -c +$((ERR_OFF + 1)) "$RUN_HOST/train.err" 2>/dev/null | grep -F "CUDA error: unspecified launch failure" > /dev/null; then
    ATTEMPT=$((ATTEMPT + 1)); echo "launch failure on $SLURM_NODELIST (rc=$TRAIN_RC); relaunching in this allocation (attempt $ATTEMPT)"
    _launch; continue
  fi
  break
done
# chain: resubmit unless finished, failed, or the chain limit is reached
# (`|| true`: under set -e a failing command after the last && ends the script -- the watchdog had already
#  exited after forcing the preemption and the link died here without resubmitting, rehearsal 3119991)
[ -z "${WATCHDOG:-}" ] || kill "$WATCHDOG" 2>/dev/null || true
if [ -f "$RUN_HOST/TRAINING_DONE" ]; then echo "training finished"
elif [ -f "$FORCED" ]; then
  rm -f "$FORCED"
  if [ "$CHAIN_N" -ge "$CHAIN_MAX" ]; then echo "forced preemption (rc=$TRAIN_RC) at chain limit $CHAIN_MAX; not resubmitting"; exit 1; fi
  echo "forced preemption (rc=$TRAIN_RC); resubmitting chain job $((CHAIN_N + 1)) from the last latest_state"
  sbatch --gpus-per-node=$GPUS ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} --dependency=afterany:$SLURM_JOB_ID --export=ALL,SEED=$SEED,GPUS=$GPUS,CHAIN_N=$((CHAIN_N + 1)),CHAIN_MAX=$CHAIN_MAX,LAUNCH_RETRY=0 "$WORK/repo/deploy/deltaai/train_b5.slurm"
elif [ "$TRAIN_RC" -ne 0 ] && [ ! -f "$RUN_HOST/TRAINING_FAILED" ] && [ "${LAUNCH_RETRY:-0}" -lt "${LAUNCH_RETRY_MAX:-1}" ] \
     && tail -c +$((ERR_OFF + 1)) "$RUN_HOST/train.err" 2>/dev/null | grep -F "CUDA error: unspecified launch failure" > /dev/null; then
  # A node-side failure, not a code failure: `unspecified launch failure` has hit gh062 (x2), gh093 and gh121
  # (2026-09-08/09), always at the first kernels, and a retry on another node ran clean. Resubmit the SAME link
  # once or twice with this node excluded, so a healthy checkout is not marked failed by a bad GPU; the trainer
  # resumes from the last published checkpoint. Only this attempt's stderr is inspected (ERR_OFF), grep -F reads
  # the whole stream (grep -q + pipefail could fail the test on a match), the exclusion list travels in the
  # process environment (a comma inside --export=NAME=a,b splits into two variables), the retry waits for this
  # job to end, and this link still exits non-zero so sacct shows the death (the dependency is afterany).
  # The in-allocation relaunch above already retried on this node; a second failure here excludes it.
  export EXCLUDE_NODES="${EXCLUDE_NODES:+$EXCLUDE_NODES,}$SLURM_NODELIST"
  [ -z "${SYNC_PID:-}" ] || { kill "$SYNC_PID" 2>/dev/null || true; wait "$SYNC_PID" 2>/dev/null || true; }
  echo "launch failure on $SLURM_NODELIST twice (rc=$TRAIN_RC); resubmitting chain job $CHAIN_N with --exclude=$EXCLUDE_NODES (retry $((${LAUNCH_RETRY:-0} + 1))/${LAUNCH_RETRY_MAX:-1})"
  sbatch --gpus-per-node=$GPUS ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} --dependency=afterany:$SLURM_JOB_ID --export=ALL,SEED=$SEED,GPUS=$GPUS,CHAIN_N=$CHAIN_N,CHAIN_MAX=$CHAIN_MAX,LAUNCH_RETRY=$((${LAUNCH_RETRY:-0} + 1)) "$WORK/repo/deploy/deltaai/train_b5.slurm"
  exit 1
elif [ -f "$RUN_HOST/TRAINING_FAILED" ] || [ "$TRAIN_RC" -ne 0 ]; then
  echo "training failed (rc=$TRAIN_RC); not resubmitting"; touch "$RUN_HOST/TRAINING_FAILED"
  exit 1    # sacct must show FAILED, not COMPLETED, for a link whose trainer died
elif [ "$CHAIN_N" -ge "$CHAIN_MAX" ]; then echo "chain limit $CHAIN_MAX reached; not resubmitting"
else
  echo "resubmitting chain job $((CHAIN_N + 1))"
  # --gpus-per-node must be repeated on the command line: the directive is fixed at 1 and an
  # --export=GPUS alone would silently drop a 4-GPU chain to one GPU on the next link.
  # EXCLUDE_NODES (if any) is inherited through --export=ALL from this job's environment; LAUNCH_RETRY resets per link.
  sbatch --gpus-per-node=$GPUS ${EXCLUDE_NODES:+--exclude=$EXCLUDE_NODES} --dependency=afterany:$SLURM_JOB_ID --export=ALL,SEED=$SEED,GPUS=$GPUS,CHAIN_N=$((CHAIN_N + 1)),CHAIN_MAX=$CHAIN_MAX,LAUNCH_RETRY=0 "$WORK/repo/deploy/deltaai/train_b5.slurm"
fi

# After each epoch: python revision/scripts/60_epoch_budget.py --log $RUN/train.err --patience 3
