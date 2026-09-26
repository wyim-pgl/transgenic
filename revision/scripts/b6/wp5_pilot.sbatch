#!/bin/bash
#SBATCH --job-name=b6-wp5-pilot
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
# No --mem (MAX_TRES billing; see deploy/deltaai/train_b5.slurm).
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --exclude=gh062,gh093
#SBATCH --output=%x-%j.out
#
# WP5 pilot (work plan v1.2 WP5; A46 rule 11): one arm (B6 or C2) per run, 4 GPUs (DDP), seed 123,
# aug_seed 20261037, exactly N_pilot = 4,142 optimizer updates, then step_04142/ (PILOT_FINAL, PRUNE_EXEMPT) and
# TRAINING_DONE. The gated config configs/b6_400m_win_v3_wp5_pilot.json pins arm set, stop target, seed and world
# size; the trainer refuses --allow-ungated with it, and this script never passes it. LR schedule = the full-run
# schedule truncated at 4,142 (the C3/C4/C6/C7 matched-budget schedule).
#
# TEMPLATE, NOT SUBMITTED. The real-data inputs do not exist yet: BINDINGS must name the WP3 binding manifest
#   {"evidence", "roles", "cache", "freeze", "corpus", "references": {species: fasta}}
# whose paths are CONTAINER paths under /work (the bind of $WORK), e.g. /work/b6/<...>. The placeholder refuses.
#
# Chain (A28/A45 style, pre-queued, no self-resubmission): submit link 1, then link k+1 with afterany on link k:
#   J1=$(sbatch --parsable --export=ALL,ARM=B6,BINDINGS=/work/<...>/bindings.json revision/scripts/b6/wp5_pilot.sbatch)
#   sbatch --dependency=afterany:$J1 --export=ALL,ARM=B6,BINDINGS=/work/<...>/bindings.json revision/scripts/b6/wp5_pilot.sbatch
# A link that finds TRAINING_DONE exits 0 without touching the GPUs; every link resumes latest_state (--resume auto)
# and the trainer refuses a checkpoint whose stop target, arm, seed or inputs differ (run identity).
set -euo pipefail
ARM=${ARM:?set ARM=B6 or ARM=C2}
BINDINGS=${BINDINGS:-__WP3_REAL_DATA_BINDINGS_JSON__}
case "$ARM" in B6|C2) ;; *) echo "REFUSED: ARM=$ARM; the WP5 pilot trains B6 and C2 only" >&2; exit 3;; esac
case "$BINDINGS" in __*__) echo "REFUSED: BINDINGS is the placeholder $BINDINGS; the real-data binding manifest does not exist yet" >&2; exit 3;; esac
: "${SCRATCH:=/work/nvme/bilv/$USER}"
command -v apptainer >/dev/null 2>&1 || module load apptainer
WORK=$SCRATCH/transgenic
SIF=${SIF:-$SCRATCH/containers/transgenic-ngc-26.08.sif}
REPO_DIR=${REPO_DIR:-$WORK/repo_b6}
N_PILOT=4142
CONFIG=configs/b6_400m_win_v3_wp5_pilot.json
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
[ -d "$REPO_DIR/.git" ] || { echo "REFUSED: REPO_DIR is not a git checkout: $REPO_DIR" >&2; exit 3; }
case "$REPO_DIR" in */repo|*/repo_seeds) echo "REFUSED: $REPO_DIR is a frozen B5 seed checkout" >&2; exit 3;; esac
[ -f "$REPO_DIR/$CONFIG" ] || { echo "REFUSED: $REPO_DIR has no $CONFIG" >&2; exit 3; }
ls "$REPO_DIR"/HFmodels/hub/models--LongSafari--hyenadna-large-1m-seqlen-hf/snapshots/*/model.safetensors >/dev/null 2>&1 \
  || { echo "REFUSED: encoder cache missing under $REPO_DIR/HFmodels" >&2; exit 3; }
# The bindings path is a container path (/work/... = $WORK/...); check the host side before spending GPU time.
BINDINGS_HOST=${BINDINGS/#\/work\//$WORK/}
[ -s "$BINDINGS_HOST" ] || { echo "REFUSED: binding manifest not found: $BINDINGS_HOST (container $BINDINGS)" >&2; exit 3; }
GPUS=${SLURM_GPUS_ON_NODE:-0}
[ "$GPUS" = 4 ] || { echo "REFUSED: $GPUS GPUs; the WP5 pilot runs on exactly 4 (A51: production on 4)" >&2; exit 3; }

RUN_HOST=$WORK/runs/b6_wp5_pilot_$ARM
RUN=/work/runs/b6_wp5_pilot_$ARM
if [ -f "$RUN_HOST/TRAINING_DONE" ]; then echo "pilot $ARM already finished ($RUN_HOST/TRAINING_DONE); nothing to do"; exit 0; fi
if [ -f "$RUN_HOST/TRAINING_FAILED" ]; then echo "REFUSED: $RUN_HOST/TRAINING_FAILED exists; remove it deliberately first" >&2; exit 1; fi
mkdir -p "$RUN_HOST"
for _m in "$RUN_HOST"/LINK_ACTIVE.*; do
  [ -e "$_m" ] || continue
  _j=${_m##*.}
  [ "$_j" != "${SLURM_JOB_ID:-}" ] || continue
  if _q=$(squeue -j "$_j" -h -o %T 2>&1); then
    [ -z "$_q" ] || { echo "REFUSED: link job $_j is still $_q on $RUN_HOST" >&2; exit 3; }
  else
    case "$_q" in *"Invalid job id"*) ;; *) echo "REFUSED: cannot confirm that link job $_j has ended ($_q)" >&2; exit 3;; esac
  fi
  rm -f "$_m"
done
touch "$RUN_HOST/LINK_ACTIVE.${SLURM_JOB_ID:-0}"
trap 'rm -f "$RUN_HOST/LINK_ACTIVE.${SLURM_JOB_ID:-0}"' EXIT
python3 "$REPO_DIR/revision/scripts/b6/storage_preflight.py" "$RUN_HOST"

echo "WP5 pilot $ARM: repo $REPO_DIR @ $(git -C "$REPO_DIR" log --oneline -1 | cut -c1-70)  image: $SIF  config sha256 $(sha256sum "$REPO_DIR/$CONFIG" | cut -d' ' -f1)"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
# Deterministic kernels (CP4 conditions: byte-stable step checkpoint across a resume). DETERMINISTIC=0 drops them;
# the choice is recorded in the run identity, so every link of one run must use the same value.
DET_ARG=""; [ "${DETERMINISTIC:-1}" = 1 ] && DET_ARG="--deterministic"
for _g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$_g apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()" \
    || echo "gpu warm-up failed on device $_g; the trainer launch follows anyway"
done

apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
  cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels && mkdir -p $RUN &&
  accelerate launch --num_processes=4 --mixed_precision=bf16 train/train_HyenaTransgenic_B6.py \
    --config $CONFIG --arm $ARM --seed 123 --bindings $BINDINGS --output-dir $RUN --resume auto \
    --stop-after-updates $N_PILOT --save-every-n-steps ${SAVE_EVERY:-200} --mixed-precision bf16 $DET_ARG \
    2>> $RUN/train.err
" &
TRAIN_PID=$!
# Pre-limit USR1: forward as SIGTERM to the trainer ranks only (python processes of the trainer script whose parent
# is the accelerate launcher); the trainer saves latest_state at the next optimizer update and exits (train_b5.slurm).
_descendants() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do echo "$p"; _descendants "$p"; done; }
_ranks() {
  local p pp c
  for p in $(_descendants "$TRAIN_PID"); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
    case "$c" in *train/train_HyenaTransgenic_B6.py*) ;; *) continue;; esac
    case "$c" in *accelerate*|*"bash -lc"*) continue;; esac
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    case "$(tr '\0' ' ' < "/proc/$pp/cmdline" 2>/dev/null)" in *accelerate*) echo "$p";; esac
  done
}
trap 'trap "" USR1; R=$(_ranks); echo "USR1: SIGTERM to trainer ranks: ${R:-NONE FOUND}"; for p in $R; do kill -TERM "$p" 2>/dev/null || true; done' USR1
TRAIN_RC=0
while :; do
  if wait "$TRAIN_PID"; then TRAIN_RC=0; break; else TRAIN_RC=$?; fi
  kill -0 "$TRAIN_PID" 2>/dev/null || break
done

STEP=$RUN_HOST/step_$(printf %05d $N_PILOT)
if [ -f "$RUN_HOST/TRAINING_DONE" ]; then
  [ -f "$STEP/PILOT_FINAL" ] || { echo "FAIL: TRAINING_DONE without $STEP/PILOT_FINAL" >&2; touch "$RUN_HOST/TRAINING_FAILED"; exit 1; }
  echo "WP5 pilot $ARM finished at $N_PILOT updates: $STEP model.safetensors sha256 $(sha256sum "$STEP/model.safetensors" | cut -d' ' -f1)"
  exit 0
elif [ "$TRAIN_RC" -eq 0 ]; then
  echo "link ended before $N_PILOT updates (latest_state saved); the next queued link resumes"
  exit 0
fi
echo "WP5 pilot $ARM: trainer failed (rc=$TRAIN_RC); see $RUN_HOST/train.err" >&2
touch "$RUN_HOST/TRAINING_FAILED"
exit 1
