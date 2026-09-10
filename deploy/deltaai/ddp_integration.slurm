#!/usr/bin/env bash
#SBATCH --job-name=ddp-int
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=ddp-int-%j.out
#
# DDP integration test (2026-09-09): certifies the multi-GPU trainer path before confirmation seeds 456/789
# run on 4 GPUs. Exercises, on a 480-row subset of the frozen B5 database (one tile tier, original row ids):
#   1. mid-epoch latest_state saves on every rank (--save-every-n-steps 2; rank 0 owns the directory)
#   2. a USR1 delivered to ONE rank only -> every rank saves and exits together (stop flag reduced)
#   3. resume from that latest_state (--resume auto) to the epoch end
#   4. an epoch-end validation with the loss reduced across ranks, epoch_01/, best, a stop decision,
#      TRAINING_DONE at --max-epochs
# It never touches /work/repo (the frozen seed-123 checkout) or /work/db: REPO_DIR defaults to repo_dev and
# the subset lives under /work/runs/ddp_int/. ~5 optimizer steps per epoch at 4 x 24 micro-batches.
set -uo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic
SIF=${SIF:-/work/nvme/bilv/wcyim/containers/transgenic-ngc.sif}
REPO_DIR=${REPO_DIR:-$WORK/repo_dev}
INT=$WORK/runs/ddp_int; RUN=$INT/run_$SLURM_JOB_ID; SUB=$INT/subset_30720.db
TIER=${TIER:-30720}; NTRAIN=${NTRAIN:-480}; NVALID=${NVALID:-48}
mkdir -p "$INT"
export TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
export TRANSGENIC_NO_COMPILE=${TRANSGENIC_NO_COMPILE-1}   # default eager (25.06 falls back anyway); pass an empty value to compile on an image where it works (26.08)
echo "repo: $REPO_DIR @ $(git -C "$REPO_DIR" log --oneline -1 | cut -c1-70)  image: $SIF"

run_in() {   # run_in <python args...>: the trainer inside the container, 4 ranks
  apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
    cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels &&
    accelerate launch --num_processes=4 --mixed_precision=bf16 train/train_HyenaTransgenic.py $*"
}

# --- 0. subset database (idempotent): original rn values, one tier, both splits ---
if [ ! -s "$SUB" ]; then
  apptainer exec --bind "$WORK":/work "$SIF" python - "$TIER" "$NTRAIN" "$NVALID" <<'PY'
import duckdb, sys
tier, ntrain, nvalid = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
src = "/work/db/b5_full_a40_v1.db"; dst = "/work/runs/ddp_int/subset_%d.db" % tier
con = duckdb.connect(dst)
con.execute(f"ATTACH '{src}' AS src (READ_ONLY)")
con.execute(f"""CREATE TABLE geneList AS
  (SELECT * FROM src.geneList WHERE split='train' AND gff IS NOT NULL AND COALESCE(train_weight,1.0)>0 AND fin-start={tier} ORDER BY rn LIMIT {ntrain})
  UNION ALL
  (SELECT * FROM src.geneList WHERE split='valid' AND gff IS NOT NULL AND COALESCE(train_weight,1.0)>0 AND fin-start={tier} ORDER BY rn LIMIT {nvalid})""")
print("subset rows:", con.execute("SELECT split, count(*) FROM geneList GROUP BY split ORDER BY split").fetchall())
con.close()
PY
fi
[ -s "$SUB" ] || { echo "REFUSED: no subset db"; exit 3; }
COMMON="--config configs/b5_400m_win_v3.json --db /work/runs/ddp_int/subset_$TIER.db --seed 456 --no-wandb \
  --save-every-n-steps 2 --max-epochs 2 --patience 1 --output-dir /work/runs/ddp_int/run_$SLURM_JOB_ID --resume auto"

# --- 1+2. first leg: train, then USR1 to ONE rank after the first latest_state exists ---
# Observed 2026-09-09 (gh121, gh139): the FIRST accelerate launch of a fresh 4-GPU allocation died at its first
# kernels with `CUDA error: unspecified launch failure`, and the identical second launch on the same node ran
# clean. Until the cause is known, leg 1 is retried once on the same node when that signature appears before
# any checkpoint exists (the chain script does the same across links, with the node excluded). ALLOC=native
# runs the trainer on PyTorch's native allocator instead of cudaMallocAsync (train/train_HyenaTransgenic.py:15
# uses setdefault, so an exported value wins) to test the allocator hypothesis.
[ "${ALLOC:-async}" = native ] && export PYTORCH_CUDA_ALLOC_CONF=backend:native
mkdir -p "$RUN"
LAUNCH_TRIES=0
while :; do
  LAUNCH_TRIES=$((LAUNCH_TRIES + 1)); ERR_OFF=$(stat -c %s "$RUN/train.err" 2>/dev/null || echo 0)
  run_in "$COMMON" 2>> "$RUN/train.err" &
  LEG=$!
  sleep 60
  if kill -0 $LEG 2>/dev/null; then break; fi                      # still alive after a minute: past the first kernels
  wait $LEG; RC0=$?
  if [ $RC0 -ne 0 ] && [ ! -d "$RUN/latest_state" ] && [ $LAUNCH_TRIES -lt ${LAUNCH_TRIES_MAX:-2} ] \
     && tail -c +$((ERR_OFF + 1)) "$RUN/train.err" | grep -F "CUDA error: unspecified launch failure" > /dev/null; then
    echo "leg1 launch failure (try $LAUNCH_TRIES, rc=$RC0) on $SLURM_NODELIST at $(date -Is); relaunching on the same node"
    continue
  fi
  run_in "$COMMON" 2>> "$RUN/train.err" &   # died for another reason or budget spent: fall through with a live handle
  LEG=$!
  break
done
echo "leg1 launch tries=$LAUNCH_TRIES alloc=${ALLOC:-async}"
n=0; while [ ! -d "$RUN/latest_state" ] && [ $n -lt 900 ]; do sleep 2; n=$((n+1)); kill -0 $LEG 2>/dev/null || break; done
if [ -d "$RUN/latest_state" ]; then
  sleep 5
  # one RANK only: the rank processes are the direct children of the accelerate launcher. Matching the trainer
  # cmdline is not enough -- DataLoader workers are forked from the ranks and carry the same cmdline, and the
  # highest pid was a worker in job 3119205 (the signal did nothing). pgrep -f from this script cannot match
  # itself: the script's own cmdline is its path, not the pattern.
  LAUNCHER=$(pgrep -f "accelerate launch --num_processes" | head -1)
  ONE=$(pgrep -P "$LAUNCHER" | sort -n | tail -1)
  echo "process tree before the signal:"; ps -o pid,ppid,stat,etime,cmd --forest -g "$(ps -o pgid= -p "$LAUNCHER" | tr -d ' ')" 2>/dev/null | cut -c1-160 | head -40
  echo "target cmdline: $(tr '\0' ' ' < /proc/$ONE/cmdline 2>/dev/null | cut -c1-160)"
  echo "TERM -> single rank pid $ONE (launcher $LAUNCHER, ranks: $(pgrep -P "$LAUNCHER" | tr '\n' ' ')) at $(date -Is)"; kill -TERM "$ONE"   # TERM, as the chain script sends it: a rank under dynamo never ran its USR1 handler (2026-09-09)
fi
wait $LEG; RC1=$?
echo "leg1 rc=$RC1  latest_state=$( [ -d "$RUN/latest_state" ] && echo yes || echo no )  preempt_lines=$(grep -c 'exiting cleanly' "$RUN/train.err")"
grep -E 'latest_state saved|exiting cleanly|signal .* received|Traceback|Error' "$RUN/train.err" | sort | uniq -c | head -12

# --- 3+4. second leg: resume, run to TRAINING_DONE ---
run_in "$COMMON" 2>> "$RUN/train.err"; RC2=$?
echo "leg2 rc=$RC2"
grep -E 'Resuming from|epoch=|latest_state saved|Early stopping|Traceback|Error' "$RUN/train.err" | sort | uniq -c | head -20
ls "$RUN"; cat "$RUN/trainer_state.json" 2>/dev/null
VERDICT=PASS
[ "$RC1" -eq 0 ] || VERDICT=FAIL
[ "$RC2" -eq 0 ] || VERDICT=FAIL
[ -f "$RUN/TRAINING_DONE" ] || VERDICT=FAIL
[ -d "$RUN/epoch_01" ] || VERDICT=FAIL
grep -q 'exiting cleanly' "$RUN/train.err" || VERDICT=FAIL
grep -q 'Traceback' "$RUN/train.err" && VERDICT=FAIL
echo "DDP_INTEGRATION $VERDICT job=$SLURM_JOB_ID repo=$(git -C "$REPO_DIR" rev-parse --short HEAD) image=$(basename "$SIF") alloc=${ALLOC:-async} launch_tries=$LAUNCH_TRIES"
