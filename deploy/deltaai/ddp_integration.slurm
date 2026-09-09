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
export TRANSGENIC_NO_COMPILE=1        # the compile attempt only costs minutes here; the recipe path is eager anyway
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
mkdir -p "$RUN"
run_in "$COMMON" 2>> "$RUN/train.err" &
LEG=$!
n=0; while [ ! -d "$RUN/latest_state" ] && [ $n -lt 900 ]; do sleep 2; n=$((n+1)); kill -0 $LEG 2>/dev/null || break; done
if [ -d "$RUN/latest_state" ]; then
  sleep 5
  # one rank only: the highest-pid trainer process (a rank, not the launcher). pgrep -f from this script
  # cannot match itself: the script's own cmdline is its path, not the pattern.
  ONE=$(pgrep -f "python.*train/train_HyenaTransgenic.py" | sort -n | tail -1)
  echo "USR1 -> single rank pid $ONE at $(date -Is)"; kill -USR1 "$ONE"
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
echo "DDP_INTEGRATION $VERDICT job=$SLURM_JOB_ID repo=$(git -C "$REPO_DIR" rev-parse --short HEAD) image=$(basename "$SIF")"
