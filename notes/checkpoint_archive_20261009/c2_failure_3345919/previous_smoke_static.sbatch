#!/bin/bash
#SBATCH --job-name=b6-gpu-smoke-static
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@720
#SBATCH --exclude=gh062,gh093
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/b6_prep_20260925/logs/%x-%j.out
# GPU smoke for the static DNA-length compile fix (233574a, user decision 2026-10-07): automatic_dynamic_shapes off,
# 2 graphs per L tier, expandable_segments allocator (as the pilot will run). Same pilot config, B6, 4 GPUs. The earlier
# smoke 3319603 (826e68e) GPU-OOMed at ~update 165 on row 3915 (rn 10431); replay showed 92 GiB from compile history.
# Original header:
# GPU smoke before the WP5 pilot resubmission (user approval 2026-10-05, ~8 GPU-h): the pilot's exact config, arm B6,
# seed 123, 4 GPUs, repo_b6_826e68e (ObsCache memory-mapped reader), but a separate run dir runs/b6_smoke_826e68e_B6.
#  0. in-job recheck of the derived mmap files (b6_prebuild_obs_mmap.py: re-hash against sidecars; must build 0);
#  1. trainer with B6_OBS_CACHE_REQUIRE_PREBUILT=1 (no derived file is built during training);
#  2. sampler every 30 s: job cgroup memory, summed RSS, per-GPU memory -> $RUN_HOST/smoke_mem.tsv;
#  3. USR1 20 min before the limit -> SIGTERM to the trainer ranks -> latest_state saved, exit (no TRAINING_DONE).
# Measures: host memory peak (the pilot links OOMed at ~465 GB after ~19 min of training), GPU peak (rank 2 GPU OOM
# in 3306594), s/step, first latest_state. The smoke run dir is never resumed into the pilot runs.
set -euo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic; HDD=/work/hdd/bilv/wcyim/transgenic/b6_prep_20260925
REPO_DIR=$WORK/repo_b6_233574a
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
ARM=B6; CONFIG=configs/b6_400m_win_v3_wp5_pilot.json; B=/work/b6_prep_20260925/bindings/train_v1.deltaai.json
command -v apptainer >/dev/null 2>&1 || module load apptainer
test "$(cat $REPO_DIR/SOURCE_COMMIT)" = 233574a
[ "${SLURM_GPUS_ON_NODE:-0}" = 4 ] || { echo "REFUSED: ${SLURM_GPUS_ON_NODE:-0} GPUs (smoke runs on 4)" >&2; exit 3; }
RUN_HOST=$WORK/runs/b6_smoke_233574a_$ARM; RUN=/work/runs/b6_smoke_233574a_$ARM
[ ! -e "$RUN_HOST" ] || { echo "REFUSED: $RUN_HOST exists; move it aside first" >&2; exit 3; }
mkdir -p "$RUN_HOST"
BINDS=(--bind "$WORK":/work --bind "$HDD":/hdd_prep:ro
       --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro
       --bind "$REPO_DIR":/work/repo)
echo "start $(date -Is) node $(hostname) repo $(git -C $REPO_DIR log --oneline -1 | cut -c1-60) config $(sha256sum $REPO_DIR/$CONFIG | cut -c1-16)"

# 0. derived mmap recheck (CPU), on the same container path the trainer uses.
CACHE=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['cache'])" "$WORK/b6_prep_20260925/bindings/train_v1.deltaai.json")
echo "cache (container) $CACHE"
apptainer exec "${BINDS[@]}" "$SIF" bash -lc "cd /work/repo && export PYTHONPATH=/work/repo/src PYTHONDONTWRITEBYTECODE=1 &&
  python revision/scripts/evidence/b6_prebuild_obs_mmap.py $CACHE" > "$RUN_HOST/mmap_recheck.jsonl"
tail -1 "$RUN_HOST/mmap_recheck.jsonl"
python3 - "$RUN_HOST/mmap_recheck.jsonl" <<'PY'
import json, sys
s = json.loads(open(sys.argv[1]).read().strip().splitlines()[-1])
assert s["shards"] == 455 and s["built"] == 0 and s["rows"] == 653922028 and s["derived_bytes"] == 53713869462, s
print("mmap recheck PASS", s)
PY
echo "recheck done $(date -Is)"

# 2. memory sampler
CG=""; _p=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null || true)
while [ -n "$_p" ] && [ "$_p" != / ]; do
  case "$_p" in *job_*) [ -f "/sys/fs/cgroup$_p/memory.current" ] && CG=/sys/fs/cgroup$_p/memory.current; case "$_p" in */job_*/*) ;; *) break;; esac;; esac
  _p=${_p%/*}
done
echo "cgroup memory file: ${CG:-none}"
( printf 'time\tcgroup_GB\tuser_rss_GB\tgpu_MiB\n'
  while :; do
    c=NA; [ -n "$CG" ] && c=$(awk '{printf "%.1f", $1/2^30}' "$CG" 2>/dev/null || echo NA)
    r=$(ps -u "$USER" -o rss= 2>/dev/null | awk '{s+=$1} END{printf "%.1f", s/2^20}' || echo NA)
    g=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | paste -sd, || echo NA)
    printf '%s\t%s\t%s\t%s\n' "$(date +%H:%M:%S)" "$c" "$r" "$g"
    sleep 30
  done ) > "$RUN_HOST/smoke_mem.tsv" 2>/dev/null &
MON_PID=$!

# 1. trainer: wp5_pilot.sbatch's command with the smoke run dir.
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
for _g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$_g apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()" \
    || echo "gpu warm-up failed on device $_g; the trainer launch follows anyway"
done
echo "trainer launch $(date -Is)"
apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" bash -lc "
  cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels B6_OBS_CACHE_REQUIRE_PREBUILT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TORCH_LOGS=recompiles && echo ALLOC=\$PYTORCH_CUDA_ALLOC_CONF &&
  accelerate launch --num_processes=4 --mixed_precision=bf16 train/train_HyenaTransgenic_B6.py \
    --config $CONFIG --arm $ARM --seed 123 --bindings $B --output-dir $RUN --resume auto \
    --save-every-n-steps 200 --mixed-precision bf16 --deterministic \
    2>> $RUN/train.err
" &
TRAIN_PID=$!
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
trap 'trap "" USR1; R=$(_ranks); echo "USR1 $(date -Is): SIGTERM to trainer ranks: ${R:-NONE FOUND}"; for p in $R; do kill -TERM "$p" 2>/dev/null || true; done' USR1
trap 'kill $MON_PID 2>/dev/null || true' EXIT
TRAIN_RC=0
while :; do
  if wait "$TRAIN_PID"; then TRAIN_RC=0; break; else TRAIN_RC=$?; fi
  kill -0 "$TRAIN_PID" 2>/dev/null || break
done
echo "trainer exit rc=$TRAIN_RC $(date -Is)"
awk -F'\t' 'NR>1 && $2!="NA"{if($2>c)c=$2} NR>1 && $3!="NA"{if($3>r)r=$3} END{print "peak cgroup_GB", c+0, "peak user_rss_GB", r+0}' "$RUN_HOST/smoke_mem.tsv" || true
ls -la "$RUN_HOST" || true
du -sh "$RUN_HOST"/latest_state 2>/dev/null || echo "no latest_state"
tail -5 "$RUN_HOST/train.err" 2>/dev/null || true
exit "$TRAIN_RC"

