#!/bin/bash
#SBATCH --job-name=b6-gpu-mem-diag
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
# No --mem (MAX_TRES billing; see deploy/deltaai/train_b5.slurm).
#SBATCH --time=01:00:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=%x-%j.out
#
# B6 memory diagnostic (diag_gpu_memory.py): why the real-data 4-GPU smoke OOMed in the compiled backward while the
# CP2 bench passed. Runs a matrix of configurations sequentially, each in a fresh process:
#   a_trainer       deterministic on,  PYTORCH_CUDA_ALLOC_CONF unset                 (trainer-identical)
#   b_expandable    deterministic on,  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   c_nondet        deterministic off, PYTORCH_CUDA_ALLOC_CONF unset
# All under DDP as the trainer (torchrun; 1 rank by default, the tool wraps DDP exactly as accelerate.prepare does).
# 1-rank DDP has the gradient buckets but no NCCL peer buffers; for those run 2 ranks:
#   sbatch --gpus-per-node=2 --export=ALL,RANKS=2 revision/scripts/b6/diag_gpu_memory.sbatch
# Each run: TORCH_LOGS=recompiles, its own JSON and log, time caps. One failing configuration does not stop the others.
# Every configuration is classified from the status in its JSON (finished / oom / error / incomplete / missing),
# never from the launcher's exit code (torchrun and accelerate turn exit 5 into 1).
#
# Time budget (60 min job): the 4-GPU smoke ran ~15 s per optimizer update of 24 micro-batches per rank, i.e.
# ~0.6 s per micro-batch (129k tiers longer, ~0.9 s). Per configuration: ~1 min model load + several minutes of
# compile warm-up (the first 3 updates, one graph per new tier / label-dynamic recompile), then MEASURE minutes of
# post-warm-up updates, stopped at an update boundary (cap UPDATES). a/b measure 10 min (~850 micro-batches, ~35
# updates at ACCUM 24), c measures 8 min (~28 updates): all >= 25 updates (600 micro-batches) at the estimate.
# Expected ~16 + 16 + 14 min; hard caps (timeout) 19 + 19 + 16 = 54 min. The JSON is rewritten after every
# micro-batch, so a run killed by its cap still leaves its measurements (classified incomplete).
#
# NOT SUBMITTED by the writer. Submit only after the author's go-ahead:
#   REPO_DIR=$WORK/repo_b6_826e68e sbatch revision/scripts/b6/diag_gpu_memory.sbatch
set -euo pipefail
: "${SCRATCH:=/work/nvme/bilv/$USER}"
command -v apptainer >/dev/null 2>&1 || module load apptainer
WORK=$SCRATCH/transgenic
SIF=${SIF:-$SCRATCH/containers/transgenic-ngc-26.08.sif}
REPO_DIR=${REPO_DIR:-$WORK/repo_b6_826e68e}
CONFIG=${CONFIG:-configs/b6_400m_win_v3_wp5_pilot.json}
RANKS=${RANKS:-1}
DDP=${DDP:-on}
UPDATES=${UPDATES:-60}              # cap; the measure window normally ends the run first
ACCUM=${ACCUM:-}                    # empty = the trainer's per-rank value (accumulation_steps(cfg,1) // 4 = 24)
SHAPES=${SHAPES:-}                  # empty = diag_gpu_memory.py default mix
CONFIGS=${CONFIGS:-a_trainer b_expandable c_nondet}
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
case "$REPO_DIR" in */repo|*/repo/|*/repo_seeds|*/repo_seeds/|*/repo_dev|*/repo_dev/)
  echo "REFUSED: $REPO_DIR is a frozen seed / dev checkout" >&2; exit 3;; esac
[ -f "$REPO_DIR/revision/scripts/b6/diag_gpu_memory.py" ] || { echo "REFUSED: $REPO_DIR has no diag_gpu_memory.py" >&2; exit 3; }
[ -f "$REPO_DIR/$CONFIG" ] || { echo "REFUSED: $REPO_DIR has no $CONFIG" >&2; exit 3; }
ls "$REPO_DIR"/HFmodels/hub/models--LongSafari--hyenadna-large-1m-seqlen-hf/snapshots/*/model.safetensors >/dev/null 2>&1 \
  || { echo "REFUSED: encoder cache missing under $REPO_DIR/HFmodels" >&2; exit 3; }
GPUS=${SLURM_GPUS_ON_NODE:-1}
case "$RANKS" in 1|2) ;; *) echo "REFUSED: RANKS=$RANKS; the diagnostic runs 1 or 2 ranks" >&2; exit 3;; esac
[ "$GPUS" = "$RANKS" ] || { echo "REFUSED: $GPUS GPUs for RANKS=$RANKS (submit with --gpus-per-node=$RANKS)" >&2; exit 3; }
case "$DDP" in on) LAUNCH="python -m torch.distributed.run --standalone --nproc_per_node=$RANKS";;
  off) [ "$RANKS" = 1 ] || { echo "REFUSED: DDP=off needs RANKS=1" >&2; exit 3; }; LAUNCH="python";;
  *) echo "REFUSED: DDP=$DDP (on|off)" >&2; exit 3;; esac
JOB=${SLURM_JOB_ID:-local}
echo "repo: $REPO_DIR @ $( (git -C "$REPO_DIR" log --oneline -1 2>/dev/null || cat "$REPO_DIR/SOURCE_COMMIT" 2>/dev/null || echo unknown) | cut -c1-70)  image: $SIF"
echo "config: $CONFIG sha256 $(sha256sum "$REPO_DIR/$CONFIG" | cut -d' ' -f1)  ranks=$RANKS ddp=$DDP updates<=$UPDATES accum=${ACCUM:-trainer}"
apptainer exec "$SIF" cat /opt/transgenic/IMAGE_INFO 2>/dev/null || echo "image: (no IMAGE_INFO)"
# Same environment as wp5_pilot.sbatch (CUBLAS_WORKSPACE_CONFIG is exported there for every run, deterministic or not).
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
OUT_HOST=$WORK/runs/b6_gpu_mem_diag/$JOB
OUT=/work/runs/b6_gpu_mem_diag/$JOB
mkdir -p "$OUT_HOST"
# One Inductor cache per job, shared by the configurations (recorded in each JSON as TORCHINDUCTOR_CACHE_DIR).
INDUCTOR_CACHE=$OUT/inductor_cache
MEMLOG=$OUT_HOST/smi.csv
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$MEMLOG" 2>/dev/null &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
for _g in $(seq 0 $((RANKS - 1))); do
  CUDA_VISIBLE_DEVICES=$_g apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()" \
    || echo "gpu warm-up failed on device $_g; the diagnostic follows anyway"
done

run_config() {   # $1 = label; returns the launcher's exit code (124/137 when timeout stopped it); not used to classify
  local label=$1 det alloc measure maxmin cap
  case "$label" in
    a_trainer)    det=on;  alloc="unset PYTORCH_CUDA_ALLOC_CONF"; measure=${MEASURE_A:-10}; maxmin=17; cap=19;;
    b_expandable) det=on;  alloc="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"; measure=${MEASURE_B:-10}; maxmin=17; cap=19;;
    c_nondet)     det=off; alloc="unset PYTORCH_CUDA_ALLOC_CONF"; measure=${MEASURE_C:-8}; maxmin=14; cap=16;;
    *) echo "unknown configuration $label" >&2; return 4;;
  esac
  local extra=""
  [ -n "$ACCUM" ] && extra="$extra --accum $ACCUM"
  [ -n "$SHAPES" ] && extra="$extra --shapes $SHAPES"
  timeout --signal=TERM --kill-after=60 "${cap}m" \
    apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
      cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels &&
      $alloc && export TORCH_LOGS=recompiles TORCHINDUCTOR_CACHE_DIR=$INDUCTOR_CACHE DIAG_LABEL=$label &&
      echo \"[$label] PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-(unset)} deterministic=$det ranks=$RANKS ddp=$DDP\" &&
      $LAUNCH revision/scripts/b6/diag_gpu_memory.py \
        --config $CONFIG --ddp $DDP --deterministic $det --updates $UPDATES --chain-elements 511 \
        --measure-minutes $measure --max-minutes $maxmin $extra --out $OUT/$label.json
    "
}

declare -A RCS
JSONS=()
for label in $CONFIGS; do
  echo "===== $label start $(date +%T)"
  rc=0
  run_config "$label" 2>&1 | tee "$OUT_HOST/$label.log" || rc=$?   # pipefail: rc is run_config's (tee exits 0)
  RCS[$label]=$rc
  JSONS+=("$OUT/$label.json")
  echo "===== $label end $(date +%T) launcher rc=$rc (informational; see CLASS below)"
done

echo "===== summary (job $JOB; classification from each JSON's status; rank-1 JSONs, if any, are <label>.rank1.json)"
for label in $CONFIGS; do echo "$label launcher_rc=${RCS[$label]}"; done
SUMRC=0
apptainer exec --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
  cd /work/repo && PYTHONPATH=/work/repo/src python revision/scripts/b6/diag_gpu_memory.py --summarize ${JSONS[*]}
" || SUMRC=$?
echo "driver_peak_mem_mib: $(awk -F', ' '{if($3+0>m)m=$3+0} END{print m+0}' "$MEMLOG")"
echo "outputs: $OUT_HOST"
# 0 = every configuration finished or recorded an OOM (both are results); 1 = an error, incomplete or missing report.
[ "$SUMRC" = 0 ] || exit 1
exit 0
