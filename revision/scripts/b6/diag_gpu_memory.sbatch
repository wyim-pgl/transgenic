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
# B6 1-GPU memory diagnostic (diag_gpu_memory.py): why the real-data 4-GPU smoke OOMed in the compiled backward
# while the CP2 bench passed. Runs a matrix of configurations sequentially, each in a fresh process:
#   a_trainer       deterministic on,  PYTORCH_CUDA_ALLOC_CONF unset                 (trainer-identical)
#   b_expandable    deterministic on,  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   c_nondet        deterministic off, PYTORCH_CUDA_ALLOC_CONF unset
# Each run: TORCH_LOGS=recompiles, its own JSON and log, a wall-clock cap. One failing configuration does not stop
# the others. Ends with a summary table (peak allocated/reserved GiB per shape, compiled graphs, OOM shape).
#
# Time budget (~50 min of the 60): per configuration ~1 min model load + several minutes of compile warm-up (one
# graph per new tier / label-dynamic recompile) + UPDATES x ACCUM micro-batches at ~1.9 s mean for the default mix
# (129k ~2.5 s, 61k ~1.2 s, 31k ~0.6 s). UPDATES=8 x ACCUM=24 = 192 micro-batches ~ 6 min; the diagnostic stops itself
# at an update boundary after MAX_MINUTES and `timeout` kills it at RUN_MINUTES (the JSON is rewritten after every
# micro-batch, so a killed run still leaves its measurements).
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
UPDATES=${UPDATES:-8}
ACCUM=${ACCUM:-}                    # empty = the trainer's per-rank value (accumulation_steps(cfg,1) // 4 = 24)
SHAPES=${SHAPES:-}                  # empty = diag_gpu_memory.py default mix
MAX_MINUTES=${MAX_MINUTES:-13}
RUN_MINUTES=${RUN_MINUTES:-16}
CONFIGS=${CONFIGS:-a_trainer b_expandable c_nondet}
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
case "$REPO_DIR" in */repo|*/repo/|*/repo_seeds|*/repo_seeds/|*/repo_dev|*/repo_dev/)
  echo "REFUSED: $REPO_DIR is a frozen seed / dev checkout" >&2; exit 3;; esac
[ -f "$REPO_DIR/revision/scripts/b6/diag_gpu_memory.py" ] || { echo "REFUSED: $REPO_DIR has no diag_gpu_memory.py" >&2; exit 3; }
[ -f "$REPO_DIR/$CONFIG" ] || { echo "REFUSED: $REPO_DIR has no $CONFIG" >&2; exit 3; }
ls "$REPO_DIR"/HFmodels/hub/models--LongSafari--hyenadna-large-1m-seqlen-hf/snapshots/*/model.safetensors >/dev/null 2>&1 \
  || { echo "REFUSED: encoder cache missing under $REPO_DIR/HFmodels" >&2; exit 3; }
GPUS=${SLURM_GPUS_ON_NODE:-1}
[ "$GPUS" = 1 ] || { echo "REFUSED: $GPUS GPUs; this diagnostic runs on exactly 1" >&2; exit 3; }
JOB=${SLURM_JOB_ID:-local}
echo "repo: $REPO_DIR @ $( (git -C "$REPO_DIR" log --oneline -1 2>/dev/null || cat "$REPO_DIR/SOURCE_COMMIT" 2>/dev/null || echo unknown) | cut -c1-70)  image: $SIF"
echo "config: $CONFIG sha256 $(sha256sum "$REPO_DIR/$CONFIG" | cut -d' ' -f1)  updates=$UPDATES accum=${ACCUM:-trainer} max_minutes=$MAX_MINUTES"
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
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()" \
  || echo "gpu warm-up failed; the diagnostic follows anyway"

run_config() {   # $1 = label; returns the diagnostic's exit code (124/137 when timeout stopped it)
  local label=$1 det alloc
  case "$label" in
    a_trainer)    det=on;  alloc="unset PYTORCH_CUDA_ALLOC_CONF";;
    b_expandable) det=on;  alloc="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True";;
    c_nondet)     det=off; alloc="unset PYTORCH_CUDA_ALLOC_CONF";;
    *) echo "unknown configuration $label" >&2; return 4;;
  esac
  local extra=""
  [ -n "$ACCUM" ] && extra="$extra --accum $ACCUM"
  [ -n "$SHAPES" ] && extra="$extra --shapes $SHAPES"
  timeout --signal=TERM --kill-after=60 "${RUN_MINUTES}m" \
    apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
      cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels &&
      $alloc && export TORCH_LOGS=recompiles TORCHINDUCTOR_CACHE_DIR=$INDUCTOR_CACHE DIAG_LABEL=$label &&
      echo \"[$label] PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-(unset)} deterministic=$det\" &&
      accelerate launch --num_processes=1 --mixed_precision=bf16 revision/scripts/b6/diag_gpu_memory.py \
        --config $CONFIG --deterministic $det --updates $UPDATES --chain-elements 511 --max-minutes $MAX_MINUTES \
        $extra --out $OUT/$label.json
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
  echo "===== $label end $(date +%T) rc=$rc"
done

echo "===== summary (job $JOB; rc 0 = completed or time budget, 5 = OOM recorded, 124/137 = killed by timeout)"
for label in $CONFIGS; do echo "$label rc=${RCS[$label]}"; done
apptainer exec --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
  cd /work/repo && PYTHONPATH=/work/repo/src python revision/scripts/b6/diag_gpu_memory.py --summarize ${JSONS[*]}
" || echo "summary table failed (see the JSONs under $OUT_HOST)"
echo "driver_peak_mem_mib: $(awk -F', ' '{if($3+0>m)m=$3+0} END{print m+0}' "$MEMLOG")"
echo "outputs: $OUT_HOST"
STATUS=0
for label in $CONFIGS; do case "${RCS[$label]}" in 0|5) ;; *) STATUS=1;; esac; done
exit $STATUS
