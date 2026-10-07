#!/bin/bash
#SBATCH --job-name=b6-gpu-mem-real
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
# B6 real-data memory replay (diag_gpu_memory.py --real-bindings): the synthetic diagnostic (job 3322147) peaked at
# 58.5 GiB allocated while the real 4-GPU smoke OOMed at 87 GiB, so a real batch carries something the synthetic one
# does not. Replays, one micro-batch per sample on 1 GPU, the last LAST rows (start rounded down to the trainer's
# 24-micro-batch window boundary; AdamW moments materialised first) of the smoke's rank-3 draw log (rank 3
# OOMed in the compiled backward; the trainer writes a row just before that sample's forward), then EXTRA samples,
# with the trainer's dataset (WP3 bindings, B6Dataset train_track, b6_collate_fn_torch, validate_b6_batch, apply_arm
# B6), model, compile, 1-rank DDP wrap, per-rank accumulation 24 and deterministic algorithms (configuration a).
#   a_trainer       PYTORCH_CUDA_ALLOC_CONF unset
#   b_expandable    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Each in a fresh process, TORCH_LOGS=recompiles, own JSON/log; classification from the JSON status (oom = a sample
# OOMed: recorded with memory_summary, the replay continued). The trainer's ~15 min input verification is skipped
# (VERIFY=1 runs it); the dataset object is the same (only the corpus hash check is skipped).
#
# Time: ~1 min model load + dataset open, compile warm-up per new DNA length (several minutes), 40-63 samples at a few
# s each (CPU rasterisation + compiled step). Caps: 20 min diagnostic budget, timeout 22 min + 1 min kill grace per
# configuration = 46 min, leaving >= 14 min for GPU warm-up and the summary.
#
# REPO_DIR must be a snapshot that contains this commit's diag_gpu_memory.py (repo_b6_826e68e does not): deploy it as
# $WORK/repo_b6_gpudiag_real (git archive of branch b6-gpu-mem-diag + SOURCE_COMMIT + HFmodels copied from
# repo_b6_826e68e), the default below.
# NOT SUBMITTED by the writer. Submit only after the author's go-ahead:
#   sbatch revision/scripts/b6/diag_gpu_memory_real.sbatch
set -euo pipefail
WORK=${WORK_HOST:-/work/nvme/bilv/$USER/transgenic}
HDD=${HDD:-/work/hdd/bilv/$USER/transgenic/b6_prep_20260925}
SIF=${SIF:-/work/nvme/bilv/$USER/containers/transgenic-ngc-26.08.sif}
REPO_DIR=${REPO_DIR:-$WORK/repo_b6_gpudiag_real}
CONFIG=${CONFIG:-configs/b6_400m_win_v3_wp5_pilot.json}
BINDINGS=${BINDINGS:-/work/b6_prep_20260925/bindings/train_v1.deltaai.json}              # container path
DRAW_LOG=${DRAW_LOG:-/work/runs/b6_smoke_826e68e_B6/draw_log/rank03.tsv}                  # container path
LAST=${LAST:-40}
EXTRA=${EXTRA:-}                    # comma list (rn or rn:epoch) or a container path of a file of them
VERIFY=${VERIFY:-0}
MAX_MINUTES=${MAX_MINUTES:-20}
CAP=${CAP:-22}
CONFIGS=${CONFIGS:-a_trainer b_expandable}
command -v apptainer >/dev/null 2>&1 || module load apptainer
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
case "$REPO_DIR" in */repo|*/repo/|*/repo_seeds|*/repo_seeds/|*/repo_dev|*/repo_dev/)
  echo "REFUSED: $REPO_DIR is a frozen seed / dev checkout" >&2; exit 3;; esac
grep -q -- "--real-bindings" "$REPO_DIR/revision/scripts/b6/diag_gpu_memory.py" 2>/dev/null \
  || { echo "REFUSED: $REPO_DIR has no diag_gpu_memory.py with the real-data replay" >&2; exit 3; }
[ -f "$REPO_DIR/$CONFIG" ] || { echo "REFUSED: $REPO_DIR has no $CONFIG" >&2; exit 3; }
ls "$REPO_DIR"/HFmodels/hub/models--LongSafari--hyenadna-large-1m-seqlen-hf/snapshots/*/model.safetensors >/dev/null 2>&1 \
  || { echo "REFUSED: encoder cache missing under $REPO_DIR/HFmodels" >&2; exit 3; }
for _p in "$BINDINGS" "$DRAW_LOG"; do
  _h=${_p/#\/work\//$WORK/}
  [ -s "$_h" ] || { echo "REFUSED: $_h (container $_p) not found" >&2; exit 3; }
done
[ -d "$HDD" ] || { echo "REFUSED: HDD prep dir not found: $HDD" >&2; exit 3; }
[ -d "$WORK/b6_prep_20260925/intervals" ] || { echo "REFUSED: nvme intervals copy not found" >&2; exit 3; }
[ "${SLURM_GPUS_ON_NODE:-1}" = 1 ] || { echo "REFUSED: ${SLURM_GPUS_ON_NODE} GPUs; the replay runs on 1" >&2; exit 3; }
JOB=${SLURM_JOB_ID:-local}
echo "repo: $REPO_DIR @ $(cat "$REPO_DIR/SOURCE_COMMIT" 2>/dev/null || git -C "$REPO_DIR" log --oneline -1 2>/dev/null | cut -c1-70 || echo unknown)  image: $SIF"
echo "config: $CONFIG sha256 $(sha256sum "$REPO_DIR/$CONFIG" | cut -d' ' -f1)  bindings $BINDINGS  draw log $DRAW_LOG last=$LAST extra=${EXTRA:-none} verify=$VERIFY"
BINDS=(--bind "$WORK":/work --bind "$HDD":/hdd_prep:ro
       --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro
       --bind "$REPO_DIR":/work/repo)
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF
OUT_HOST=$WORK/runs/b6_gpu_mem_real/$JOB
OUT=/work/runs/b6_gpu_mem_real/$JOB
mkdir -p "$OUT_HOST"
INDUCTOR_CACHE=$OUT/inductor_cache
MEMLOG=$OUT_HOST/smi.csv
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$MEMLOG" 2>/dev/null &
SMI=$!
trap 'kill $SMI 2>/dev/null || true' EXIT
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()" \
  || echo "gpu warm-up failed; the replay follows anyway"

run_config() {   # $1 = label; returns the launcher's exit code (informational only)
  local label=$1 alloc
  case "$label" in
    a_trainer)    alloc="unset PYTORCH_CUDA_ALLOC_CONF";;
    b_expandable) alloc="export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True";;
    *) echo "unknown configuration $label" >&2; return 4;;
  esac
  local extra=""
  [ -n "$EXTRA" ] && extra="--real-indices $EXTRA"
  [ "$VERIFY" = 1 ] && extra="$extra --real-verify"
  timeout --signal=TERM --kill-after=60 "${CAP}m" \
    apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" bash -lc "
      cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels B6_OBS_CACHE_REQUIRE_PREBUILT=1 &&
      $alloc && export TORCH_LOGS=recompiles TORCHINDUCTOR_CACHE_DIR=$INDUCTOR_CACHE DIAG_LABEL=real_$label &&
      echo \"[real_$label] PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-(unset)} deterministic=on ddp=on(1 rank)\" &&
      python -m torch.distributed.run --standalone --nproc_per_node=1 revision/scripts/b6/diag_gpu_memory.py \
        --config $CONFIG --ddp on --deterministic on --arm B6 --chain-elements 511 \
        --real-bindings $BINDINGS --real-from-draw-log $DRAW_LOG --last $LAST $extra \
        --max-minutes $MAX_MINUTES --out $OUT/real_$label.json
    "
}

declare -A RCS
JSONS=()
for label in $CONFIGS; do
  echo "===== $label start $(date +%T)"
  rc=0
  run_config "$label" 2>&1 | tee "$OUT_HOST/real_$label.log" || rc=$?   # pipefail: rc is run_config's
  RCS[$label]=$rc
  JSONS+=("$OUT/real_$label.json")
  echo "===== $label end $(date +%T) launcher rc=$rc (informational; see CLASS below)"
done

echo "===== summary (job $JOB; classification from each JSON's status)"
for label in $CONFIGS; do echo "$label launcher_rc=${RCS[$label]}"; done
SUMRC=0
apptainer exec "${BINDS[@]}" "$SIF" bash -lc "
  cd /work/repo && PYTHONPATH=/work/repo/src python revision/scripts/b6/diag_gpu_memory.py --summarize ${JSONS[*]}
" || SUMRC=$?
echo "driver_peak_mem_mib: $(awk -F', ' '{if($3+0>m)m=$3+0} END{print m+0}' "$MEMLOG")"
echo "outputs: $OUT_HOST"
[ "$SUMRC" = 0 ] || exit 1
exit 0
