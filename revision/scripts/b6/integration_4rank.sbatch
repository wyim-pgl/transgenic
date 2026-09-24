#!/bin/bash
#SBATCH --job-name=b6-int4
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
# No --mem (MAX_TRES billing; see deploy/deltaai/train_b5.slurm).
#SBATCH --time=01:30:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=%x-%j.out
#
# Rule 15 / plan CP4: compiled four-rank integration of the B6 trainer. Fails on any Dynamo fallback, covers
# zero-track and non-zero-track batches and RC tiles, cached-generation parity against per-step re-encoding
# (greedy and beam), and resume determinism (preempt at update 3, --resume auto, byte-identical step checkpoint).
# Real B6 400M model (pretrained Hyena from $REPO_DIR/HFmodels), bf16, Inductor, non-reentrant checkpointing,
# synthetic plan-7.7 rows at the 30,720-nt tier. Run only after the CP2 memory bench has PASSED.
# Written by WP4 Writer C; NOT submitted by the writer:
#   DATA=synthetic REPO_DIR=$WORK/repo_b6 sbatch revision/scripts/b6/integration_4rank.sbatch
#   DATA=wp3-fixtures REPO_DIR=$WORK/repo_b6 sbatch revision/scripts/b6/integration_4rank.sbatch   (after the merge)
set -euo pipefail
: "${SCRATCH:=/work/nvme/bilv/$USER}"
command -v apptainer >/dev/null 2>&1 || module load apptainer
WORK=$SCRATCH/transgenic
SIF=${SIF:-$SCRATCH/containers/transgenic-ngc-26.08.sif}
REPO_DIR=${REPO_DIR:-$WORK/repo_b6}
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
[ -d "$REPO_DIR/.git" ] || { echo "REFUSED: REPO_DIR is not a git checkout: $REPO_DIR" >&2; exit 3; }
case "$REPO_DIR" in */repo|*/repo_seeds) echo "REFUSED: $REPO_DIR is a frozen seed checkout" >&2; exit 3;; esac
ls "$REPO_DIR"/HFmodels/hub/models--LongSafari--hyenadna-large-1m-seqlen-hf/snapshots/*/model.safetensors >/dev/null 2>&1 \
  || { echo "REFUSED: encoder cache missing under $REPO_DIR/HFmodels" >&2; exit 3; }
GPUS=${SLURM_GPUS_ON_NODE:-2}
case "$GPUS" in 1|2|4) ;; *) echo "REFUSED: $GPUS GPUs; tests run on 1-2 GPUs (A51), production on 4" >&2; exit 3;; esac
echo "repo: $REPO_DIR @ $(git -C "$REPO_DIR" log --oneline -1 | cut -c1-70)  image: $SIF"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
OUT=/work/runs/b6_int4/run_$SLURM_JOB_ID
for _g in $(seq 0 $((GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$_g apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()" \
    || echo "gpu warm-up failed on device $_g; the test follows anyway"
done
# DATA is required (review 3 M8). Reachable counts on 4 ranks, 2 epochs:
#   synthetic:    48 rows -> 12 micro/rank/epoch, accumulation 8 -> 2/rank -> 12 updates; n-pilot 5, stop-at 3
#   wp3-fixtures: WP3's 5 train rows -> ceil(5/4) = 2 micro/rank/epoch, accumulation 4 -> 1/rank -> 4 updates;
#                 n-pilot 3, stop-at 1 (the driver refuses unreachable values with exit 4)
DATA=${DATA:?set DATA=synthetic or DATA=wp3-fixtures}
case $DATA in
  synthetic)    DATA_ARGS="--data synthetic --tier 30720 --rows 48 --accumulation 8 --n-pilot 5 --stop-at 3";;
  wp3-fixtures) DATA_ARGS="--data wp3-fixtures --accumulation 4 --n-pilot 3 --stop-at 1";;
  *) echo "REFUSED: DATA=$DATA (synthetic|wp3-fixtures)" >&2; exit 3;;
esac
RC=0
apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
  cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels && mkdir -p $OUT &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 revision/scripts/b6/integration_4rank.py \
    --model real --config configs/b6_400m_win_v3.json $DATA_ARGS --backend inductor --mixed-precision bf16 \
    --gen-tokens 12 --out $OUT
" || RC=$?
case $RC in 0) echo "B6 4-rank integration: PASS ($WORK/runs/b6_int4/run_$SLURM_JOB_ID/report.json)";;
  2) echo "B6 4-rank integration: FAIL (see report.json)";; 4) echo "B6 4-rank integration: REFUSED counts";; *) echo "B6 4-rank integration: ERROR rc=$RC";; esac
exit $RC
