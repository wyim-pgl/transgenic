#!/bin/bash
#SBATCH --job-name=b6-mem129k
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=64
# No --mem: ghx4 bills MAX_TRES and DefMemPerGPU already scales with --gpus-per-node (deploy/deltaai/train_b5.slurm).
#SBATCH --time=01:00:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=%x-%j.out
#
# B6 CP2 memory gate (plan B6_WP2_WP4_PLAN_v4 section 10; A46 rule 15): 4 ranks, compile on, non-reentrant
# checkpointing, 129,024 nt, dense K = 31, 511 chain elements, worst-case 8,192-token target. PASS = every rank's
# peak reserved <= 100 GB and no Dynamo fallback / post-warm-up recompile. A FAIL means an amendment (plan 10).
# Written by WP4 Writer C; NOT submitted by the writer. Submit only after the author's go-ahead:
#   REPO_DIR=$WORK/repo_b6 sbatch revision/scripts/b6/bench_memory_129k.sbatch
set -euo pipefail
: "${SCRATCH:=/work/nvme/bilv/$USER}"
command -v apptainer >/dev/null 2>&1 || module load apptainer
WORK=$SCRATCH/transgenic
SIF=${SIF:-$SCRATCH/containers/transgenic-ngc-26.08.sif}
# Never the frozen seed checkouts (/work/repo, repo_seeds): the bench runs from a separate B6 clone.
REPO_DIR=${REPO_DIR:-$WORK/repo_b6}
[ -s "$SIF" ] || { echo "REFUSED: SIF not found: $SIF" >&2; exit 3; }
[ -d "$REPO_DIR/.git" ] || { echo "REFUSED: REPO_DIR is not a git checkout: $REPO_DIR" >&2; exit 3; }
case "$REPO_DIR" in */repo|*/repo_seeds) echo "REFUSED: $REPO_DIR is a frozen seed checkout" >&2; exit 3;; esac
ls "$REPO_DIR"/HFmodels/hub/models--LongSafari--hyenadna-large-1m-seqlen-hf/snapshots/*/model.safetensors >/dev/null 2>&1 \
  || { echo "REFUSED: encoder cache missing under $REPO_DIR/HFmodels" >&2; exit 3; }
GPUS=${SLURM_GPUS_ON_NODE:-2}
case "$GPUS" in 1|2|4) ;; *) echo "REFUSED: $GPUS GPUs; tests run on 1 (pre-check) or 2 (gate, A51), production on 4" >&2; exit 3;; esac
echo "repo: $REPO_DIR @ $(git -C "$REPO_DIR" log --oneline -1 | cut -c1-70)  image: $SIF"
apptainer exec "$SIF" cat /opt/transgenic/IMAGE_INFO 2>/dev/null || echo "image: (no IMAGE_INFO)"
export TRITON_LIBCUDA_PATH=/.singularity.d/libs HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
OUT=$WORK/runs/b6_bench; mkdir -p "$OUT"
MEMLOG=$OUT/mem129k-smi-$SLURM_JOB_ID.csv
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -l 2 > "$MEMLOG" 2>/dev/null &
SMI=$!
# First-kernel launch failures on fresh GH200 allocations (train_b5.slurm, 2026-09-09): warm each GPU first.
for _g in $(seq 0 $((GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$_g apptainer exec --nv "$SIF" python -c "import torch; torch.ones(1).cuda(); torch.cuda.synchronize()" \
    || echo "gpu warm-up failed on device $_g; the bench follows anyway"
done
RC=0
apptainer exec --nv --writable-tmpfs --bind "$WORK":/work --bind "$REPO_DIR":/work/repo "$SIF" bash -lc "
  cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels &&
  accelerate launch --num_processes=$GPUS --mixed_precision=bf16 revision/scripts/b6/bench_memory_129k.py \
    --config configs/b6_400m_win_v3.json --tier 129024 --chain-elements 511 --steps ${STEPS:-12} --warmup 4 \
    --gate-gb 100 --out /work/runs/b6_bench/mem129k-$SLURM_JOB_ID.json
" || RC=$?
kill $SMI 2>/dev/null || true
echo "driver_peak_mem_mib_per_gpu: $(awk -F', ' '{if($2+0>m[$1])m[$1]=$2+0} END{for(i in m) printf "gpu%s=%d ", i, m[i]}' "$MEMLOG")"
case $RC in 0) echo "B6 memory gate: PASS";; 2) echo "B6 memory gate: FAIL (amendment required, plan 10 CP2)";;
  4) echo "B6 memory bench: REFUSED arguments (steps > warmup >= 1)";;
  3) echo "B6 memory bench: NOT_A_GATE_RUN (settings differ from the frozen gate; certifies nothing)";;
  *) echo "B6 memory bench: ERROR rc=$RC";; esac
exit $RC
