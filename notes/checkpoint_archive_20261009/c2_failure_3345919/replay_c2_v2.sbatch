#!/bin/bash
#SBATCH --job-name=c2-recompile-diag2
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/b6_prep_20260925/logs/c2-recompile-%j.out
set -euo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic
REPO=$WORK/repo_b6_c2diag_233574a_20261011
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
OUT=$WORK/runs/c2_recompile_diag/${SLURM_JOB_ID:?}
mkdir -p "$OUT"
(cd "$REPO" && sha256sum -c c2_diag_sources_v2.sha256)
command -v apptainer >/dev/null 2>&1 || module load apptainer
BINDS=(--bind "$WORK":/work --bind "$REPO":/work/repo:ro
 --bind /work/hdd/bilv/wcyim/transgenic/b6_prep_20260925:/hdd_prep:ro
 --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro)
export APPTAINERENV_CUBLAS_WORKSPACE_CONFIG=:4096:8
export APPTAINERENV_TRITON_LIBCUDA_PATH=/.singularity.d/libs
export APPTAINERENV_HF_HUB_OFFLINE=1 APPTAINERENV_TRANSFORMERS_OFFLINE=1
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINERENV_TORCH_LOGS=recompiles APPTAINERENV_OMP_NUM_THREADS=8
export APPTAINERENV_NCCL_DEBUG=WARN
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/work/runs/c2_recompile_diag/3360834/inductor_cache
export APPTAINERENV_DIAG_LABEL=C2_3345919_replay
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
unset APPTAINERENV_TRANSGENIC_NO_COMPILE APPTAINERENV_TRANSGENIC_NO_GRAD_CKPT APPTAINERENV_TRANSGENIC_CKPT_REENTRANT
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 2 > "$OUT/smi.csv" 2>/dev/null &
SMI=$!
trap 'kill "$SMI" 2>/dev/null || true' EXIT
set +e
timeout --signal=TERM --kill-after=60 50m apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" bash -lc "
 cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels B6_OBS_CACHE_REQUIRE_PREBUILT=1 &&
 python -m torch.distributed.run --standalone --nproc_per_node=1 revision/scripts/b6/diag_gpu_memory.py \
 --config configs/b6_400m_win_v3_wp5_pilot.json --ddp on --deterministic on --arm C2 \
 --real-bindings /work/b6_prep_20260925/bindings/train_v1.deltaai.json \
 --real-from-draw-log /work/repo/c2_replay_rank03_v2.tsv --last 10000 \
 --max-minutes 44 --out /work/runs/c2_recompile_diag/$SLURM_JOB_ID/replay.json
" > "$OUT/replay.log" 2>&1
RC=$?
set -e
printf '%s\n' "$RC" > "$OUT/launcher_rc.txt"
[ "$RC" = 0 ] || exit "$RC"
python3 - "$OUT/replay.json" <<'VERIFY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['status']=='completed',r['status']
assert len(r['samples'])==190,len(r['samples'])
assert all(s['status']=='ok' and s['draw_sha_match'] is True for s in r['samples'])
assert r['samples'][-1]['rn']==6195
print('REAL_REPLAY_COMPLETE',len(r['samples']),'graphs',r['final_compiled_graphs'])
VERIFY
