#!/bin/bash
#SBATCH --job-name=c2-first-forward
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=01:30:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/b6_prep_20260925/logs/%x-%j.out
set -euo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic
OPS=$WORK/ops/c2_first_forward_20261012
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
ROOT_HOST=$WORK/runs/c2_first_forward/$SLURM_JOB_ID
ROOT=/work/runs/c2_first_forward/$SLURM_JOB_ID
EVIDENCE=/work/runs/c2_resume_determinism/3361666
B=/work/b6_prep_20260925/bindings/train_v1.deltaai.json
command -v apptainer >/dev/null 2>&1 || module load apptainer
[ "${SLURM_GPUS_ON_NODE:-0}" = 4 ]
(cd "$OPS" && sha256sum --quiet -c manifest.sha256)
(cd "$WORK/repo_b6_d7b1a37" && sha256sum --quiet -c validation_sources.sha256)
python3 "$WORK/repo_b6_d7b1a37/revision/scripts/b6/storage_preflight.py" "$WORK/runs/b6_wp5_pilot_C2"
mkdir -p "$(dirname "$ROOT_HOST")"
mkdir "$ROOT_HOST"
BINDS=(--bind "$WORK":/work --bind "$OPS":/probe:ro
 --bind "$WORK/repo_b6_d7b1a37":/work/repo:ro
 --bind "$WORK/runs/b6_wp5_pilot_C2":/work/runs/b6_wp5_pilot_C2:ro
 --bind "$WORK/runs/c2_resume_determinism/3361666":/work/runs/c2_resume_determinism/3361666:ro
 --bind /work/hdd/bilv/wcyim/transgenic/b6_prep_20260925:/hdd_prep:ro
 --bind "$WORK/b6_prep_20260925":/work/b6_prep_20260925:ro
 --bind "$WORK/genomes":/work/genomes:ro
 --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro)
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINERENV_TORCH_LOGS=recompiles APPTAINERENV_B6_OBS_CACHE_REQUIRE_PREBUILT=1
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
unset APPTAINERENV_TRANSGENIC_NO_COMPILE APPTAINERENV_TRANSGENIC_NO_GRAD_CKPT APPTAINERENV_TRANSGENIC_CKPT_REENTRANT
exec_python() {
 apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" \
  bash -c 'cd /work/repo; export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels; exec python "$@"' _ "$@"
}
echo "VERIFY_START $(date -Is)"
exec_python /probe/probe.py verify --bindings "$B" --source /work/runs/b6_wp5_pilot_C2 --record "$ROOT/verified_record.json" > "$ROOT_HOST/verify.log" 2>&1
for phase in cold primed primed_reset; do
 exec_python /probe/probe.py clone --source "$EVIDENCE/B210" --target "$ROOT/$phase"
 mkdir "$ROOT_HOST/cache_$phase" "$ROOT_HOST/triton_$phase"
 export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=$ROOT/cache_$phase
 export APPTAINERENV_TRITON_CACHE_DIR=$ROOT/triton_$phase
 echo "PHASE_START $phase $(date -Is)"
 apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" \
  bash -c 'cd /work/repo; export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels; exec accelerate launch --num_processes=4 --mixed_precision=bf16 /probe/first_forward.py "$@"' _ \
  run --bindings "$B" --record "$ROOT/verified_record.json" --evidence "$EVIDENCE" --run "$ROOT/$phase" --case "$phase" \
  > "$ROOT_HOST/$phase.log" 2>&1
 echo "PHASE_FINISHED $phase $(date -Is)"
done
exec_python /probe/first_forward.py compare --root "$ROOT" > "$ROOT_HOST/comparison.log" 2>&1
(cd "$WORK/repo_b6_d7b1a37" && sha256sum --quiet -c validation_sources.sha256)
echo "FIRST_FORWARD_OBSERVATIONS_COMPLETE $(date -Is)"
