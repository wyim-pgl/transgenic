#!/bin/bash
#SBATCH --job-name=c2-history-validation
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/b6_prep_20260925/logs/%x-%j.out
set -euo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic
REPO_DIR=${REPO_DIR:?immutable reviewed source required}
OPS_DIR=${OPS_DIR:?immutable reviewed helper directory required}
PROOF_JOB_ID=${PROOF_JOB_ID:?completed initial proof job required}
[[ "$PROOF_JOB_ID" =~ ^[0-9]+$ ]]
PRIOR_HOST=$WORK/runs/c2_compile_history_proof/$PROOF_JOB_ID
PRIOR=/work/runs/c2_compile_history_proof/$PROOF_JOB_ID
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
ROOT_HOST=$WORK/runs/c2_compile_history_validation/$SLURM_JOB_ID
ROOT=/work/runs/c2_compile_history_validation/$SLURM_JOB_ID
ORIGINAL=/work/runs/b6_wp5_pilot_C2
BINDINGS=/work/b6_prep_20260925/bindings/train_v1.deltaai.json
command -v apptainer >/dev/null 2>&1 || module load apptainer
[ "${SLURM_GPUS_ON_NODE:-0}" = 4 ]
(cd "$OPS_DIR" && sha256sum --quiet -c manifest.sha256)
(cd "$REPO_DIR" && sha256sum --quiet -c validation_sources.sha256)
prior_status=$(sacct -X -n -P -j "$PROOF_JOB_ID" --format=JobIDRaw,State,ExitCode)
[[ "$prior_status" == "$PROOF_JOB_ID|COMPLETED|0:0" ]]
python3 "$REPO_DIR/revision/scripts/b6/storage_preflight.py" "$WORK/runs/b6_wp5_pilot_C2"
mkdir -p "$(dirname "$ROOT_HOST")"
mkdir "$ROOT_HOST"
# One operational, rank-shared cache namespace retained for subsequent verified
# recovery links. All FX/AOT cache options stay at the actual runtime defaults.
mkdir -p "$WORK/compiler_cache/c2_history_v1/inductor" "$WORK/compiler_cache/c2_history_v1/triton"
BINDS=(--bind "$WORK":/work --bind "$OPS_DIR":/probe:ro
 --bind "$REPO_DIR":/work/repo:ro
 --bind "$PRIOR_HOST":"$PRIOR":ro
 --bind "$WORK/runs/b6_wp5_pilot_C2":/work/runs/b6_wp5_pilot_C2:ro
 --bind /work/hdd/bilv/wcyim/transgenic/b6_prep_20260925:/hdd_prep:ro
 --bind "$WORK/b6_prep_20260925":/work/b6_prep_20260925:ro
 --bind "$WORK/genomes":/work/genomes:ro
 --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro)
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINERENV_TORCH_LOGS=recompiles APPTAINERENV_B6_OBS_CACHE_REQUIRE_PREBUILT=1
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/work/compiler_cache/c2_history_v1/inductor
export APPTAINERENV_TRITON_CACHE_DIR=/work/compiler_cache/c2_history_v1/triton
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
unset APPTAINERENV_TRANSGENIC_NO_COMPILE APPTAINERENV_TRANSGENIC_NO_GRAD_CKPT APPTAINERENV_TRANSGENIC_CKPT_REENTRANT
exec_python() {
 apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" \
  bash -c 'cd /work/repo; export PYTHONPATH=/work/repo/src:/work/repo HF_HOME=/work/repo/HFmodels; exec python "$@"' _ "$@"
}
exec_python /probe/history_proof.py check-prior --root "$PRIOR" --job-id "$PROOF_JOB_ID"
echo "NGC_TEST_START $(date -Is)"
exec_python -m unittest tests.test_b6_replay_state tests.test_b6_compile_history tests.test_b6_replay_integration -v \
 > "$ROOT_HOST/ngc_tests.log" 2>&1
exec_python /probe/test_history_proof.py > "$ROOT_HOST/harness_tests.log" 2>&1
echo "VERIFY_START $(date -Is)"
exec_python /probe/probe.py verify --bindings "$BINDINGS" --source "$ORIGINAL" --record "$ROOT/verified_record.json" \
 > "$ROOT_HOST/verify.log" 2>&1
exec_python /probe/probe.py clone --source "$ORIGINAL" --target "$ROOT/continuous"
exec_python /probe/history_proof.py manifest --run "$ROOT/continuous" \
 --receipt /probe/original_checkpoint_receipt.json --out "$ROOT/bootstrap.json"
for phase in continuous resumed; do
 extra=()
 if [ "$phase" = continuous ]; then
  extra=(--bootstrap "$ROOT/bootstrap.json" --snapshot "$ROOT/at400")
 else
  exec_python /probe/probe.py clone --source "$ROOT/at400" --target "$ROOT/resumed"
 fi
 echo "PHASE_START $phase $(date -Is)"
 apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" \
  bash -c 'cd /work/repo; export PYTHONPATH=/work/repo/src:/work/repo HF_HOME=/work/repo/HFmodels; exec accelerate launch --num_processes=4 --mixed_precision=bf16 /probe/history_proof.py "$@"' _ \
  run --bindings "$BINDINGS" --record "$ROOT/verified_record.json" --run "$ROOT/$phase" --stop 410 --snapshot-step 400 "${extra[@]}" \
  > "$ROOT_HOST/$phase.log" 2>&1
 echo "PHASE_FINISHED $phase $(date -Is)"
done
exec_python /probe/history_proof.py compare --root "$ROOT" --profile validate400 \
 --source-commit "$(cat "$REPO_DIR/SOURCE_COMMIT")" --predecessor-root "$PRIOR" --predecessor-job-id "$PROOF_JOB_ID" > "$ROOT_HOST/comparison.log" 2>&1
(cd "$REPO_DIR" && sha256sum --quiet -c validation_sources.sha256)
(cd "$OPS_DIR" && sha256sum --quiet -c manifest.sha256)
echo "HISTORY_VALIDATION_PASS $(date -Is)"
