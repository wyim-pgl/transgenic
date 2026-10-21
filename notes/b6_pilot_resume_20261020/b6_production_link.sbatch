#!/bin/bash
#SBATCH --job-name=b6-production-history
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@900
#SBATCH --exclude=gh062,gh093
#SBATCH --no-requeue
#SBATCH --output=%x-%j.out

# Sourceable only for exercising the exact supervisor with real local processes.
_descendants() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do echo "$p"; _descendants "$p"; done; }
_ranks() {
  local p pp c
  for p in $(_descendants "$TRAIN_PID"); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in *b6_production_link.py\ run*) ;; *) continue;; esac
    case "$c" in *accelerate*|*"bash -c"*) continue;; esac
    pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    case "$(tr '\0' ' ' < "/proc/$pp/cmdline" 2>/dev/null)" in *accelerate*) echo "$p";; esac
  done
}
launch_and_wait() {
  local TRAIN_PID TRAIN_RC=0 R
  "$@" &
  TRAIN_PID=$!
  trap 'trap "" USR1; R=$(_ranks); echo "USR1: SIGTERM to trainer ranks: ${R:-NONE FOUND}"; for p in $R; do kill -TERM "$p" 2>/dev/null || true; done' USR1
  while :; do
    if wait "$TRAIN_PID"; then TRAIN_RC=0; break; else TRAIN_RC=$?; fi
    kill -0 "$TRAIN_PID" 2>/dev/null || break
  done
  trap - USR1
  return "$TRAIN_RC"
}
sanitize_environment() {
 local name
 for name in $(compgen -v); do
 case "$name" in SBATCH_*|APPTAINER_*|SINGULARITY_*|APPTAINERENV_*|SINGULARITYENV_*|TORCHINDUCTOR_*|TORCHDYNAMO_*|TORCH_COMPILE_*|TRITON_*|TRANSGENIC_*|PYTHON*|TORCH_HOME|XDG_CACHE_HOME|TORCH_ALLOW_TF32_CUBLAS_OVERRIDE|NVIDIA_TF32_OVERRIDE) unset "$name";; esac
done
}
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then return 0; fi
set -euo pipefail
: "${WORK:?approved WORK required}" "${SIF:?pinned SIF required}" "${REQUEST:?relative request required}"
: "${REQUEST_SHA256:?request digest required}" "${REPO_DIR:?pinned source required}" "${OPS_DIR:?pinned helpers required}"
[ "${SLURM_GPUS_ON_NODE:-0}" = 4 ]
[ "${SLURM_RESTART_COUNT:-0}" = 0 ]
[ "$SIF" = /work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif ]
[ -s "$SIF" ]
command -v apptainer >/dev/null 2>&1 || module load apptainer
# sbatch arguments/resource validation belongs to the controller; inherited options
# cannot propagate from this allocation to its command environment.
sanitize_environment
(cd "$OPS_DIR" && sha256sum --quiet -c manifest.sha256)
(cd "$REPO_DIR" && sha256sum --quiet -c validation_sources.sha256)
ROUTES=$(python3 "$OPS_DIR/b6_production_link.py" paths --work "$WORK" --request "$REQUEST" --request-sha256 "$REQUEST_SHA256")
mapfile -t ROUTE <<< "$ROUTES"
[ "${#ROUTE[@]}" = 4 ]
[ "${ROUTE[0]}" = "$REPO_DIR" ] && [ "${ROUTE[1]}" = "$OPS_DIR" ]
RUN_HOST=${ROUTE[2]}
EXPECTED_IMAGE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image_sha256"])' "$WORK/$REQUEST")
ACTUAL_IMAGE=$(sha256sum "$SIF")
[ "${ACTUAL_IMAGE%% *}" = "$EXPECTED_IMAGE" ]
OUTPUT=${ROUTE[3]}
python3 "$REPO_DIR/revision/scripts/b6/storage_preflight.py" "$RUN_HOST"
mkdir -p "$(dirname "$OUTPUT")"
mkdir "$OUTPUT"
mkdir -p "$WORK/compiler_cache/b6_history_v1/inductor" "$WORK/compiler_cache/b6_history_v1/triton"
BINDS=(--bind "$WORK":/work --bind "$REPO_DIR":/work/repo:ro --bind "$OPS_DIR":/probe:ro
 --bind "$REPO_DIR":"/work/${REPO_DIR#"$WORK"/}":ro --bind "$OPS_DIR":"/work/${OPS_DIR#"$WORK"/}":ro
 --bind /work/hdd/bilv/wcyim/transgenic/b6_prep_20260925:/hdd_prep:ro
 --bind "$WORK/b6_prep_20260925":/work/b6_prep_20260925:ro
 --bind "$WORK/genomes":/work/genomes:ro
 --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro)
# --cleanenv plus explicit values binds the container runtime to the proved defaults.
ENVIRONMENT=(--cleanenv
 --env CUBLAS_WORKSPACE_CONFIG=:4096:8 --env TRITON_LIBCUDA_PATH=/.singularity.d/libs
 --env HF_HUB_OFFLINE=1 --env TRANSFORMERS_OFFLINE=1 --env NCCL_DEBUG=WARN --env OMP_NUM_THREADS=8
 --env PYTHONDONTWRITEBYTECODE=1 --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
 --env TORCH_LOGS=recompiles --env B6_OBS_CACHE_REQUIRE_PREBUILT=1
 --env TORCHINDUCTOR_CACHE_DIR=/work/compiler_cache/b6_history_v1/inductor
 --env TRITON_CACHE_DIR=/work/compiler_cache/b6_history_v1/triton
 --env PYTHONPATH=/work/repo/src:/work/repo --env HF_HOME=/work/repo/HFmodels
 --env SLURM_JOB_ID="$SLURM_JOB_ID")
COMMON=(--work /work --request "$REQUEST" --request-sha256 "$REQUEST_SHA256")
exec_python() {
 apptainer exec --nv --writable-tmpfs "${ENVIRONMENT[@]}" "${BINDS[@]}" "$SIF" \
  bash -c 'cd /work/repo; exec python "$@"' _ "$@"
}
exec_python /probe/b6_production_link.py preflight "${COMMON[@]}" > "$OUTPUT/preflight.log" 2>&1
TRAIN_RC=0
launch_and_wait apptainer exec --nv --writable-tmpfs "${ENVIRONMENT[@]}" "${BINDS[@]}" "$SIF" \
 bash -c 'cd /work/repo; exec accelerate launch --num_processes=4 --mixed_precision=bf16 /probe/b6_production_link.py run "$@"' _ \
 "${COMMON[@]}" > "$OUTPUT/train.log" 2>&1 || TRAIN_RC=$?
if [ "$TRAIN_RC" -ne 0 ]; then
 touch "$RUN_HOST/TRAINING_FAILED"
 echo "B6 trainer/launcher failed: $TRAIN_RC" >&2
 exit "$TRAIN_RC"
fi
exec_python /probe/b6_production_link.py postflight "${COMMON[@]}" --trainer-exit "$TRAIN_RC" > "$OUTPUT/postflight.log" 2>&1
