#!/bin/bash
#SBATCH --job-name=b6-history-validation
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=06:00:00
#SBATCH --exclude=gh062,gh093
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/b6_prep_20260925/logs/%x-%j.out
set -euo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic
REPO_DIR=${REPO_DIR:?immutable source198ce14 required}
OPS_DIR=${OPS_DIR:?immutable manifested B6 helper release required}
# Explicit existing frozen mmap path; no auto-created observation cache.
MMAP_DIR=${MMAP_DIR:?existing container-visible frozen mmap directory required}
# Metadata-only reference for conservative checkpoint-copy peak estimation.
REFERENCE_CHECKPOINT=${REFERENCE_CHECKPOINT:?existing full checkpoint directory required}
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
ROOT_HOST=$WORK/runs/b6_compile_history_validation/$SLURM_JOB_ID
ROOT=/work/runs/b6_compile_history_validation/$SLURM_JOB_ID
BINDINGS=/work/b6_prep_20260925/bindings/train_v1.deltaai.json
CONFIG=/work/repo/configs/b6_400m_win_v3_wp5_pilot.json
command -v apptainer >/dev/null 2>&1 || module load apptainer
[ "${SLURM_GPUS_ON_NODE:-0}" = 4 ]
[ "$(cat "$REPO_DIR/SOURCE_COMMIT")" = 198ce14ce53266f53b7bda5d0c4cdfd94deb43b4 ]
(cd "$OPS_DIR" && sha256sum --quiet -c manifest.sha256)
(cd "$REPO_DIR" && sha256sum --quiet -c validation_sources.sha256)
# Parent stages these dependencies unchanged in the manifested release.
[ -f "$OPS_DIR/c2/history_proof.py" ]
[ -f "$OPS_DIR/c2/probe.py" ]
mkdir -p "$(dirname "$ROOT_HOST")"
mkdir "$ROOT_HOST"
python3 "$REPO_DIR/revision/scripts/b6/storage_preflight.py" "$ROOT_HOST" > "$ROOT_HOST/storage_preflight.json"
# Seven complete checkpoint-sized copies conservatively cover each phase's
# current/previous/tmp state plus snapshot400; 2GiB allows bounded traces/logs.
# This gate is independent of the source's mandatory100GiB headroom gate.
python3 - "$ROOT_HOST" "$REFERENCE_CHECKPOINT" <<'PY'
import json, os, sys
from pathlib import Path
root, reference = map(Path, sys.argv[1:])
storage = json.loads((root / 'storage_preflight.json').read_text())
if storage['project_id'] != 22169 or storage['pool'] != 'ddn_ssd':
    raise ValueError('proof requires project22169/ddn_ssd')
required = ['meta.json', 'identity.json', 'accelerate_state/model.safetensors',
            'accelerate_state/optimizer.bin', 'accelerate_state/scheduler.bin']
required += ['accelerate_state/random_states_%d.pkl' % rank for rank in range(4)]
if not all((reference / name).is_file() for name in required):
    raise ValueError('proof-copy reference must be a complete four-rank checkpoint')
files = [p for p in reference.rglob('*') if p.is_file()]
if not files or any(p.is_symlink() for p in files):
    raise ValueError('invalid proof-copy reference files')
checkpoint_bytes = sum(p.stat().st_size for p in files)
peak = checkpoint_bytes * 7 + 2 * 1024**3
if min(storage['quota_remaining_bytes'], storage['filesystem_available_bytes']) < peak:
    raise RuntimeError('proof-copy peak does not fit')
with (root / 'proof_copy_peak.json').open('x') as stream:
    json.dump(dict(reference=str(reference), checkpoint_bytes=checkpoint_bytes, checkpoint_copies=7,
                   log_allowance_bytes=2*1024**3, proof_copy_peak_bytes=peak, storage=storage), stream, sort_keys=True)
PY
mkdir -p "$WORK/compiler_cache/b6_history_v1/inductor" "$WORK/compiler_cache/b6_history_v1/triton"
BINDS=(--bind "$WORK":/work --bind "$OPS_DIR":/probe:ro
 --bind "$REPO_DIR":/work/repo:ro --bind "$SIF":/runtime.sif:ro
 --bind /work/hdd/bilv/wcyim/transgenic/b6_prep_20260925:/hdd_prep:ro
 --bind "$WORK/b6_prep_20260925":/work/b6_prep_20260925:ro
 --bind "$WORK/genomes":/work/genomes:ro
 --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro)
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINERENV_TORCH_LOGS=recompiles APPTAINERENV_B6_OBS_CACHE_REQUIRE_PREBUILT=1
export APPTAINERENV_B6_OBS_CACHE_MMAP_DIR="$MMAP_DIR"
export APPTAINERENV_B6_C2_HELPER_DIR=/probe/c2
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/work/compiler_cache/b6_history_v1/inductor
export APPTAINERENV_TRITON_CACHE_DIR=/work/compiler_cache/b6_history_v1/triton
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
unset APPTAINERENV_TRANSGENIC_NO_COMPILE APPTAINERENV_TRANSGENIC_NO_GRAD_CKPT APPTAINERENV_TRANSGENIC_CKPT_REENTRANT
exec_python() {
 apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" \
  bash -c 'cd /work/repo; export PYTHONPATH=/work/repo/src:/work/repo HF_HOME=/work/repo/HFmodels; exec python "$@"' _ "$@"
}
exec_python -m unittest tests.test_b6_replay_state tests.test_b6_compile_history tests.test_b6_replay_integration -v \
 > "$ROOT_HOST/ngc_tests.log" 2>&1
exec_python /probe/test_b6_receipt_contract.py -v > "$ROOT_HOST/contract_tests.log" 2>&1
exec_python /probe/test_b6_history_proof.py -v > "$ROOT_HOST/harness_tests.log" 2>&1
exec_python /probe/b6_history_proof.py verify --bindings "$BINDINGS" --record "$ROOT/verified_record.json" \
 --config "$CONFIG" --mmap-dir "$MMAP_DIR" --image /runtime.sif \
 --source-manifest /work/repo/validation_sources.sha256 --helper-manifest /probe/manifest.sha256 \
 > "$ROOT_HOST/verify.log" 2>&1
exec_python /probe/b6_history_proof.py init-fresh --root "$ROOT"
for phase in continuous resumed; do
 extra=()
 if [ "$phase" = continuous ]; then
  extra=(--fresh --snapshot "$ROOT/at400")
 else
  exec_python /probe/b6_history_proof.py clone --source "$ROOT/at400" --target "$ROOT/resumed"
 fi
 apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" \
  bash -c 'cd /work/repo; export PYTHONPATH=/work/repo/src:/work/repo HF_HOME=/work/repo/HFmodels; exec accelerate launch --num_processes=4 --mixed_precision=bf16 /probe/b6_history_proof.py "$@"' _ \
  run --config "$CONFIG" --bindings "$BINDINGS" --record "$ROOT/verified_record.json" \
  --run "$ROOT/$phase" --stop 410 "${extra[@]}" > "$ROOT_HOST/$phase.log" 2>&1
done
# Detect source/helper drift before any READY receipt is written.
(cd "$REPO_DIR" && sha256sum --quiet -c validation_sources.sha256)
(cd "$OPS_DIR" && sha256sum --quiet -c manifest.sha256)
exec_python /probe/b6_history_proof.py compare --root "$ROOT" > "$ROOT_HOST/comparison.log" 2>&1
exec_python /probe/b6_history_proof.py validate-receipt --root "$ROOT" >> "$ROOT_HOST/comparison.log" 2>&1
echo "B6_HISTORY_VALIDATION_PASS $(date -Is)"
