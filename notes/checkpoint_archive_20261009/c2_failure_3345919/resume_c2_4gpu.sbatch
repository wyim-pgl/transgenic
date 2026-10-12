#!/bin/bash
#SBATCH --job-name=c2-recovery-4gpu
#SBATCH --account=bilv-dtai-gh
#SBATCH --partition=ghx4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --signal=B:USR1@720
#SBATCH --exclude=gh062,gh093
#SBATCH --output=/work/nvme/bilv/wcyim/transgenic/b6_prep_20260925/logs/%x-%j.out
# Uses the exact pilot trainer/config in a verified update200 copy. Preemption is the normal chain signal.
# Submit only after validation3361110 passes. Minimum useful recovery result: update400, all four ranks saved.
set -euo pipefail
WORK=/work/nvme/bilv/wcyim/transgenic
REPO_DIR=$WORK/repo_b6_d7b1a37
SIF=/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif
RUN_HOST=$WORK/runs/b6_wp5_pilot_C2_recovery_20261011
RUN=/work/runs/b6_wp5_pilot_C2_recovery_20261011
B=/work/b6_prep_20260925/bindings/train_v1.deltaai.json
CONFIG=configs/b6_400m_win_v3_wp5_pilot.json
command -v apptainer >/dev/null 2>&1 || module load apptainer
[ "${SLURM_GPUS_ON_NODE:-0}" = 4 ] || exit 3
[ "$(cat "$REPO_DIR/SOURCE_COMMIT")" = d7b1a37 ] || exit 3
(cd "$REPO_DIR" && sha256sum -c validation_sources.sha256)
python3 "$REPO_DIR/revision/scripts/b6/storage_preflight.py" "$RUN_HOST"
python3 - "$RUN_HOST" "$WORK/runs/c2_recompile_diag/3361110/replay.json" <<'PRECHECK'
import hashlib,json,sys
from pathlib import Path
r=Path(sys.argv[1]);v=json.load(open(sys.argv[2]))
assert v['status']=='validation_complete' and v['validation']['status']=='passed'
assert v['validation']['real_losses_exact'] is True and len(v['validation']['matrix'])==108
assert not (r/'TRAINING_FAILED').exists() and not (r/'TRAINING_DONE').exists()
m=json.load(open(str(r/'latest_state/meta.json')))
assert (m['global_step'],m['epoch'],m['step'],m['pending_micro'],m['arm'])==(200,0,4800,0,'C2')
provenance=json.load(open(str(r/'RECOVERY_PROVENANCE.json')))
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
for name,record in provenance['copied_files'].items():assert sha(r/name)==record['sha256'],name
for name,record in provenance['draw_logs'].items():
 p=r/'draw_log'/name
 assert len(p.read_bytes().splitlines())==4800 and sha(p)==record['committed_prefix_sha256'],name
print('RECOVERY_PRECHECK_PASS update200 four-rank state and committed draw prefixes')
PRECHECK

BINDS=(--bind "$WORK":/work --bind "$REPO_DIR":/work/repo:ro
 --bind /work/hdd/bilv/wcyim/transgenic/b6_prep_20260925:/hdd_prep:ro
 --bind "$WORK/b6_prep_20260925/intervals":/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/intervals:ro)
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRITON_LIBCUDA_PATH=/.singularity.d/libs
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=8
export APPTAINERENV_PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export APPTAINERENV_TORCH_LOGS=recompiles APPTAINERENV_B6_OBS_CACHE_REQUIRE_PREBUILT=1
export APPTAINERENV_TORCHINDUCTOR_CACHE_DIR=/work/runs/c2_recompile_diag/3360834/inductor_cache
unset TRANSGENIC_NO_COMPILE TRANSGENIC_NO_GRAD_CKPT TRANSGENIC_CKPT_REENTRANT
unset APPTAINERENV_TRANSGENIC_NO_COMPILE APPTAINERENV_TRANSGENIC_NO_GRAD_CKPT APPTAINERENV_TRANSGENIC_CKPT_REENTRANT
nvidia-smi --query-gpu=timestamp,index,memory.used --format=csv,noheader,nounits -l 5 > "$RUN_HOST/recovery_gpu_mem.csv" 2>/dev/null &
MON_PID=$!
trap 'kill "$MON_PID" 2>/dev/null || true' EXIT
for gpu in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES=$gpu apptainer exec --nv "$SIF" python -c 'import torch; torch.ones(1).cuda(); torch.cuda.synchronize()'
done
echo "RECOVERY_START $(date -Is) job $SLURM_JOB_ID node $(hostname)"
apptainer exec --nv --writable-tmpfs "${BINDS[@]}" "$SIF" bash -lc "
 cd /work/repo && export PYTHONPATH=/work/repo/src HF_HOME=/work/repo/HFmodels &&
 accelerate launch --num_processes=4 --mixed_precision=bf16 train/train_HyenaTransgenic_B6.py \\
 --config $CONFIG --arm C2 --seed 123 --bindings $B --output-dir $RUN --resume auto \\
 --stop-after-updates 4099 --save-every-n-steps 200 --mixed-precision bf16 --deterministic \\
 >> $RUN/recovery_train.log 2>&1
" &
TRAIN_PID=$!
_descendants() { local p; for p in $(pgrep -P "$1" 2>/dev/null); do echo "$p"; _descendants "$p"; done; }
_ranks() {
 local p pp c
 for p in $(_descendants "$TRAIN_PID"); do
  c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null)
  case "$c" in *train/train_HyenaTransgenic_B6.py*) ;; *) continue;; esac
  case "$c" in *accelerate*|*"bash -lc"*) continue;; esac
  pp=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
  case "$(tr '\0' ' ' < "/proc/$pp/cmdline" 2>/dev/null)" in *accelerate*) echo "$p";; esac
 done
}
trap 'trap "" USR1; ranks=$(_ranks); echo "RECOVERY_USR1 ranks: ${ranks:-NONE}"; for p in $ranks; do kill -TERM "$p" 2>/dev/null || true; done' USR1
TRAIN_RC=0
while :; do
 if wait "$TRAIN_PID"; then TRAIN_RC=0; break; else TRAIN_RC=$?; fi
 kill -0 "$TRAIN_PID" 2>/dev/null || break
done
if [ "$TRAIN_RC" -ne 0 ]; then touch "$RUN_HOST/TRAINING_FAILED"; exit "$TRAIN_RC"; fi
python3 - "$RUN_HOST" "$SLURM_JOB_ID" <<'POSTCHECK'
import hashlib,json,sys,time
from pathlib import Path
r=Path(sys.argv[1]);m=json.load(open(str(r/'latest_state/meta.json')))
assert not (r/'TRAINING_FAILED').exists() and not (r/'MEMORY_ABORT').exists()
assert not (r/'TRAINING_DONE').exists()
assert m['arm']=='C2' and 400<=m['global_step']<4099 and m['pending_micro']==0 and m['epoch']==0,m
assert m['step']==m['global_step']*24,m
provenance=json.load(open(str(r/'RECOVERY_PROVENANCE.json')))
for rank in range(4):
 name='rank%02d.tsv'%rank
 lines=(r/'draw_log'/name).read_bytes().splitlines(keepends=True)
 assert len(lines)==m['step'],(name,len(lines),m)
 assert hashlib.sha256(b''.join(lines[:4800])).hexdigest()==provenance['draw_logs'][name]['committed_prefix_sha256']
 assert (r/'latest_state/accelerate_state'/('random_states_%d.pkl'%rank)).is_file()
result=dict(status='passed',job_id=int(sys.argv[2]),source_commit='d7b1a37',started_from_update=200,
            final_checkpoint=m,validation_job=3361110,checked_unix=time.time())
with (r/'RECOVERY_VALIDATION_PASSED.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True)
print('RECOVERY_VALIDATED '+json.dumps(result))
POSTCHECK
