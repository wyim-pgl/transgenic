#!/usr/bin/env bash
#SBATCH --job-name=b6_blastx_remaining8
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-7%2
#SBATCH --no-requeue
set -euo pipefail
R=/data/gpfs/assoc/pgl/data/Transgenic
W=$R/worktrees/b6-est-blastx-950cf4c
BASE=$R/evidence/est_blastx/remaining8_20260915
L=$R/evidence/est_blastx/preflight_20260914
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python
DIAMOND=/data/gpfs/assoc/pgl/bin/conda/conda_envs/evidencemodeler/bin/diamond
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=16
[[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-7]$ ]] || exit 2
cd "$BASE"
sha256sum -c inputs.sha256
SP=$("$PY" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert len(r)==8; print(r[int(sys.argv[2])]["species"])' "$BASE/inputs.json" "$SLURM_ARRAY_TASK_ID")
case "$SP" in Athaliana|Bdistachyon|Gmax|Osativa|Ppatens|Ptrichocarpa|Sbicolor|Vvinifera) ;; *) exit 2;; esac
D=$BASE/$SP
mkdir "$D"
trap 'rc=$?; printf "%s\n" "$rc" > "$D/batch.exitcode"' EXIT
mkdir "$D/tmp"
printf '%s\n' '5fa32fe8e2131aa469f9980b733eaab81eca5d778a72715119bea7b967d4e291  '"$W/revision/scripts/evidence/est_blastx.py" '32ea1a2e86a8c8d7cb33fa1fcc7d71842b9cdc71d072b5c13276611909d90302  '"$W/revision/scripts/64_orthodb_filter.py" '0fb259d705815d44f6c667a976fb71db00ec415afc3cba9938f088b90bfed07d  '"$DIAMOND" | sha256sum -c -
Q=$R/evidence/est/$SP/univec/est.univec.fa.gz
QSHA=$("$PY" - "$BASE/inputs.json" "$SLURM_ARRAY_TASK_ID" <<'PY'
import hashlib,json,sys
from pathlib import Path
row=json.loads(Path(sys.argv[1]).read_text())[int(sys.argv[2])]
q=Path(row['query']); md5=hashlib.md5(); sha=hashlib.sha256()
with q.open('rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):md5.update(b);sha.update(b)
assert md5.hexdigest()==row['query_md5'],'A37 primary EST MD5 mismatch'
s=q.parent/'summary.json'
assert hashlib.sha256(s.read_bytes()).hexdigest()==row['summary_sha256']
assert json.loads(s.read_text())['records_out']==row['records']
p=Path('/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_align')/row['species']/'PROVENANCE.txt'
assert hashlib.sha256(p.read_bytes()).hexdigest()==row['receipt_sha256'],'A37 receipt changed'
print(sha.hexdigest())
PY
)
printf '%s  %s\n' "$QSHA" "$Q" > "$D/query.sha256"
COMMON=(--species "$SP" --query "$Q" --query-sha256 "$QSHA" --roles "$R/evidence/DATASET_ROLES.est_v1.tsv" --roles-sha256 946eeeff0ac3f7c2747cdb2ff144e0eced4578a5df5d4f6f18aaf6c6e53f3ef2 --diamond "$DIAMOND" --threads 16 --tmpdir "$D/tmp")
/usr/bin/time -v -o "$D/swissprot.time.txt" "$PY" "$W/revision/scripts/evidence/est_blastx.py" run "${COMMON[@]}" \
 --source-kind swissprot --source "$R/evidence/protein/diamond/swissprot_viridiplantae_reviewed.fa.gz" --source-sha256 8d66a18e2ad350fd640f1fb97a180a8091c544efd33351e5f451bb03e11379c9 \
 --lineage "$L/swissprot/lineage.tsv" --lineage-sha256 402bca970569cc2c0192ba185c12c8fcaed99ecafd71f46a5e19f4f2789615ce --out-dir "$D/swissprot"
/usr/bin/time -v -o "$D/orthodb.time.txt" "$PY" "$W/revision/scripts/evidence/est_blastx.py" run "${COMMON[@]}" \
 --source-kind orthodb --source "$R/evidence/protein/odb12_Viridiplantae.filtered.fa.gz" --source-sha256 fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f \
 --lineage "$L/orthodb/lineage.tsv" --lineage-sha256 812449ea6b4c8d21b095efeb1767bd23e6622d07e781768d84b718c0bf2c71d5 --out-dir "$D/orthodb"
printf '%s: Swiss-Prot and OrthoDB raw searches finished; acceptance/phase projection remain downstream.\n' "$SP"
