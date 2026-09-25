#!/bin/bash
#SBATCH --job-name=b6_vv_meta
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%j.out
set -euo pipefail
ROOT=/data/gpfs/assoc/pgl/data/Transgenic
PREP=$ROOT/evidence/b6_prepare_20260925
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
export PYTHONPATH="$PREP/source_missing_library/src:$PREP/source_missing_library/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
"$PY" - "$ROOT" "$PREP" 8 <<'PY'
import csv, hashlib, json, sys
from pathlib import Path
import b6_c0_metadata as metadata
root, prep = map(Path, sys.argv[1:3])
species = ('Athaliana','Bdistachyon','Gmax','Osativa','Ppatens','Ptrichocarpa','Sbicolor','Sitalica','Vvinifera')
sp = species[int(sys.argv[3])]
roles = prep / 'inputs/DATASET_ROLES.est_v1.tsv'
metadata.require(hashlib.sha256(roles.read_bytes()).hexdigest() == (prep/'inputs/roles.sha256').read_text().split()[0], 'roles snapshot checksum differs')
rows = [r for r in csv.DictReader(roles.open(), delimiter='\t') if r['dataset'] == 'est/'+sp]
metadata.require(len(rows) == 1 and rows[0]['role'] == 'c2_training_eligible' and rows[0]['species'] == sp, 'EST role is not uniquely training eligible')
base = root / 'evidence/est' / sp
meta = base / 'est_esummary.tsv'
provenance = json.loads(Path(str(meta)+'.PROVENANCE.json').read_text())
metadata.require(provenance['label'] == sp and provenance['records'] == provenance['count'] > 0, 'EST summary is incomplete or wrong species')
metadata.require(provenance['output_sha256'] == metadata.verify_sha256_sidecar(meta), 'EST summary provenance checksum differs')
metadata.require(isinstance(provenance['taxid'], int) and provenance['taxid'] > 0, 'invalid taxid')
screened = base / 'univec/est.univec.fa.gz'
expected = Path(str(screened)+'.md5').read_text().split()[0]
h = hashlib.md5()
with screened.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024), b''):
        h.update(block)
metadata.require(h.hexdigest() == expected, 'screened EST checksum differs')
metadata.require(sp == 'Vvinifera', 'this retry is for Vvinifera only')
out = prep / 'metadata_est_missing_library' / sp
out.mkdir(parents=True, exist_ok=False)
receipt = dict(species=sp, screened_md5=expected, est_meta_sha256=provenance['output_sha256'],
               bioproject='NCBI_EST:txid'+str(provenance['taxid']), missing_library='NCBI-EST', scope='REAL_INPUT_PREPARATION_NOT_MODEL_TRAINING')
(out/'input_receipt.json').write_text(json.dumps(receipt, sort_keys=True, indent=2)+'\n')
rc = metadata.main(['--source','est','--species',sp,'--roles',str(roles),
                    '--screened',str(screened),'--est-meta',str(meta),
                    '--bioproject',receipt['bioproject'],'--missing-library','NCBI-EST','--out',str(out/'c0_metadata.tsv')])
if rc != 0:
    raise SystemExit(rc)
(out/'METADATA_DONE').write_text('metadata producer succeeded; C0/corpus/cache still required\n')
PY
