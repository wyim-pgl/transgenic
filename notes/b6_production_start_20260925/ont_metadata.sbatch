#!/bin/bash
#SBATCH --job-name=b6_ont_meta
#SBATCH --account=cpu-s1-pgl-0
#SBATCH --partition=cpu-s1-pgl-0
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-89%4
#SBATCH --output=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/logs/%x_%A_%a.out
set -euo pipefail
PREP=/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925
PY=/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python
export PYTHONPATH="$PREP/source/src:$PREP/source/revision/scripts/evidence"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
"$PY" - "$PREP" "${SLURM_ARRAY_TASK_ID:?}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import b6_c0_metadata as metadata
prep=Path(sys.argv[1]); inputs=prep/'inputs'
for name in ('ont_tasks.json','DATASET_ROLES.longread_v1.tsv','ONT_PROTOCOL.b6_v1.draft.tsv'):
    expected=(inputs/(name+'.sha256')).read_text().split()[0]
    metadata.require(hashlib.sha256((inputs/name).read_bytes()).hexdigest()==expected, name+' snapshot checksum differs')
tasks=json.loads((inputs/'ont_tasks.json').read_text())
metadata.require(len(tasks)==90,'expected exactly 90 approved ONT training runs')
task=tasks[int(sys.argv[2])]
reads=Path(task['reads']); primary=Path(task['primary'])
metadata.require((primary/'DONE').is_file(),'primary alignment has no DONE')
provenance=dict(line.split(None,1) for line in (primary/'PROVENANCE.txt').read_text().splitlines() if len(line.split(None,1))==2)
metadata.require(Path(provenance['reads']).resolve()==reads.resolve(),'reads path differs from primary alignment')
h=hashlib.md5()
with reads.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
metadata.require(h.hexdigest()==provenance['reads_md5'],'reads checksum differs from primary alignment')
out=prep/'metadata/ont'/task['species']/Path(task['dataset']).name/task['run']
out.mkdir(parents=True,exist_ok=False)
receipt=dict(task,reads_md5=h.hexdigest(),scope='REAL_INPUT_PREPARATION_NOT_MODEL_TRAINING')
(out/'input_receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
metadata.require(task['umi_status']=='absent' and task['protocol'] in ('cDNA','direct_RNA'),'protocol/UMI not as approved')
rc=metadata.main(['--source','ont','--species',task['species'],'--roles',str(inputs/'DATASET_ROLES.longread_v1.tsv'),
                  '--dataset',task['dataset'],'--run',task['run'],'--library',task['library'],
                  '--reads',str(reads),'--protocol',task['protocol'],'--umi-status',task['umi_status'],'--out',str(out/'c0_metadata.tsv')])
if rc: raise SystemExit(rc)
(out/'METADATA_DONE').write_text('metadata producer succeeded; C0/corpus/cache still required\n')
PY
