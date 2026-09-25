"""Freeze the 49 approved PacBio run/library pairs for metadata-only production preparation."""
import csv, hashlib, json
from pathlib import Path

root=Path('/data/gpfs/assoc/pgl/data/Transgenic')
prep=root/'evidence/b6_prepare_20260925'
inputs=prep/'inputs'
sheet=inputs/'PACBIO_LIBRARY.b6_v1.draft.tsv'
if hashlib.sha256(sheet.read_bytes()).hexdigest()!='8de4d5830ff1a88587d838c3521a64f2b62803c2f0c4e4d717a401ce102e4dce':
    raise RuntimeError('library sheet differs from A49.1 keyed snapshot')
roles_data=(root/'evidence/DATASET_ROLES.longread_v1.tsv').read_bytes()
roles=list(csv.DictReader(roles_data.decode().splitlines(),delimiter='\t'))
libraries=list(csv.DictReader(sheet.open(),delimiter='\t'))
rows=list(csv.reader((root/'evidence/audit_align_v1/runs.tsv').open(),delimiter='\t'))
tasks=[]
for row in rows:
    if len(row)!=8: raise ValueError('audit run list is not the reviewed 8-column format')
    _,source,sp,dsbase,run,reads,uf,primary=row
    if source!='pacbio': continue
    match=[r for r in roles if r['run']==run and r['species']==sp and r['role']=='c2_training_eligible']
    lib=[r for r in libraries if r['run']==run and r['species']==sp and r['role']=='c2_training_eligible']
    if len(match)!=1 or len(lib)!=1 or match[0]['dataset']!=lib[0]['dataset']:
        raise RuntimeError('ambiguous role/library for '+run)
    if Path(match[0]['dataset']).name!=dsbase or not lib[0]['library']:
        raise RuntimeError('missing library or dataset mismatch: '+run)
    if not (root/reads).is_file() or not (root/primary/'DONE').is_file():
        raise RuntimeError('missing frozen input: '+run)
    tasks.append(dict(species=sp,dataset=match[0]['dataset'],run=run,library=lib[0]['library'],
                      reads=str(root/reads),primary=str(root/primary)))
if len(tasks)!=49 or len({t['run'] for t in tasks})!=49:
    raise RuntimeError('expected exactly 49 distinct approved PacBio runs')
for name,data in [('DATASET_ROLES.longread_v1.tsv',roles_data),
                  ('pacbio_tasks.json',(json.dumps(tasks,sort_keys=True,indent=2)+'\n').encode())]:
    with (inputs/name).open('xb') as f: f.write(data)
for name in ('DATASET_ROLES.longread_v1.tsv','pacbio_tasks.json','PACBIO_LIBRARY.b6_v1.draft.tsv'):
    data=(inputs/name).read_bytes()
    with (inputs/(name+'.sha256')).open('x') as f:
        f.write(hashlib.sha256(data).hexdigest()+'  '+name+'\n')
print(json.dumps(dict(tasks=len(tasks),species=sorted({t['species'] for t in tasks}))))
