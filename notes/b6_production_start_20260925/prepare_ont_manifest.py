"""Freeze the ONT training runs of the ready species for metadata-only production preparation.

Protocol comes from the author-approved ONT_PROTOCOL sheet (2026-09-25 14:44 PDT, cDNA83/direct_RNA42); UMI is
'absent' for every training run by the author decision of 2026-09-25 evening. Athaliana is left out (canonical
dataset/alignment directory mismatch and unaligned validation runs), recorded separately."""
import csv, hashlib, json
from pathlib import Path

root = Path('/data/gpfs/assoc/pgl/data/Transgenic')
prep = root / 'evidence/b6_prepare_20260925'
inputs = prep / 'inputs'
SPECIES = ('Gmax', 'Osativa', 'Ppatens', 'Ptrichocarpa', 'Sitalica', 'Vvinifera')
SHEET_SHA = 'e21425e1eeaa60ac87648464b684f61fabe06c848ec8d0de4da448d54695b029'

src = Path.home() / 'ONT_PROTOCOL.b6_v1.draft.tsv'
if hashlib.sha256(src.read_bytes()).hexdigest() != SHEET_SHA:
    raise RuntimeError('ONT protocol sheet differs from the approved snapshot')
roles_path = inputs / 'DATASET_ROLES.longread_v1.tsv'
if hashlib.sha256(roles_path.read_bytes()).hexdigest() != (inputs / 'DATASET_ROLES.longread_v1.tsv.sha256').read_text().split()[0]:
    raise RuntimeError('roles snapshot checksum differs')
roles = list(csv.DictReader(roles_path.open(), delimiter='\t'))
sheet = list(csv.DictReader(src.open(), delimiter='\t'))
audit = [r for r in csv.reader((root / 'evidence/audit_align_v1/runs.tsv').open(), delimiter='\t')]
tasks = []
for x in roles:
    if x['species'] not in SPECIES or not x['dataset'].startswith(('ont/', 'training/ont/')) or x['role'] == 'excluded':
        continue
    if x['role'] != 'c2_training_eligible':
        raise RuntimeError('non-training ONT row in a ready species: ' + x['run'])
    sp, run, dsbase = x['species'], x['run'], Path(x['dataset']).name
    s = [r for r in sheet if r['run'] == run and r['species'] == sp and r['dataset'] == x['dataset']]
    if len(s) != 1 or s[0]['role'] != x['role']:
        raise RuntimeError('sheet row is not unique or role differs: ' + run)
    s = s[0]
    if s['protocol'] not in ('cDNA', 'direct_RNA') or s['umi_status'] != 'absent' or not s['library']:
        raise RuntimeError('protocol/UMI/library not as approved: ' + run)
    a = [r for r in audit if len(r) == 8 and r[1] == 'ont' and r[2] == sp and r[3] == dsbase and r[4] == run]
    if len(a) != 1:
        raise RuntimeError('audit run list lacks a unique row: ' + run)
    _, _, _, _, _, reads, uf, primary = a[0]
    if not (root / reads).is_file() or not (root / primary / 'DONE').is_file():
        raise RuntimeError('missing frozen input: ' + run)
    tasks.append(dict(species=sp, dataset=x['dataset'], run=run, library=s['library'], protocol=s['protocol'],
                      umi_status='absent', reads=str(root / reads), primary=str(root / primary)))
if len(tasks) != 90 or len({(t['species'], t['run']) for t in tasks}) != 90:
    raise RuntimeError(f'expected exactly 90 distinct ONT training runs, got {len(tasks)}')
for name, data in [('ONT_PROTOCOL.b6_v1.draft.tsv', src.read_bytes()),
                   ('ont_tasks.json', (json.dumps(tasks, sort_keys=True, indent=2) + '\n').encode())]:
    with (inputs / name).open('xb') as f:
        f.write(data)
    with (inputs / (name + '.sha256')).open('x') as f:
        f.write(hashlib.sha256(data).hexdigest() + '  ' + name + '\n')
count = {}
for t in tasks:
    count.setdefault(t['species'], {}).setdefault(t['protocol'], 0)
    count[t['species']][t['protocol']] += 1
print(json.dumps(dict(tasks=len(tasks), by_species=count), sort_keys=True))
