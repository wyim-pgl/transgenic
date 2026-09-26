"""Athaliana ONT C0 view: 35 training runs (cui2020 under its canonical dataset name, author decision 7; col0_DRP009401
metadata from the PRJDB14952 retry) + 3 FLIC validation runs (separate driver, decision 8). Primary and audit views
use canonical dataset basenames; audit only for training rows."""
import csv, hashlib, json
from pathlib import Path
r = Path('/data/gpfs/assoc/pgl/data/Transgenic'); P = r / 'evidence/b6_prepare_20260925'; sp = 'Athaliana'
train = json.loads((P / 'inputs/ont_tasks_athaliana.json').read_text()); flic = json.loads((P / 'inputs/ont_tasks_flic.json').read_text())
roles = [x for x in csv.DictReader((P / 'inputs/DATASET_ROLES.longread_v1.tsv').open(), delimiter='\t')
         if x['species'] == sp and x['dataset'].startswith(('ont/', 'training/ont/')) and x['role'] != 'excluded']
assert {(t['dataset'], t['run']) for t in train + flic} == {(x['dataset'], x['run']) for x in roles}, 'tasks != roles'
run_dir = P / f'c0/{sp}_ont_v2'; run_dir.mkdir(parents=True, exist_ok=False); (run_dir / 'out').mkdir()
def fields(p): return dict(l.split(None, 1) for l in p.read_text().splitlines() if len(l.split(None, 1)) == 2)
recs, mm, st, n = {}, set(), set(), 0
for t, training in [(x, True) for x in train] + [(x, False) for x in flic]:
    ds, run = Path(t['dataset']).name, t['run']; prim = Path(t['primary'])
    meta = P / f'metadata/ont/{sp}/{ds}/{run}'
    if t['dataset'] == 'training/ont/Athaliana/col0_DRP009401':
        meta = P / f'metadata/ont_retry/{sp}/{ds}/{run}'
    assert (meta / 'METADATA_DONE').is_file(), meta
    prov = json.loads((meta / 'c0_metadata.tsv.PROVENANCE.json').read_text())
    assert prov['output_sha256'] == hashlib.sha256((meta / 'c0_metadata.tsv').read_bytes()).hexdigest(), meta
    assert prov['protocol'] == t['protocol'] and prov['umi_status'] == 'absent', meta
    n += prov['reads']
    view = run_dir / f'align/{sp}/{ds}/{run}'; view.mkdir(parents=True)
    for f in [f'{run}.bam', f'{run}.paf.gz', 'DONE', 'PROVENANCE.txt']:
        assert (prim / f).is_file(), prim / f
        (view / f).symlink_to(prim / f)
    for f in ['c0_metadata.tsv', 'c0_metadata.tsv.sha256', 'c0_metadata.tsv.PROVENANCE.json']:
        (view / f).symlink_to(meta / f)
    srcs = [prim / 'PROVENANCE.txt']
    if training:
        aud = Path(t['audit']); assert (aud / 'DONE').is_file(), aud
        av = run_dir / f'audit_view/{sp}/{ds}'; av.mkdir(parents=True, exist_ok=True); (av / run).symlink_to(aud)
        srcs.append(aud / 'PROVENANCE.txt')
    for q in srcs:
        d = fields(q); mm.add(d['minimap2'].strip()); st.add(d['samtools'].strip())
        recs[str(q)] = hashlib.sha256(q.read_bytes()).hexdigest()
assert len(mm) == 1 and len(st) == 1, (mm, st)
record = dict(tools={'minimap2': {'version': mm.pop()}, 'samtools': {'version': st.pop()}}, source_records=recs)
f = run_dir / 'tool_provenance.json'; f.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
Path(str(f) + '.sha256').write_text(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n')
print('VIEW', sp, 'runs', len(train) + len(flic), 'reads', n, json.dumps(record['tools']))
