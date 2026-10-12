"""Build ONT C0 alignment views (primary alignment + verified ONT metadata) and tool provenance for the ready species."""
import csv, hashlib, json, sys
from pathlib import Path
r = Path('/data/gpfs/assoc/pgl/data/Transgenic'); P = r / 'evidence/b6_prepare_20260925'
tasks = json.loads((P / 'inputs/ont_tasks.json').read_text())
def fields(path): return dict(l.split(None, 1) for l in path.read_text().splitlines() if len(l.split(None, 1)) == 2)
for sp in sys.argv[1:]:
    rows = [t for t in tasks if t['species'] == sp]
    roles = [x for x in csv.DictReader((P / 'inputs/DATASET_ROLES.longread_v1.tsv').open(), delimiter='\t')
             if x['species'] == sp and x['dataset'].startswith(('ont/', 'training/ont/')) and x['role'] != 'excluded']
    assert {(t['dataset'], t['run']) for t in rows} == {(x['dataset'], x['run']) for x in roles}, sp + ' tasks != roles'
    run_dir = P / f'c0/{sp}_ont_v1'; run_dir.mkdir(parents=True, exist_ok=False); (run_dir / 'out').mkdir()
    recs, mm, st, n = {}, set(), set(), 0
    for t in rows:
        ds, run = Path(t['dataset']).name, t['run']
        prim = Path(t['primary']); aud = r / f'evidence/audit_align_v1/{sp}/{ds}/{run}'; meta = P / f'metadata/ont/{sp}/{ds}/{run}'
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
        for q in (prim / 'PROVENANCE.txt', aud / 'PROVENANCE.txt'):
            d = fields(q); mm.add(d['minimap2'].strip()); st.add(d['samtools'].strip())
            recs[str(q)] = hashlib.sha256(q.read_bytes()).hexdigest()
    assert len(mm) == 1 and len(st) == 1, (sp, mm, st)
    record = dict(tools={'minimap2': {'version': mm.pop()}, 'samtools': {'version': st.pop()}}, source_records=recs)
    f = run_dir / 'tool_provenance.json'; f.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    Path(str(f) + '.sha256').write_text(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n')
    print('VIEW', sp, 'runs', len(rows), 'reads', n, json.dumps(record['tools']))
