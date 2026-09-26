"""C0 view for an N=100 audit rebuild (author decision 12): same primary alignments and metadata as the N=50 builds,
audit alignments and tool provenance from evidence/audit_align_n100. Usage: c0_view_n100.py <species> <est|pacbio|ont>"""
import csv, hashlib, json, sys
from pathlib import Path
r = Path('/data/gpfs/assoc/pgl/data/Transgenic'); P = r / 'evidence/b6_prepare_20260925'; A = r / 'evidence/audit_align_n100'
sp, src = sys.argv[1], sys.argv[2]
run = P / f'c0/{sp}_{src}_n100'; run.mkdir(parents=True, exist_ok=False); (run / 'out').mkdir()
def fields(q): return dict(l.split(None, 1) for l in q.read_text().splitlines() if len(l.split(None, 1)) == 2)
def link(view, prim, meta, names):
    view.mkdir(parents=True)
    for f in names: assert (prim / f).is_file(), prim / f; (view / f).symlink_to(prim / f)
    assert (meta / 'METADATA_DONE').is_file(), meta
    prov = json.loads((meta / 'c0_metadata.tsv.PROVENANCE.json').read_text())
    assert prov['output_sha256'] == hashlib.sha256((meta / 'c0_metadata.tsv').read_bytes()).hexdigest(), meta
    for f in ['c0_metadata.tsv', 'c0_metadata.tsv.sha256', 'c0_metadata.tsv.PROVENANCE.json']: (view / f).symlink_to(meta / f)
units = []   # (primary_dir, audit_dir)
if src == 'est':
    prim = r / f'evidence/est_align/{sp}'
    meta = P / (f'metadata_est_clone_display_fix/{sp}' if sp == 'Gmax' else f'metadata_est_missing_library/{sp}' if sp == 'Vvinifera' else f'metadata/est/{sp}')
    link(run / f'align/{sp}', prim, meta, [f'{sp}.est.bam', f'{sp}.est.paf.gz', 'DONE', 'PROVENANCE.txt'])
    units.append((prim, A / f'{sp}/est/est_primary'))
else:
    roles = [x for x in csv.DictReader((P / 'inputs/DATASET_ROLES.longread_v1.tsv').open(), delimiter='\t')
             if x['species'] == sp and x['role'] != 'excluded' and x['dataset'].startswith((f'training/{src}/', f'{src}/'))]
    for x in roles:
        assert x['role'] == 'c2_training_eligible', x
        ds, rn = Path(x['dataset']).name, x['run']
        prim = r / f'evidence/longread_align_v1/{sp}/{ds}/{rn}'
        link(run / f'align/{sp}/{ds}/{rn}', prim, P / f'metadata/{src}/{sp}/{ds}/{rn}', [f'{rn}.bam', f'{rn}.paf.gz', 'DONE', 'PROVENANCE.txt'])
        units.append((prim, A / f'{sp}/{ds}/{rn}'))
recs, mm, st = {}, set(), set()
for prim, aud in units:
    assert (aud / 'DONE').is_file(), aud
    for q in (prim / 'PROVENANCE.txt', aud / 'PROVENANCE.txt'):
        d = fields(q); mm.add(d['minimap2'].strip()); st.add(d['samtools'].strip()); recs[str(q)] = hashlib.sha256(q.read_bytes()).hexdigest()
assert len(mm) == 1 and len(st) == 1, (mm, st)
rec = dict(tools={'minimap2': {'version': mm.pop()}, 'samtools': {'version': st.pop()}}, source_records=recs)
f = run / 'tool_provenance.json'; f.write_text(json.dumps(rec, sort_keys=True, indent=2) + '\n')
Path(str(f) + '.sha256').write_text(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n')
print('VIEW', sp, src, 'units', len(units))
