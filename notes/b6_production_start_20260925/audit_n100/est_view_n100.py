"""EST C0 view for an N=100 audit rebuild (author decision 12): same primary alignment + metadata as the N=50 build,
tool provenance bound to the primary and the N=100 audit PROVENANCE."""
import hashlib, json, sys
from pathlib import Path
r = Path('/data/gpfs/assoc/pgl/data/Transgenic'); p = r / 'evidence/b6_prepare_20260925'; sp = sys.argv[1]
run = p / f'c0/{sp}_est_n100'; run.mkdir(parents=True, exist_ok=False); view = run / f'align/{sp}'; view.mkdir(parents=True); (run / 'out').mkdir()
primary = r / f'evidence/est_align/{sp}'; audit = r / f'evidence/audit_align_n100/{sp}/est/est_primary'
meta = p / (f'metadata_est_clone_display_fix/{sp}' if sp == 'Gmax' else f'metadata_est_missing_library/{sp}' if sp == 'Vvinifera' else f'metadata/est/{sp}')
assert (meta / 'METADATA_DONE').is_file() and (audit / 'DONE').is_file(), (meta, audit)
for f in [f'{sp}.est.bam', f'{sp}.est.paf.gz', 'DONE', 'PROVENANCE.txt']: (view / f).symlink_to(primary / f)
for f in ['c0_metadata.tsv', 'c0_metadata.tsv.sha256', 'c0_metadata.tsv.PROVENANCE.json']: (view / f).symlink_to(meta / f)
def fields(q): return dict(l.split(None, 1) for l in q.read_text().splitlines() if len(l.split(None, 1)) == 2)
a, b = fields(primary / 'PROVENANCE.txt'), fields(audit / 'PROVENANCE.txt')
assert a['minimap2'] == b['minimap2']
rec = dict(tools={'minimap2': {'version': a['minimap2']}, 'samtools': {'version': a['samtools']}},
           source_records={str(q): hashlib.sha256(q.read_bytes()).hexdigest() for q in (primary / 'PROVENANCE.txt', audit / 'PROVENANCE.txt')})
f = run / 'tool_provenance.json'; f.write_text(json.dumps(rec, sort_keys=True, indent=2) + '\n')
Path(str(f) + '.sha256').write_text(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n')
print('VIEW', view)
