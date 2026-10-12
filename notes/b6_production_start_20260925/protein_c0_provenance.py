"""Write per-species protein C0 tool provenance (miniprot + DIAMOND + samtools) with source record hashes."""
import hashlib, json, re, sys
from pathlib import Path
r = Path('/data/gpfs/assoc/pgl/data/Transgenic'); P = r / 'evidence/b6_prepare_20260925'
def fields(p): return dict(l.split(None, 1) for l in p.read_text().splitlines() if len(l.split(None, 1)) == 2)
for sp in sys.argv[1:]:
    bx = r / ('evidence/est_blastx/sitalica_pilot_20260914' if sp == 'Sitalica' else f'evidence/est_blastx/remaining8_20260915/{sp}')
    mp = r / f'evidence/protein_alignments/{sp}/PROVENANCE.txt'
    mv = fields(mp)['miniprot'].strip()
    dv = set()
    for s in ('orthodb', 'swissprot'):
        v = json.loads((bx / s / 'provenance.json').read_text())['diamond_version']
        dv.add(re.search(r'diamond v(\d+\.\d+\.\d+\.\d+)', v).group(1))
    assert mv == '0.18-r281' and dv == {'2.1.24.178'}, (sp, mv, dv)
    recs = {str(q): hashlib.sha256(q.read_bytes()).hexdigest()
            for q in (mp, bx / 'orthodb/provenance.json', bx / 'swissprot/provenance.json')}
    record = dict(tools={'miniprot': {'version': mv}, 'diamond': {'version': dv.pop()}, 'samtools': {'version': 'samtools 1.3.1'}},
                  source_records=recs)
    run = P / f'c0/{sp}_protein_v1'; run.mkdir(parents=True, exist_ok=False); (run / 'out').mkdir()
    f = run / 'tool_provenance.json'; f.write_text(json.dumps(record, sort_keys=True, indent=2) + '\n')
    Path(str(f) + '.sha256').write_text(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + f.name + '\n')
    print('PROV', sp, json.dumps(record['tools']))
