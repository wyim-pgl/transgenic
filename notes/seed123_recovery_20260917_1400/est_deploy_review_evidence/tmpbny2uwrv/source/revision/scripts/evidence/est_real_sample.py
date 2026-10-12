import hashlib, json
from pathlib import Path
def main(argv):
    args = dict(zip(argv[::2], argv[1::2]))
    out = Path(args['--out']); out.mkdir()
    (out / 'diagnostics.jsonl').write_bytes(b'fixture only\n')
    report = dict(status='COMPLETE', diagnostics_sha256=hashlib.sha256((out / 'diagnostics.jsonl').read_bytes()).hexdigest(), manifest_file={'sha256':args['--sha']}, training_eligible=False, full_c0=False, a11_status='unassessable', route_a_complete=False, molecule_counts_complete=False, raster=False)
    (out / 'report.json').write_text(json.dumps(report))
    (out / 'COMPLETE').write_text(hashlib.sha256((out / 'report.json').read_bytes()).hexdigest()+'\n')
    pass
    return 0
