"""Small metadata-only remote probe; never hashes or scans read payloads."""
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
ROOT = Path('/data/gpfs/assoc/pgl/data/Transgenic')
runs = {'SRR28341943', 'SRR28341944', 'SRR28341945', 'ERR13994458'}
roles = ROOT / 'evidence/DATASET_ROLES.longread_v1.tsv'
print('observed', datetime.datetime.now().astimezone().isoformat())
print('roles_sha256', hashlib.sha256(roles.read_bytes()).hexdigest())
rows = [r for r in csv.DictReader(roles.open(), delimiter='\t') if r['run'] in runs]
print(json.dumps(rows, indent=2))
for line in (ROOT / 'evidence/longread_runs_v1.tsv').read_text().splitlines():
    if any(run in line for run in runs):
        print('run_inventory', line)
        p = ROOT / line.split('\t')[-1]
        print('file', p, 'exists', p.exists(), 'bytes', p.stat().st_size if p.exists() else None)
for row in rows:
    p = ROOT / 'evidence' / row['dataset']
    print('dataset_dir', p, 'exists', p.exists())
    if p.is_dir():
        print('small_directory_listing', [str(x) for x in p.iterdir()][:30])
        for suffix in ('.fa.gz', '.md5', '.DONE', '.source.md5'):
            q = p / (row['run'] + suffix)
            print('actual_file', str(q), q.stat().st_size if q.exists() else None)
            if q.exists() and suffix != '.fa.gz' and q.stat().st_size < 8192:
                print('receipt', q.read_text())
print('hifi_dir_candidates', [(str(ROOT/'evidence'/x), (ROOT/'evidence'/x).exists()) for x in ('pacbio/Athaliana_PRJEB77203', 'hifi/Athaliana_PRJEB77203')])
for technology in ('pacbio', 'hifi'):
    directory = ROOT / 'evidence' / technology
    if directory.is_dir():
        print('hifi_shallow_inventory', [x.name for x in directory.iterdir()])
bin = Path('/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin')
for tool in ('minimap2', 'samtools', 'seqkit'):
    p = bin / tool
    print('tool', p, p.exists())
    if p.exists():
        result = subprocess.run([str(p), 'version' if tool == 'seqkit' else '--version'], capture_output=True, text=True, timeout=10)
        print('version', result.returncode, (result.stdout + result.stderr)[:1500])
