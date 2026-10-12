"""Read-only compute-node identity/dependency diagnostic; no training data writes."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import socket
import sys

root = Path('/data/gpfs/assoc/pgl/data/Transgenic')
report = {'host': socket.gethostname(), 'python': sys.executable, 'dependencies': {}}
for name in ('duckdb', 'numpy', 'pysam'):
    try:
        module = importlib.import_module(name)
        report['dependencies'][name] = {'version': module.__version__, 'file': module.__file__}
    except Exception as exc:
        report['dependencies'][name] = {'error': repr(exc)}
print(json.dumps(report), flush=True)

def identity(path):
    before = path.stat()
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b''):
            sha.update(chunk)
            md5.update(chunk)
    after = path.stat()
    assert (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'input changed while hashing'
    result = {'path': str(path), 'bytes': after.st_size, 'sha256': sha.hexdigest(), 'md5': md5.hexdigest()}
    print(json.dumps({'identity': result}), flush=True)
    return result

corpus = root / 'training_input/db/b5_full_a40_v1.db'
assert not Path(str(corpus) + '.wal').exists() and not Path(str(corpus) + '-wal').exists()
db_identity = identity(corpus)
assert db_identity['sha256'] == '6840c93fd0b999e09b8dc2897adca1f4dbae36f2ef0f8fcc10ee66371295e222'
assert db_identity['bytes'] == 23534252032
bam_identity = identity(root / 'evidence/est_align/Athaliana/Athaliana.est.bam')
assert bam_identity['md5'] == 'cdedd3d42282c17e305f8bcd17c3be27'
refs = [identity(root / p) for p in ('genomes/Athaliana_167_TAIR10.fa', 'training_input/genomes/Athaliana_167_TAIR10.fa')]
assert refs[0]['sha256'] == refs[1]['sha256'], 'BAM and bundled reference byte mismatch'
import duckdb
with duckdb.connect(str(corpus), read_only=True) as con:
    con.execute('SET threads=2')
    con.execute("SET memory_limit='2GB'")
    rows = con.execute("SELECT species_id, fasta, fasta_sha256, gff, gff_sha256 FROM build_manifest WHERE species_id='Athaliana'").fetchall()
    print(json.dumps({'build_manifest': rows}), flush=True)
    assert rows, 'no Athaliana build manifest'
    print(json.dumps({'tile_blocks_schema': con.execute('DESCRIBE tile_blocks').fetchall(), 'blocks': con.execute("SELECT split, count(*) FROM tile_blocks WHERE species_id='Athaliana' GROUP BY split").fetchall(), 'tiles': con.execute("SELECT split,count(*) FROM geneList WHERE species_id='Athaliana' GROUP BY split").fetchall()}), flush=True)
    linked = all(row[2] == refs[0]['sha256'] for row in rows)
    print(json.dumps({'build_reference_bytes_match': linked}), flush=True)
    assert linked, 'corpus build FASTA hash differs; sequence equivalence needs separate verification'
assert not Path(str(corpus) + '.wal').exists() and not Path(str(corpus) + '-wal').exists()
print(json.dumps({'identity_checks': 'PASS', 'cpu_dependencies_ready': all('version' in v for v in report['dependencies'].values()), 'production_build_started': False}), flush=True)
