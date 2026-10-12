"""Read-only login probe: stat and small package/receipt metadata; no package imports."""
import datetime, glob, json, os, platform
from pathlib import Path
R = '/data/gpfs/assoc/pgl/data/Transgenic'
paths = [R+'/genomes/Athaliana_167_TAIR10.fa', R+'/genomes/Athaliana_167_TAIR10.gene.clean.gff3', R+'/training_input/db/b5_full_a40_v1.db', R+'/evidence/mvp_full_20260915/fullc0lite.db', '/data/gpfs/home/wyim/scratch/singularity/micromamba.sif', '/data/gpfs/home/wyim/micromamba/bin/python', '/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/bin/python3.11']
result = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'host':platform.node(), 'files':[], 'packages':[], 'receipts':{}, 'scope':'stat_and_small_metadata_only_no_imports_no_db_reads'}
for name in paths:
    p = Path(name)
    try:
        s = p.stat(); result['files'].append({'path':name,'realpath':str(p.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns})
    except OSError as e: result['files'].append({'path':name,'error':str(e)})
for base in ['/data/gpfs/assoc/pgl/bin/conda/conda_envs/transgenic-revision/lib/python3.11/site-packages', '/data/gpfs/home/wyim/micromamba/lib/python3.11/site-packages', '/data/gpfs/home/wyim/.local/lib/python3.11/site-packages', R+'/worktrees/mvp-cpu-deps-20260914/site']:
    for package in ['torch','numpy','duckdb','accelerate','transformers','pysam','safetensors']:
        for name in glob.glob(base+'/'+package+'-*.dist-info/METADATA'):
            with open(name) as f:
                lines = [line.strip() for line in f if line.startswith(('Name:','Version:','Requires-Python:'))]
            result['packages'].append({'path':name,'metadata':lines})
for name in ['SUCCESS.json','input_identities.json','raster_results.json']:
    p = Path(R+'/evidence/mvp_full_20260915')/name
    if p.is_file() and p.stat().st_size < 65536:
        result['receipts'][str(p)] = p.read_text()
print(json.dumps(result,indent=2))
