#!/bin/bash
# Compatibility probe for the existing frozen scorer; not a WP5 evaluation.
#SBATCH --job-name=transgenic-wp5-cpu-scorer
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --constraint=work
#SBATCH --no-requeue
#SBATCH --output=cpu-scoring-%j.out
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess

job = os.environ.get('SLURM_JOB_ID', '')
assert job.isdigit(), 'Allocation required'
assert platform.machine() == 'x86_64', 'This is an x86 scorer probe'
details = subprocess.check_output(['scontrol', 'show', 'job', '-o', job], text=True)
fields = dict(x.split('=', 1) for x in shlex.split(details) if '=' in x)
assert fields['Account'] == 'bilv-delta-cpu' and fields['Partition'] == 'cpu'
for key in ('ReqTRES', 'AllocTRES'):
    assert fields[key] != '(null)'
    for item in fields[key].split(','):
        name, value = item.split('=', 1)
        assert not name.startswith('gres/gpu') or int(value) == 0
root = Path(os.environ['SLURM_SUBMIT_DIR'])
binary = root / 'bin/gffcompare'
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
assert digest == 'a440e1533611a40a51cf2b5b35a2e15ec6b885660dd35f6b531e0d65601a963a'
version = subprocess.run([str(binary), '--version'], text=True, capture_output=True, check=True)
assert (version.stdout + version.stderr).strip() == 'gffcompare v0.12.6'
out = root / ('fixture-' + job)
out.mkdir(exist_ok=False)

def exon(start, end, gene, transcript):
    return 'chr1\tfixture\texon\t%d\t%d\t.\t+\t.\tgene_id "%s"; transcript_id "%s";\n' % (start, end, gene, transcript)

ref = out / 'reference.gtf'
query = out / 'query.gtf'
ref.write_text(exon(1001, 1100, 'r1', 'ref1') + exon(1201, 1300, 'r1', 'ref1'))
query.write_text(exon(1001, 1100, 'q1', 'exact') + exon(1201, 1300, 'q1', 'exact') +
                 exon(1001, 1100, 'q2', 'altered') + exon(1251, 1350, 'q2', 'altered'))
command = [str(binary), '-r', str(ref), '-o', str(out / 'comparison'), str(query)]
run = subprocess.run(command, text=True, capture_output=True, check=True)
tmaps = list(out.glob('*.tmap'))
assert len(tmaps) == 1, 'Expected one query tmap'
rows = [line.split('\t') for line in tmaps[0].read_text().splitlines()[1:] if line]
codes = {row[4]: row[2] for row in rows}
assert set(codes) == {'exact', 'altered'}, 'Missing or additional fixture transcript'
assert codes['exact'] == '=' and codes['altered'] != '=', 'Scorer fixture mismatch'
receipt = dict(kind='wp5_cpu_scorer_compatibility', job_id=job,
               hostname=platform.node(), architecture=platform.machine(),
               account=fields['Account'], allocated_tres=fields['AllocTRES'],
               binary_sha256=digest, version=(version.stdout + version.stderr).strip(),
               command=command, fixture_codes=codes,
               scope='Pinned binary compatibility and tiny synthetic intron-chain fixture only; no model inference, pilot gate, real-data score or scientific approval.',
               stdout=run.stdout, stderr=run.stderr)
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
PY
