"""Readonly SSH transfer; no canonical reads/reference bytes or remote hashing.
Sample + SAM stat-confirmed 28,767,390 bytes: bounded local hash verification.
"""
import io,json,pathlib,shlex,subprocess,tarfile
HERE=pathlib.Path(__file__).resolve().parent
R='/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp_ont_orientation_20260916_pilot'
out='results/SRR28341943/'
names=['receipt.json','COMPLETE','commands.json','orientation/DONE','orientation/audit.json','sample.fa']
for stage in ['version_minimap2','version_samtools','version_seqkit','sample','sample_alignment','orientation_audit']:
    names += [stage+'.stdout',stage+'.stderr']
files=[out+n for n in names]+['release/slurm-6165555.out','release/slurm-6165555.err']
files+=['release/'+n for n in ['gate.py','source_pins.json','authority.tsv','orientation_pilot.sbatch']]
files+=['release/frozen/'+n for n in json.loads((HERE.parent/'source_pins.json').read_text())['files']]
# stat only, bounded explicit file list; reject links and unexpectedly large transfer.
cmd='stat -c '+shlex.quote('%s|%F')+' '+' '.join(shlex.quote(R+'/'+n) for n in files)
r=subprocess.run(['ssh','pronghorn',cmd],capture_output=True,check=True)
(HERE/'fetch_stat.txt').write_bytes(r.stdout)
rows=r.stdout.decode().splitlines()
assert len(rows)==len(files) and all(x.split('|')[1] in ('regular file','regular empty file') for x in rows)
assert sum(int(x.split('|')[0]) for x in rows)<32_000_000
cmd='tar -C '+shlex.quote(R)+' -cf - '+' '.join(map(shlex.quote,files))
r=subprocess.run(['ssh','pronghorn',cmd],capture_output=True,check=True)
assert len(r.stdout)<33_000_000
with tarfile.open(fileobj=io.BytesIO(r.stdout)) as t:
    assert set(t.getnames())==set(files)
    for m in t:
        assert m.isfile() and m.name in files
        p=HERE/m.name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(t.extractfile(m).read())
(HERE/'fetch_manifest.json').write_text(json.dumps(dict(remote_root=R,files=files,total_file_bytes=sum(int(x.split('|')[0]) for x in rows)),indent=2)+'\n')
print('Fetched',len(files),'files;',sum(int(x.split('|')[0]) for x in rows),'bytes')
