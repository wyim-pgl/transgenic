#!/usr/bin/env python3
"""Local-only exact b660 export + original sparse reference Git authority."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import authority
import run_panel_cpu as job

HERE=Path(__file__).resolve().parent

def build(repo,out):
    repo=Path(repo).resolve(strict=True); out=Path(out).absolute(); out.mkdir(parents=True,exist_ok=False)
    source=out/'source'; source.mkdir()
    release=repo/'notes/mvp_execution_20260916/final_p1/release/release.json'
    shutil.copyfile(release,source/'source_pins.json')
    pins=json.loads(release.read_text())
    if job.sha(release)!='7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689': raise ValueError('wrong reviewed source inventory')
    for name,digest in pins['source_files'].items():
        raw=authority.git(repo,'show',job.SOURCE_COMMIT+':'+name)
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('b660 source bytes mismatch: '+name)
        path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
    for name,digest in job.COMPANIONS.items():
        raw=(repo/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest: raise ValueError('accepted companion mismatch: '+name)
        path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
    receipt=authority.install(repo,source); job.verify_source(source)
    # Task3 may extract ONLY this tar into its own exact-code root (no .git there).
    with tarfile.open(out/'reference_authority.tar','w') as tar:
        for name in ('.git','reference_authority.json','b5_species_v1.authority.tsv'): tar.add(source/name,arcname=name)
    for name in ('authority.py','run_panel_cpu.py','panel_cpu.sbatch','cpu_dependency.json'): shutil.copyfile(HERE/name,out/name)
    deps=json.loads((HERE/'cpu_dependency.json').read_text())
    if job.sha(HERE/'cpu_dependency.json')!='cf4d509e8914d6bcf44e6a96a79fe127e4521f7592e4a768a7e2659760ec3435': raise ValueError('CPU dependency receipt changed')
    with tarfile.open(out/'cpu_site.tar','w') as tar:
        for name,digest in deps['files'].items():
            if job.sha(HERE/'cpu_site'/name)!=digest: raise ValueError('CPU dependency differs')
            tar.add(HERE/'cpu_site'/name,arcname='cpu_site/'+name)
    with tarfile.open(out/'source.tar','w') as tar: tar.add(source,arcname='source')
    job.dump(out/'transport.json',{'source_commit':job.SOURCE_COMMIT,'reference_authority':receipt,
             'files':{n:job.sha(out/n) for n in ('source.tar','reference_authority.tar','authority.py','run_panel_cpu.py','panel_cpu.sbatch','cpu_site.tar','cpu_dependency.json')}})
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); print(build(a.repo,a.out))
