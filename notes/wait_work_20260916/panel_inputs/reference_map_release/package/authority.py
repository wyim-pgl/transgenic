#!/usr/bin/env python3
"""Transport five ORIGINAL Git objects, not a replacement manifest or synthetic commit.
This is intentionally an incomplete object database: only this pinned git-show path
is supported. No histories, unrelated blobs, refs, alternates or replacements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

COMMIT = '92d8020ee79eeedd7f1d50167b1c8dc94700fb16'
PATH = 'data/manifests/b5_species_v1.tsv'
SPEC = COMMIT + ':' + PATH

def git(root, *args, data=None):
    env = {k:v for k,v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null', GIT_NO_REPLACE_OBJECTS='1')
    return subprocess.run(['git','-C',str(root),*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,env=env).stdout

def chain(root):
    specs = [(COMMIT,'commit'),(COMMIT+':','tree'),(COMMIT+':data','tree'),(COMMIT+':data/manifests','tree'),(SPEC,'blob')]
    rows=[]
    for spec, kind in specs:
        oid=git(root,'rev-parse',spec).decode().strip()
        raw=git(root,'cat-file',kind,oid)
        actual=git(root,'hash-object','-t',kind,'--stdin',data=raw).decode().strip()
        if actual != oid: raise ValueError('object content does not hash to '+oid)
        rows.append({'oid':oid,'type':kind,'bytes':len(raw)})
    return rows

def install(repo, root):
    root=Path(root).resolve(strict=True)
    if (root/'.git').exists(): raise FileExistsError(root/'.git')
    rows=chain(repo)
    git(root,'init','--quiet','--template=')
    for row in rows:
        raw=git(repo,'cat-file',row['type'],row['oid'])
        actual=git(root,'hash-object','-w','-t',row['type'],'--stdin',data=raw).decode().strip()
        if actual!=row['oid']: raise ValueError('Git object transport changed identity')
    original=git(repo,'show',SPEC)
    if git(root,'show',SPEC)!=original: raise ValueError('git show bytes differ')
    receipt={'commit':COMMIT,'path':PATH,'objects':rows,'manifest_sha256':hashlib.sha256(original).hexdigest(), 'scope':'sparse_original_git_objects_only_not_checkout_or_history'}
    with (root/'reference_authority.json').open('x') as f: json.dump(receipt,f,sort_keys=True,indent=2); f.write('\n')
    with (root/'b5_species_v1.authority.tsv').open('xb') as f: f.write(original)
    return verify(root)

def verify(root):
    root=Path(root).resolve(strict=True)
    if not (root/'.git').is_dir() or (root/'.git').is_symlink(): raise ValueError('own real .git required; no parent-repo fallback')
    if (root/'.git/objects/info/alternates').exists() or (root/'.git/refs/replace').exists(): raise ValueError('alternate/replacement authority forbidden')
    receipt=json.loads((root/'reference_authority.json').read_text())
    if receipt['commit']!=COMMIT or receipt['path']!=PATH: raise ValueError('authority pins differ')
    rows=chain(root)
    data=git(root,'show',SPEC)
    if rows!=receipt['objects'] or hashlib.sha256(data).hexdigest()!=receipt['manifest_sha256'] or data!=(root/'b5_species_v1.authority.tsv').read_bytes(): raise ValueError('authority receipt/bytes mismatch')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--root',type=Path,required=True); p.add_argument('--from-repo',type=Path)
    a=p.parse_args(); print(json.dumps(install(a.from_repo,a.root) if a.from_repo else verify(a.root),indent=2))
