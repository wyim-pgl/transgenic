#!/usr/bin/env python3
"""Local-only adaptation of task2 build_release.build(repo,out); no deployment."""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import authority
COMMIT='9603e20fd4870a9f5b3d5f68e2df8ba0f8a49558'
COMPONENT='ad3e5a7aacbc5d71c3a59d9b1b6f6fd0afeff12efab10b7ed123f734e767380e'
DECODER='b42dc4b7e70f7e2cc8cfa41dc10817a3cc2d2dc2837bb0b9637d9fab2a28f088'
REMOTE='/data/gpfs/assoc/pgl/data/Transgenic/evidence/mvp50_reference_map_20260916'
OLD='b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6'
OLD_INVENTORY='7b6e1ed4bbf635c7b25b47b2564453d9b176f6a21f858c87c95e270f5bee0689'
OLD_COMPONENT='555c3618e792dbed06bfc2c392b99efe79fd16853e8c92abe252a5b34cbd0c44'
def digest(raw): return hashlib.sha256(raw).hexdigest()
def sha(path): return digest(Path(path).read_bytes())
def canonical(obj): return (json.dumps(obj,sort_keys=True,separators=(',',':'))+'\n').encode()
def dump(path,obj): path.write_bytes(canonical(obj))
def load(path):
    spec=importlib.util.spec_from_file_location('staged_cpu',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module

def build(repo,out):
    repo=Path(repo).resolve(strict=True); out=Path(out).absolute(); out.mkdir(parents=True,exist_ok=False)
    old=HERE.parent/'final_release_bindfix1'
    assert sha(old/'transport.json')=='1883df3c32c5b560030dbcb89543b3d2730d7c08518d47e4f932b62f88b18c0a'
    previous=json.loads((old/'transport.json').read_text())
    for n,h in previous['files'].items(): assert sha(old/n)==h,n
    source=out/'source'; source.mkdir()
    release=repo/'notes/mvp_execution_20260916/final_p1/release/release.json'
    assert sha(release)==OLD_INVENTORY
    prior=json.loads(release.read_text()); files={}; changed=[]
    assert len(prior['source_files'])==100
    for name,oldhash in prior['source_files'].items():
        raw=authority.git(repo,'show',COMMIT+':'+name); files[name]=digest(raw)
        if files[name]!=oldhash: changed.append(name)
        path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
    assert changed==['revision/scripts/mvp/mvp_produce_counts.py'],changed
    invariants=['src','train','configs','revision/protocols','revision/scripts/mvp/real_probes.py','revision/scripts/mvp/run_bounded.py','revision/scripts/mvp/make_launch_readiness.py','revision/scripts/mvp/mvp_decode_runtime.py']
    assert not authority.git(repo,'diff',OLD,COMMIT,'--',*invariants)
    components={n:h for n,h in files.items() if n.startswith(('src/transgenic/','configs/','revision/scripts/mvp/')) or n=='train/train_HyenaTransgenic_MVP.py'}
    assert len(components)==86 and digest(canonical(components))==COMPONENT
    tree=ast.parse((source/'revision/scripts/mvp/mvp_decode_runtime.py').read_text())
    names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DECODER_ONLY' for t in n.targets))
    assert digest(canonical({n:files[n] for n in names}))==DECODER
    pins={'commit':COMMIT,'scope':'evaluation_CPU_observations_not_training','source_files':files}
    dump(source/'source_pins.json',pins)
    inventory=sha(source/'source_pins.json')
    for name,h in prior['accepted_companions_not_in_source_commit'].items():
        raw=(repo/name).read_bytes(); assert digest(raw)==h
        path=source/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(raw)
    # Preserve reviewed sparse Git metadata/object bytes and both separate archives exactly.
    for name in ('authority.py','reference_authority.tar','cpu_site.tar','cpu_dependency.json','panel_cpu.sbatch'):
        shutil.copyfile(old/name,out/name)
    with tarfile.open(out/'reference_authority.tar') as tar:
        for m in tar.getmembers():
            assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            assert m.isfile() or m.isdir()
        tar.extractall(source,filter='data')
    receipt=authority.verify(source)
    template=authority.git(repo,'show',COMMIT+':tasknotes/reference_mapping/run_panel_cpu.py').decode()
    assert digest(template.encode())=='09281c0bd38304e2adc317f8f9216fb3686627c64b768985dd84586dd0c75a23'
    wrapper=template
    replacements={OLD:COMMIT,OLD_COMPONENT:COMPONENT,OLD_INVENTORY:inventory}
    for a,b in replacements.items(): assert wrapper.count(a)==1; wrapper=wrapper.replace(a,b)
    (out/'run_panel_cpu.py').write_text(wrapper)
    (out/'entrypoint.diff').write_text(''.join(difflib.unified_diff(template.splitlines(True),wrapper.splitlines(True),fromfile='9603-template',tofile='staged/run_panel_cpu.py')))
    job=load(out/'run_panel_cpu.py'); verified=job.verify_source(source)
    dump(out/'SOURCE_VERIFICATION.json',verified)
    def normalized(info):
        info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
        info.mode=0o755 if info.isdir() else 0o444
        return info
    with tarfile.open(out/'source.tar','w',format=tarfile.PAX_FORMAT) as tar: tar.add(source,arcname='source',filter=normalized)
    payloads=('source.tar','reference_authority.tar','authority.py','run_panel_cpu.py','panel_cpu.sbatch','cpu_site.tar','cpu_dependency.json')
    dump(out/'transport.json',{'source_commit':COMMIT,'reference_authority':receipt,'files':{n:sha(out/n) for n in payloads},
        'planned_root':REMOTE,'expected_source_root':REMOTE+'/source','source_inventory_sha256':inventory,
        'evaluation_component_sha256':COMPONENT,'evaluation_component_files':86,'decoder_code_sha256':DECODER,
        'source_file_count':100,'source_fix_review':'SCOPED_ACCEPT_8508c4b','deployment_status':'PROVISIONAL_PENDING_INDEPENDENT_DEPLOYMENT_DELTA',
        'mode':'full_fresh_50_panel_not_reused','source_changes_from_b660':changed})
    for n in (*payloads,'transport.json'): (out/n).chmod(0o444)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--repo',required=True,type=Path); p.add_argument('--out',required=True,type=Path)
    a=p.parse_args(); print(build(a.repo,a.out))
