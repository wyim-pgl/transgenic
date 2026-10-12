import ast, hashlib, importlib.util, json, os, subprocess, sys, tarfile
from pathlib import Path
R=Path('/mnt/f/scratch/develop/Transgenic/transgenic_local')
P=R/'notes/wait_work_20260916/panel_inputs/reference_map_release/package'
E=Path(__file__).parent
S=P/'source'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args): return subprocess.check_output(['git','-C',str(R),*args])
def canon(x): return (json.dumps(x,sort_keys=True,separators=(',',':'))+'\n').encode()
t=json.loads((P/'transport.json').read_text())
assert sha(P/'transport.json')=='11d0550bd46c121c6b53e1cb51f0c528381ff41763e7e95e754d286a4d0448b5'
for n,h in t['files'].items(): assert sha(P/n)==h
pins=json.loads((S/'source_pins.json').read_text()); files=pins['source_files']
assert sha(S/'source_pins.json')==t['source_inventory_sha256']
assert len(files)==100
old='b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6'; new=t['source_commit']
changed=[]
for n,h in files.items():
    assert (S/n).read_bytes()==git('show',new+':'+n)
    if git('show',old+':'+n)!=(S/n).read_bytes(): changed.append(n)
assert changed==t['source_changes_from_b660']
component={n:h for n,h in files.items() if n.startswith(('src/transgenic/','configs/','revision/scripts/mvp/')) or n=='train/train_HyenaTransgenic_MVP.py'}
assert len(component)==86 and hashlib.sha256(canon(component)).hexdigest()==t['evaluation_component_sha256']
tree=ast.parse((S/'revision/scripts/mvp/mvp_decode_runtime.py').read_text())
names=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='DECODER_ONLY' for x in n.targets))
assert hashlib.sha256(canon({n:files[n] for n in names})).hexdigest()==t['decoder_code_sha256']
assert not git('diff',old,new,'--','src','train','configs','revision/protocols','revision/scripts/mvp/real_probes.py','revision/scripts/mvp/run_bounded.py','revision/scripts/mvp/make_launch_readiness.py','revision/scripts/mvp/mvp_decode_runtime.py')
archives={}
for name in ('source.tar','reference_authority.tar','cpu_site.tar'):
    with tarfile.open(P/name) as ar:
        members=ar.getmembers(); ns=[m.name for m in members]
        assert len(ns)==len(set(ns))
        assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and (m.isfile() or m.isdir()) for m in members)
        content={m.name:ar.extractfile(m).read() for m in members if m.isfile()}
        if name=='source.tar':
            actual={str(f.relative_to(P)) for f in S.rglob('*') if f.is_file()}
            assert set(content)==actual
            assert all(raw==(P/n).read_bytes() for n,raw in content.items())
        elif name=='reference_authority.tar':
            assert all(raw==(S/n).read_bytes() for n,raw in content.items())
        else:
            dep=json.loads((P/'cpu_dependency.json').read_text())
            assert set(content)=={'cpu_site/'+n for n in dep['files']}
            assert all(hashlib.sha256(content['cpu_site/'+n]).hexdigest()==h for n,h in dep['files'].items())
        archives[name]=len(content)
sys.path[:0]=[str(P),str(S/'revision/scripts/mvp'),str(S/'src')]
import authority, mvp_produce_counts as producer
assert authority.verify(S)==t['reference_authority']
for row in t['reference_authority']['objects']:
    local=authority.git(S,'cat-file',row['type'],row['oid'])
    assert local==git('cat-file',row['type'],row['oid'])
# Direct staged-package publication boundary exercise; tiny local inputs only.
f=E/'tiny-corpus'; f.write_bytes(b'tiny test not DB')
receipt={'sources':[producer.file_identity(f)]}
final=E/'late-wal-final.json'
producer.stage_json(final,{'diagnostic':True})
wal=Path(str(f)+'.wal'); wal.write_bytes(b'late WAL')
try: producer.publish_staged([final],lambda:producer.recheck_reference_mapping(receipt))
except ValueError as exc: assert str(exc)=='corpus WAL appeared'
else: raise AssertionError('late WAL published')
assert not final.exists() and Path(str(final)+'.pending').exists() and wal.exists()
positive=E/'positive.json'; producer.stage_json(positive,{'ok':True}); producer.publish_staged([positive],lambda:None)
assert positive.exists() and not Path(str(positive)+'.pending').exists()
producer.stage_json(positive,{'overwrite':True})
try: producer.publish_staged([positive],lambda:None)
except FileExistsError: pass
else: raise AssertionError('overwrote occupied output')
assert json.loads(positive.read_text())=={'ok':True}
print(json.dumps({'status':'PASS','transport_sha256':sha(P/'transport.json'),'source_files':100,'component_files':86,'changed':changed,'archive_regular_file_counts':archives,'original_git_objects':5,'late_wal':'rejected_pending_preserved','exclusive_publication':'passed','pins':{k:t[k] for k in ('source_inventory_sha256','evaluation_component_sha256','decoder_code_sha256')}},indent=2))
