"""Read-only bounded collector derived from preserved v3 collector. No scheduler polling."""
import ast, pathlib, subprocess, json, base64, hashlib
P=pathlib.Path(__file__).resolve().parent
prior=P.parents[1]/'ont_fixture_v3/result/collect_readonly.py'
tree=ast.parse(prior.read_text())
remote=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REMOTE' for t in n.targets))
remote=remote.replace('/data/gpfs/home/wyim/ont_f4_20260917_1653','/data/gpfs/home/wyim/ont_f4_fix_20260917_1714')
remote=remote.replace('budget=22*1024*1024','budget=20*1024*1024')
remote=remote.replace("closure=json.loads", "assert digest(root/'release/MANIFEST.sha256')=='13da604041044f3ce0192b2049d644a603b9bbef8d14d38c70fedc07ae92d3b9'\nassert digest(root/'controller_approval.json')=='981023667e3dc0b18a44ae62f6b8b6694dc177d84f19e5989bf26a713d804da5'\nassert digest(root/'release/runtime_evidence/expected_closure.json')=='17028c5096c789ec7d198c489d08eec8cbc0224ec1ebd492115a475d9ee243ce'\nclosure=json.loads")
remote=remote.replace('for p,e in sorted(records.items()):','assert len(records)==180\nfor p,e in sorted(records.items()):')
(P/'remote_collector.py').write_text(remote)
r=subprocess.run(['ssh','-o','ConnectTimeout=15','pronghorn','/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python3.11 -I -B -'],input=remote.encode(),capture_output=True,timeout=120)
(P/'collection.json').write_bytes(r.stdout);(P/'collection.stderr').write_bytes(r.stderr)
assert r.returncode==0,(r.returncode,r.stderr)
assert len(r.stdout)+len(r.stderr)<31*1024*1024
D=json.loads(r.stdout)
for rel,b64 in D['files'].items():
 b=base64.b64decode(b64);assert len(b)<=4194304
 dest=P/'raw'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b)
 assert hashlib.sha256(b).hexdigest()==next(i['sha256'] for i in D['inventory'] if i['path']==rel)
print(json.dumps({'wire_bytes':len(r.stdout)+len(r.stderr),'file_bytes':D['retrieved_file_bytes'],'runtime_files':len(D['runtime']),'mismatches':[x for x in D['runtime']+D['runtime_parents'] if x.get('error') or x.get('differences')]},indent=2))
