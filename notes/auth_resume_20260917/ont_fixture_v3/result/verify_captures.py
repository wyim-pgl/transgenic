"""Offline verification: python3 -B verify_captures.py. Writes nothing."""
import pathlib,json,hashlib
P=pathlib.Path(__file__).resolve().parent
D=json.loads((P/'collection.json').read_text())
I={x['path']:x for x in D['inventory']}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
pins={'release/MANIFEST.sha256':'2cd93e7b93182a51dc9198d4f697af7b9f36b826e0286d1538f26c96af09b904','controller_approval.json':'84e596d5360b9049f13a2b8fcea34d77484d9494f551532e45b40d8f7cdec91f','release/runtime_evidence/expected_closure.json':'17028c5096c789ec7d198c489d08eec8cbc0224ec1ebd492115a475d9ee243ce'}
for path,h in pins.items():assert sha(P/'raw'/path)==I[path]['sha256']==h
for path in D['files']:assert sha(P/'raw'/path)==I[path]['sha256']
lines=(P/'raw/release/MANIFEST.sha256').read_text().splitlines()
for line in lines:
 h,name=line.split(maxsplit=1);name=name.lstrip('*')
 assert I['release/'+name]['sha256']==h
 assert sha(P.parent/name)==h
closure=json.loads((P/'raw/release/runtime_evidence/expected_closure.json').read_text())
actual={r['path']:r for r in D['runtime']};parents={r['path']:r for r in D['runtime_parents']}
count=0
def walk(x):
 global count
 if isinstance(x,dict):
  if 'path' in x and 'sha256' in x:
   count+=1;r=actual[x['path']];assert not r.get('error') and not r['differences']
   for k,v in r['actual'].items():
    if k in x:assert x[k]==v,(x['path'],k)
   for p in x.get('parents',[]):
    a=parents[p['path']];assert not a.get('error') and not a['differences']
    for k,v in a['actual'].items():assert p[k]==v
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
walk(closure)
assert I['out']['type']=='directory'
output=[r for r in I.values() if r['path']=='out' or r['path'].startswith('out/') or r['path'] in ('stdout','stderr')]
logical=sum(r['logical'] for r in output);allocated=sum(r['allocated'] for r in output)
assert logical<268435456 and allocated<268435456
assert not [r for r in I.values() if r['path'].startswith('out/')]
print(json.dumps({'captured_at':D['time'],'pins_verified':pins,'manifest_entries_verified_local_and_remote':len(lines),'closure_file_records_verified':count,'unique_runtime_files':len(actual),'explicit_parent_paths':len(parents),'output_and_logs_inventory':output,'logical_bytes_including_out_directory':logical,'allocated_bytes_including_out_directory':allocated,'threshold_bytes':268435456,'all_retrieved_hashes_verified':True},indent=2))
