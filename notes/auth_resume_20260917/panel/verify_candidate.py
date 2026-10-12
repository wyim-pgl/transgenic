"""Local package checks only; no production imports or remote calls."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent
def read(n): return json.loads((H/n).read_text())
def check(path,pin):
 b=Path(path).read_bytes(); assert len(b)==pin['size']; assert hashlib.sha256(b).hexdigest()==pin['sha256']
p=read('request_pin.json')['expected_verification_request']; check(H/'candidate_request.json',p)
r=read('candidate_request.json'); h=read('historical_inputs.json')
paths=[r[k] for k in ('panels','fasta','reference')]+[r['candidate'][k] for k in ('acceptance','inventory','receipts','corpus','evidence')]
assert paths==[i['path'] for i in h['inputs'][:8]]
remote=json.loads(read('remote_metadata_explicit_python.json')['stdout'])
rows={i['path']:i for i in remote['files']}
for i in h['inputs']:
 s=rows[i['path']]; assert s['size']==i['size']; assert s['resolved']==i['path']; assert not s['wal']
c=read('consumer_closure_candidate.json')
assert len(c['consumer_closure'])==13
for i in c['consumer_closure']+[c['launcher_separate']]: check(i['path'],i)
for f in H.iterdir():
 if f.is_file(): assert '\u2014' not in f.read_text(),f
assert read('candidate_config.json')['budget_request']['approved'] is False
result=dict(status='PASS_LOCAL_CANDIDATE_BINDINGS_ONLY',historical_input_stat_matches=15,closure_files_rehashed=13,launcher_rehashed=True,request=p,actual50_run=False,remote_deployment=False)
with (H/'local_verification.json').open('x') as f: json.dump(result,f,indent=2); f.write('\n')
print(json.dumps(result,indent=2))
