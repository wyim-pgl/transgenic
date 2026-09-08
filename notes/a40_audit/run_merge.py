import datetime, hashlib, json, pathlib, subprocess
R=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4194304),b''): h.update(b)
 return h.hexdigest()
j=json.load(open(R/'provenance/merge-invocation.json'))
argv=j['argv']; argv[argv.index('--manifest')+1]=str(R/'merged/b5_full_a40_v1.merge-receipt.json')
argv+=['--build-status',str(R/'provenance/build-status.json')]
cwd=R/'audit/verification-source'
record={'argv':argv,'cwd':str(cwd),'started':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'status':'RUNNING','adopted':False}
record['source_sha256']={str(p.relative_to(cwd)):sha(p) for p in cwd.rglob('*.py')}
for rel,h in json.load(open(R/'provenance/source-sha256.json')).items():
 assert sha(R/'source'/rel)==h,rel
for p,h in j['qc_sha256'].items(): assert sha(p)==h,p
out=R/'audit/merge-status-v2.json'
out.write_text(json.dumps(record,indent=2))
with open(R/'audit/merge-v2.log','w') as log:
 code=subprocess.call(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
record.update(exit_status=code,status='PASS' if code==0 else 'FAILED',finished=datetime.datetime.now(datetime.timezone.utc).isoformat())
out.write_text(json.dumps(record,indent=2))
raise SystemExit(code)
