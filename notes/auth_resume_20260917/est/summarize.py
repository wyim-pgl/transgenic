"""Summarize captured receipts only, never original data."""
from pathlib import Path
import json, hashlib, base64
H=Path(__file__).resolve().parent
captures=[json.loads(p.read_text()) for p in sorted(H.glob('*.capture.json'))]
rows={r['path']:r for c in captures for r in c['records']}
base='/data/gpfs/assoc/pgl/data/Transgenic/evidence/est_blastx/remaining8_20260915/Ptrichocarpa/'
summary={'status':'candidate_evidence_not_registered_or_accepted','remote_read_bytes':sum(c['read_bytes'] for c in captures),'producers':[]}
for source in ['orthodb','swissprot']:
    root=base+source+'/'
    p=json.loads(rows[root+'provenance.json']['text'])
    commands=json.loads(rows[root+'commands.json']['text'])
    log=rows[root+'blastx.log']['text']
    item={'source':source,'provenance':{k:v for k,v in p.items() if k not in ('counts_by_taxid','retained_taxids','excluded_taxids') and not isinstance(v,list)},'command_argv':commands,'evidence':{k:{f:rows[root+k][f] for f in ['path','size','sha256']} for k in ['provenance.json','commands.json','blastx.log','makedb.log']},'dispatch_log_lines':[l for l in log.splitlines() if any(x in l for x in ['Database:','Block size','Algorithm:','reference block 1/','reference block 3/','Joining output blocks','Reported ','queries aligned.'])]}
    # Omit lengthy taxonomy structures regardless of producer version.
    item['provenance']={k:v for k,v in item['provenance'].items() if not isinstance(v,dict) or k=='inputs'}
    summary['producers'].append(item)
(H/'producer_candidates.json').write_text(json.dumps(summary,indent=2)+'\n')
parent=H.parents[1]/'seed123_recovery_20260917_1400/est_result/raw/worker/sample'
# Only bounded prefix of an already retained diagnostic receipt, not source data.
p=parent/'diagnostics.jsonl'
with p.open('rb') as f: prefix=f.read(1048576)
lines=prefix.split(b'\n')[:-1]
member=next((json.loads(l) for l in lines if json.loads(l).get('type')=='member'),None)
report_path=parent/'report.json'
assert report_path.stat().st_size<=1048576
report_bytes=report_path.read_bytes(); report=json.loads(report_bytes)
r={'source_receipt':str(p),'receipt_total_size':p.stat().st_size,'prefix_read_bytes':len(prefix),'prefix_sha256':hashlib.sha256(prefix).hexdigest(),'report_path':str(report_path),'report_sha256':hashlib.sha256(report_bytes).hexdigest(),'reported_diagnostics_sha256':report.get('diagnostics_sha256'),'member':member,'note':'Extracted retained receipt only; whole diagnostics not rehashed and no raw source access.'}
if member:
    r['decoded_headers']={k:base64.b64decode(member[k]['header']['bytes_base64']).decode() for k in ['original','screened']}
(H/'retained_member_sample.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(summary,indent=2))
print(json.dumps({'member_found':member is not None,'decoded_headers':r.get('decoded_headers'),'member_identity':member.get('identity') if member else None},indent=2))
