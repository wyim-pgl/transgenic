"""Local capture consistency checks, not source-data certification."""
from pathlib import Path
import hashlib,json,re,base64
H=Path(__file__).resolve().parent
captures=[json.loads(p.read_text()) for p in sorted(H.glob('*.capture.json'))]
reads=[r for c in captures for r in c['records'] if r['action']=='read']
assert not [r for c in captures for r in c['records'] if r.get('error')]
for r in reads:
    b=r['text'].encode()
    assert len(b)==r['size']<=1048576
    assert hashlib.sha256(b).hexdigest()==r['sha256']
assert sum(r['size'] for r in reads)==sum(c['read_bytes'] for c in captures)<=16777216
rows={r['path']:r for c in captures for r in c['records']}
s=json.loads((H/'producer_candidates.json').read_text())
for item in s['producers']:
    e=item['evidence']; p=item['provenance']; make,blast=item['command_argv']
    log=rows[e['blastx.log']['path']]['text']; mlog=rows[e['makedb.log']['path']]['text']
    def arg(cmd,key): return cmd[cmd.index(key)+1]
    assert make[0]==blast[0] and arg(make,'--db')==arg(blast,'--db')
    assert arg(blast,'--query')==p['inputs']['query']
    assert arg(make,'--in') in mlog and arg(blast,'--db') in log
    assert 'sequences: '+str(p['sequences_out'])+',' in log
    assert 'Database sequences  '+str(p['sequences_out']) in mlog
    assert 'Reported '+str(p['raw_hsps'])+' pairwise alignments' in log
    assert arg(blast,'--tmpdir') in log
    assert arg(blast,'--frameshift')=='15' and arg(blast,'--outfmt')=='6'
    assert not any(x in blast for x in ['--iterate','--multiprocessing','--global-ranking'])
    assert 'diamond v2.1.24.178' in log and 'diamond v2.1.24.178' in mlog
    assert blast[blast.index('--outfmt')+2:]==['qseqid','qlen','sseqid','slen','qstart','qend','sstart','send','length','evalue','bitscore','qframe','btop']
r=json.loads((H/'retained_member_sample.json').read_text()); m=r['member']
for k in ['original','screened']:
    b=base64.b64decode(m[k]['header']['bytes_base64'])
    assert hashlib.sha256(b).hexdigest()==m[k]['header']['sha256']
    assert hashlib.sha256(m[k]['sequence'].encode()).hexdigest()==m[k]['sequence_sha256']
assert m['original']['sequence']==m['screened']['sequence']
assert m['relationship']['start0']==0 and m['relationship']['end0']==648
assert len(m['screened']['sequence'])==648
for p in H.iterdir():
    if p.is_file(): assert '\u2014' not in p.read_text(),p
result={'status':'PASS_capture_consistency_only','remote_read_files':len(reads),'remote_read_bytes':sum(r['size'] for r in reads),'max_remote_file_bytes':max(r['size'] for r in reads),'producer_command_log_links_checked':2,'raw_data_reads':0,'member_header_report_sequence_receipt_checked':True,'application_tests_run':False}
(H/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
