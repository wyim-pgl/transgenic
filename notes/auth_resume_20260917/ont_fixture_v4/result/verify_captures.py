"""Offline evidence verification, stdlib only, writes nothing. python3 -B verify_captures.py"""
import pathlib,json,hashlib,base64,gzip,struct,csv,io,collections
P=pathlib.Path(__file__).resolve().parent
D=json.loads((P/'collection.json').read_text());I={x['path']:x for x in D['inventory']};R=P/'raw';O=R/'out'
A=O/'synthetic/evidence/mvp_scoring_20260916/SRR28341943'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
pins={'release/MANIFEST.sha256':'13da604041044f3ce0192b2049d644a603b9bbef8d14d38c70fedc07ae92d3b9','controller_approval.json':'981023667e3dc0b18a44ae62f6b8b6694dc177d84f19e5989bf26a713d804da5','release/runtime_evidence/expected_closure.json':'17028c5096c789ec7d198c489d08eec8cbc0224ec1ebd492115a475d9ee243ce'}
for p,h in pins.items():assert sha(R/p)==I[p]['sha256']==h
for p,b in D['files'].items():
 assert (R/p).read_bytes()==base64.b64decode(b)
 assert sha(R/p)==I[p]['sha256'] and (R/p).stat().st_size<=4194304
assert not any(x.get('retrieval_refused') for x in I.values())
lines=(R/'release/MANIFEST.sha256').read_text().splitlines();assert len(lines)==17
for line in lines:
 h,n=line.split(maxsplit=1);n=n.lstrip('*');assert sha(P.parent/n)==I['release/'+n]['sha256']==h
actual={x['path']:x for x in D['runtime']};parents={x['path']:x for x in D['runtime_parents']};expected={};count=0
def walk(x):
 global count
 if isinstance(x,dict):
  if 'path' in x and 'sha256' in x:
   count+=1;expected[x['path']]=x;r=actual[x['path']];assert not r.get('error') and not r['differences']
   for k,v in r['actual'].items():
    if k in x:assert x[k]==v,(x['path'],k)
   for e in x.get('parents',[]):
    a=parents[e['path']];assert not a.get('error') and not a['differences']
    for k,v in a['actual'].items():assert e[k]==v
  for v in x.values():walk(v)
 elif isinstance(x,list):
  for v in x:walk(v)
walk(load(R/'release/runtime_evidence/expected_closure.json'))
assert set(actual)==set(expected) and len(actual)==180
assert sha(O/'runtime_closure_observed.json')==pins['release/runtime_evidence/expected_closure.json']
f=load(O/'file_limit_observed.json');assert f['before']==[4194304,-1] and f['after']==[4194304,4194304]
assert json.loads((R/'stderr').read_text())['file_limit_before']==f['before']
assert load(O/'launch_environment.json')['file_limit']==f['after']
cg=load(O/'cgroup_observed.json');assert cg['version']==1
assert '/job_6165965/step_0' in cg['membership']
for h in cg['hierarchy'][:2]:
 assert h['memory.limit_in_bytes']==h['memory.memsw.limit_in_bytes']=='4294967296' and h['memory.use_hierarchy']=='1'
r=load(O/'receipt.json');assert r==json.loads((R/'stdout').read_text().splitlines()[-1])
assert (O/'COMPLETE').read_text().strip()==sha(O/'receipt.json')
assert not any('FAILED' in pathlib.PurePosixPath(p).name for p in I)
assert r['exit_code']==0 and r['status']=='FIXTURE_WORKLOAD_PASS_PENDING_CONTROLLER'
assert r['scope']=='TEST_ONLY_REAL_TOOL_FIXTURE' and not r['training_eligible'] and not r['real_fixture_controller_accepted'] and r['decision'] is None
assert len(r['seams'])==4
for section,root in [('diagnostics',O),('outputs',A)]:
 for p,e in r[section].items():assert sha(root/p)==e['sha256'] and (root/p).stat().st_size==e['size']
rows=list(csv.DictReader((A/'junctions.tsv').read_text().splitlines(),delimiter='\t'))
assert rows==[dict(chrom='fixture_chr',intron_start0='20500',intron_end0='20699',tx_strand='+',n_ONT='3',n_HiFi='0',supported='true'),dict(chrom='fixture_chr',intron_start0='30500',intron_end0='30699',tx_strand='+',n_ONT='2',n_HiFi='0',supported='false')]
names=['below_threshold_0','below_threshold_1','positive_0','positive_1','positive_2','unspliced']
assert sorted((A/'panel_reads.txt').read_text().splitlines())==names
assert sorted(x[1:].split()[0] for x in (A/'panel_reads.fa').read_text().splitlines() if x.startswith('>'))==names
rawreads=gzip.decompress((O/'synthetic/evidence/ont/Athaliana_FLIC_PRJNA1087576/SRR28341943.fa.gz').read_bytes()).decode()
inputnames=[x[1:].split()[0] for x in rawreads.splitlines() if x.startswith('>')]
assert len(inputnames)==7 and set(inputnames)-set(names)=={'outside'}
audit=load(A/'orientation/audit.json');assert audit['status']=='UNRESOLVED' and audit['use_uf'] is False and audit['valid_for_orientation']==5 and audit['min_eligible']==1000
ledger=[json.loads(x) for x in (O/'nested_execution.jsonl').read_text().splitlines()]
assert len(ledger)==9
for x in ledger:
 auditcmd=any('67_orientation_audit.py' in a for a in x['logical_argv'])
 assert x['exit_code']==(4 if auditcmd else 0)
 assert '-uf' not in x['actual_argv']
assert 'junction' in ledger[-1]['actual_argv'] and 'entry' in ledger[-1]['actual_argv']
for name,t in load(O/'runtime_observed.json')['tools'].items():assert t['actual_sha256']==t['sha256']
# Independently decode the actual BGZF/BAM payload, without invoking remote tools.
b=io.BytesIO(gzip.decompress((A/'panel.bam').read_bytes()));assert b.read(4)==b'BAM\x01'
def i32():return struct.unpack('<i',b.read(4))[0]
header=b.read(i32());refs=[]
for _ in range(i32()):refs.append((b.read(i32()).rstrip(b'\0').decode(),i32()))
bam=[];junctions=collections.defaultdict(set)
while True:
 size=b.read(4)
 if not size:break
 n=struct.unpack('<i',size)[0];v=b.read(n);assert len(v)==n
 rid,pos,binmqnl,flagnc,lseq,nrid,npos,tlen=struct.unpack_from('<iiIIiiii',v)
 nl=binmqnl&255;nc=flagnc&65535;flag=flagnc>>16;name=v[32:32+nl-1].decode()
 cigar=struct.unpack_from('<'+'I'*nc,v,32+nl);ops=[];cursor=pos
 for packed in cigar:
  length=packed>>4;op=packed&15;ops.append(str(length)+'MIDNSHP=XB'[op])
  if op==3:junctions[(refs[rid][0],cursor,cursor+length-1)].add(name)
  if op in (0,2,3,7,8):cursor+=length
 bam.append({'name':name,'flag':flag,'reference':refs[rid][0],'pos0':pos,'cigar':''.join(ops)})
assert sorted(x['name'] for x in bam)==names
assert set(junctions)=={('fixture_chr',20500,20699),('fixture_chr',30500,30699)}
assert junctions[('fixture_chr',20500,20699)]=={'positive_0','positive_1','positive_2'}
assert junctions[('fixture_chr',30500,30699)]=={'below_threshold_0','below_threshold_1'}
assert (A/'panel.bam.bai').read_bytes().startswith(b'BAI\x01')
output=[x for x in I.values() if x['path']=='out' or x['path'].startswith('out/') or x['path'] in ('stdout','stderr')]
logical=sum(x['logical'] for x in output);allocated=sum(x['allocated'] for x in output)
assert logical<268435456 and allocated<268435456
wire=sum((P/n).stat().st_size for n in ['snapshot_01.txt','snapshot_01.stderr','collection.json','collection.stderr']);assert wire<=33554432
snapshot=(P/'snapshot_01.txt').read_text();assert '6165965.extern|extern|OUT_OF_MEMORY|0:125|' in snapshot
print(json.dumps({'verified':True,'technical_workload_pass':True,'clean_F4_accepted':False,'blocker':'6165965.extern OUT_OF_MEMORY 0:125','capture_time':D['time'],'file_limit':f,'pins':pins,'manifest_entries':len(lines),'runtime_file_records':count,'unique_runtime_paths':len(actual),'runtime_parent_paths':len(parents),'compute_closure_exact_match':True,'bam_records':bam,'junction_support_names':{str(k):sorted(v) for k,v in junctions.items()},'selected_names':names,'excluded_names':sorted(set(inputnames)-set(names)),'nested_exit_codes':[x['exit_code'] for x in ledger],'seams':r['seams'],'affinity_limitation':'Four-CPU assertion completed in pinned code; actual affinity CPU list was not persisted. CPU_IDs 10-13 are scheduler evidence only.','cgroup_limitation':'Actual membership/hierarchical limits saved; launcher PID not separately persisted; no post-run reaping inspection.','receipt_sha256':sha(O/'receipt.json'),'logical_bytes_including_directories':logical,'allocated_bytes_including_directories':allocated,'regular_file_logical_bytes':sum(x['logical'] for x in output if x['type']=='file'),'regular_file_allocated_bytes':sum(x['allocated'] for x in output if x['type']=='file'),'raw_wire_bytes':wire,'retrieved_file_bytes':D['retrieved_file_bytes'],'max_retrieved_file_bytes':max((R/p).stat().st_size for p in D['files']),'output_inventory':output},indent=2))
