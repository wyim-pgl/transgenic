"""Verify fetched bounded bytes only; never invoke gate.verify_publication here."""
import collections,csv,hashlib,json,math,pathlib,subprocess
H=pathlib.Path(__file__).resolve().parent; O=H/'results/SRR28341943'; L=H/'release'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def md5(p): return hashlib.md5(p.read_bytes()).hexdigest()
r=json.loads((O/'receipt.json').read_text()); a=json.loads((O/'orientation/audit.json').read_text())
assert json.loads((L/'slurm-6165555.out').read_text())==r
assert (L/'slurm-6165555.out').read_bytes()==(O/'receipt.json').read_bytes()+b'\n'
assert (L/'slurm-6165555.err').stat().st_size==0
assert (O/'COMPLETE').read_text()==sha(O/'receipt.json')+'\n'
assert 'FAILED_ABSENT' in (H/'remote_stat_accounting.txt').read_text()
for n,b in r['outputs'].items(): assert (O/n).stat().st_size==b['size'] and sha(O/n)==b['sha256'],n
assert json.loads((O/'commands.json').read_text())==r['commands']
assert json.loads((O/'orientation_audit.stdout').read_text())==a==r['audit']
pins=json.loads((L/'source_pins.json').read_text())
assert pins['commit']==r['source_commit']=='b660bf1e7dc48287ca34a0d6e3aeb53d45f599b6'
for n,s in pins['files'].items():
    assert sha(L/'frozen'/n)==s==sha(H.parent/'frozen'/n)
    assert hashlib.sha256(subprocess.check_output(['git','show',pins['commit']+':'+n],cwd=H.parent)).hexdigest()==s
for n in ['gate.py','source_pins.json','authority.tsv','orientation_pilot.sbatch']:
    assert (L/n).read_bytes()==(H.parent/n).read_bytes()
assert sha(L/'gate.py')=='0b5be5e4863a348f0fb35a946e6d989980428483a72055e4d3f12492b7f7a17f'
assert sha(L/'orientation_pilot.sbatch')=='e54378e5f4d86659d6ba44d4fe8ef88c53ca6b57b6ff5f0e939519268bb39832'
auth=r['reference_gate']; assert sha(L/'authority.tsv')==auth['manifest_sha256']
assert subprocess.check_output(['git','show',auth['authority']],cwd=H.parent)==(L/'authority.tsv').read_bytes()
row=[x for x in csv.DictReader((L/'authority.tsv').read_text().splitlines(),delimiter='\t') if x['species_id']=='Athaliana'][0]
for k in ['fasta','gff']: assert auth['files'][k]['path']==row[k] and auth['files'][k]['md5']==row[k+'_md5']
for b in r['inputs']:
    if '/release/' in b['path']:
        p=L/b['path'].split('/release/',1)[1]
        assert sha(p)==b['sha256'] and p.stat().st_size==b['size']
b=r['read_binding']; role=b['role']; assert b['reads_md5']==role['local_fa_md5']=='d802a86ae1673909ee21a0cd663c99ac'
assert b['reads']['path']==r['root']+'/evidence/'+role['dataset']+'/'+r['run']+'.fa.gz'
assert role['role']=='b1_validation_only' and role['species']=='Athaliana' and role['genotype_stratum']=='reference'
key=':'.join([b['reads_md5'],row['fasta_md5'],row['gff_md5'],md5(L/'frozen/revision/scripts/67_orientation_audit.py')])
assert key==r['orientation_key']
assert (O/'orientation/DONE').read_text()=='key='+key+'\nrun='+r['run']+'\naudit_md5='+md5(O/'orientation/audit.json')+'\n'
s=json.loads((O/'sample.stdout').read_text()); ids=[x.split()[0] for x in (O/'sample.fa').read_text().splitlines() if x.startswith('>')]
assert len(ids)==s['sampled']==10000 and s['run_records']==int(role['expected_reads'])==6952504
assert hashlib.md5(''.join(ids).encode()).hexdigest()==s['sample_ids_md5']
assert s['seed']==int(b['reads_md5'][:16],16) and s['short_run'] is False
c=a['counters']; k=a['read_sense']; n=a['valid_for_orientation']; p=k/n; z=1.96; d=1+z*z/n; center=p+z*z/(2*n); radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))
assert [round((center-radius)/d,6),round((center+radius)/d,6)]==a['wilson95']
assert k==c['read_sense']==2679 and n==c['agree']==c['valid_for_orientation']==a['agree']==5008
assert a['fraction']==a['sense_read_fraction']==p and a['motif_annotation_agreement']==5008/5025
assert c['eligible']==a['eligible']==5025
assert sum(c[x] for x in ['unmapped','secondary_or_supplementary','low_mapq','unspliced','no_ts_tag','no_qualifying_gene','ambiguous_gene','no_intron_inside_gene','eligible'])==c['records']==10129
assert a['min_eligible']==1000 and a['threshold']==.95 and a['min_mapq']==20 and a['status']=='FAIL' and a['use_uf'] is False
sam=[x.split('\t') for x in (O/'sample_alignment.stdout').read_text().splitlines() if not x.startswith('@')]
assert len(sam)==10129 and len({x[0] for x in sam})==10000
counts=collections.Counter()
for f in sam:
    flag=int(f[1]); cat='unmapped' if flag&4 else 'secondary_or_supplementary' if flag&0x900 else 'low_mapq' if int(f[4])<20 else 'unspliced' if 'N' not in f[5] else 'no_ts_tag' if not any(t.startswith('ts:A:') for t in f[11:]) else 'annotation_dependent'
    counts[cat]+=1
for cat,value in counts.items():
    if cat!='annotation_dependent': assert c[cat]==value
stages=['version_minimap2','version_samtools','version_seqkit','sample','sample_alignment','orientation_audit']
assert [x['stage'] for x in r['commands']]==stages and all(x['exit_code']==0 for x in r['commands'])
cmd={x['stage']:x['argv'] for x in r['commands']}; binary='/data/gpfs/assoc/pgl/bin/conda/conda_envs/RNASEQ_bch709/bin/'
for tool,flag in [('minimap2','--version'),('samtools','--version'),('seqkit','version')]: assert cmd['version_'+tool]==[binary+tool,flag]
versions={t:(O/('version_'+t+'.stdout')).read_text() for t in ['minimap2','samtools','seqkit']}
assert versions['minimap2']=='2.28-r1209\n' and versions['samtools'].startswith('samtools 1.3.1\n') and versions['seqkit']=='seqkit v2.13.0\n'
remote_out=r['root']+'/evidence/mvp_ont_orientation_20260916_pilot/results/'+r['run']; py='/data/gpfs/assoc/pgl/bin/conda/conda_envs/sylvan/bin/python'
assert cmd['sample']==[py,r['source']+'/revision/scripts/68_sample_fasta.py','--fasta',b['reads']['path'],'--out',remote_out+'/sample.fa','--n','10000','--md5',b['reads_md5']]
assert cmd['sample_alignment']==[binary+'minimap2','-t','4','-ax','splice','-k14','-G','200000','--secondary=no','--cs=long',row['fasta'],remote_out+'/sample.fa']
assert cmd['orientation_audit']==[py,r['source']+'/revision/scripts/67_orientation_audit.py','--sam',remote_out+'/sample_alignment.stdout','--gff',row['gff'],'--out',remote_out+'/orientation/audit.json','--run',r['run'],'--species','Athaliana','--dataset',role['dataset']]
assert r['job']=='6165555' and r['host']=='cpu-50' and r['scope']=='orientation_sample' and r['structural_status']=='UNASSESSABLE' and r['training_eligible'] is False and r['decision'] is None
result=dict(status='BOUNDED_LOCAL_CONSISTENCY_VERIFIED_NOT_PILOT_COMPLETE',receipt_sha256=sha(O/'receipt.json'),outputs_rehashed=len(r['outputs']),output_bytes=sum(x['size'] for x in r['outputs'].values()),source_files_git_verified=len(pins['files']),versions=versions,sample=s,sam_filter_counts=dict(counts),sense_fraction=p,wilson95=a['wilson95'],orientation_key=key,audit_md5=md5(O/'orientation/audit.json'),limitations=['No fresh canonical-read/reference/tool hash or whole-run recount','No annotation-dependent audit recomputation','Extern OUT_OF_MEMORY 0:125 blocks clean-scheduler acceptance'])
(H/'verification.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
