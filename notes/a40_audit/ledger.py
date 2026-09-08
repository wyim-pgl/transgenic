import collections as C, csv, hashlib, json, pathlib, subprocess, sys, types
import duckdb
R=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1'); S=R/'source'; A=R/'audit'; DB=R/'merged/b5_full_a40_v1.db'
def load(path,name):
 m=types.ModuleType(name); m.__file__=str(path); sys.modules[name]=m
 exec(compile(path.read_text(),str(path),'exec'),m.__dict__); return m
b=load(S/'src/transgenic/datasets/build_b5.py','ledger_builder'); gc=b.gc
out={'status':'RUNNING','checks':{}}
def save(): (A/'ledger.json').write_text(json.dumps(out,indent=2))
def check(name,value):
 out['checks'][name]=value; save()
 if not value: out['status']='FAILED'; save(); raise SystemExit('FAILED '+name)
validation=b.validate_b5_database(str(DB)); out['merged_validation']=validation; check('merged_validator',validation['ok'] and not validation['violations'])
cmd=[sys.executable,str(S/'scripts/report_b5_database.py'),'--db',str(DB),'--reference-distributions','--out',str(A/'population-v2.json'),'--tables-dir',str(A/'population-tables')]
with open(A/'population-v2.log','w') as fh: code=subprocess.call(cmd,stdout=fh,stderr=subprocess.STDOUT)
check('population_report_exit',code==0)
pop=json.load(open(A/'population-v2.json')); check('population_report_all_checks',all(pop['checks'].values()))
con=duckdb.connect(str(DB),read_only=True); old=duckdb.connect('/home/pgl/scratch1/wyim/transgenic_data/b5/merged/b5_full_v1.db',read_only=True)
for name,c in [('old',old),('new',con)]:
 out[name]={}
 for label,q in [('empty_by_split',"SELECT split,count(*) FROM geneList WHERE gff='<empty>' GROUP BY 1 ORDER BY 1"),('token_by_species',"SELECT species_id,max(gsf_token_count),sum(gsf_token_count) FROM geneList GROUP BY 1 ORDER BY 1"),('N_by_species',"SELECT species_id,sum(length(sequence)-length(replace(sequence,'N',''))),sum(length(sequence)) FROM geneList GROUP BY 1 ORDER BY 1"),('original_genes',"SELECT count(*),max(end0-start0) FROM gene_key_map"),('rows_by_species_split_tier_orientation',"SELECT species_id,split,fin-start,is_rc,count(*) FROM geneList GROUP BY 1,2,3,4 ORDER BY 1,2,3,4")]: out[name][label]=c.execute(q).fetchall()
 with open(A/f'{name}-rows-species-split-tier-orientation.tsv','w') as fh:
  w=csv.writer(fh,delimiter='\t'); w.writerow(['species','split','tier','is_rc','rows']); w.writerows(out[name]['rows_by_species_split_tier_orientation'])
check('no_null_labels',con.execute('SELECT count(*) FROM geneList WHERE gff IS NULL').fetchone()[0]==0)
check('rc_has_forward',con.execute('SELECT count(*) FROM geneList r LEFT JOIN geneList f ON r.species_id=f.species_id AND r.gene_id=f.gene_id AND NOT f.is_rc WHERE r.is_rc AND f.rn IS NULL').fetchone()[0]==0)
check('rc_membership',con.execute('SELECT count(*) FROM ((SELECT w.* EXCLUDE(is_rc) FROM window_genes w WHERE w.is_rc EXCEPT SELECT w.* EXCLUDE(is_rc) FROM window_genes w JOIN geneList r ON r.species_id=w.species_id AND r.gene_id=w.window_id AND r.is_rc WHERE NOT w.is_rc) UNION ALL (SELECT w.* EXCLUDE(is_rc) FROM window_genes w JOIN geneList r ON r.species_id=w.species_id AND r.gene_id=w.window_id AND r.is_rc WHERE NOT w.is_rc EXCEPT SELECT w.* EXCLUDE(is_rc) FROM window_genes w WHERE w.is_rc))').fetchone()[0]==0)
check('rc_sequence_and_split',con.execute("SELECT count(*) FROM geneList r JOIN geneList f ON r.species_id=f.species_id AND r.gene_id=f.gene_id AND NOT f.is_rc WHERE r.is_rc AND (r.sequence<>reverse(translate(f.sequence,'ACGTNacgtn','TGCANtgcan')) OR r.split<>f.split OR r.strict_holdout<>f.strict_holdout)").fetchone()[0]==0)
offset=0
for p in sorted((R/'full').glob('*.db')):
 con.execute("ATTACH '"+str(p)+"' AS src (READ_ONLY)")
 bad=con.execute('SELECT count(*) FROM (SELECT geneModel,row_number() OVER (ORDER BY rn)+? expected FROM src.geneList) s JOIN geneList m ON m.species_id=? AND m.geneModel=s.geneModel WHERE m.rn<>s.expected',[offset,p.stem]).fetchone()[0]
 check('deterministic_rn_'+p.stem,bad==0)
 offset+=con.execute('SELECT count(*) FROM src.geneList').fetchone()[0]; con.execute('DETACH src')
tok=load(S/'src/transgenic/model/tokenization_transgenic.py','audit_tokenizer').GFFTokenizer(vocab_version='v3')
counts=C.Counter(); maxima=C.Counter()
cur=con.execute('SELECT species_id,geneModel,gff,fin-start,gsf_token_count FROM geneList')
for sp,gm,label,L,stored in cur.fetchall():
 try:
  tokens=tok._tokenize(label); ids=[tok._convert_token_to_id(t) for t in tokens]
  assert len(ids)==stored and len(ids)<=8192 and tok.unk_token_id not in ids
  assert gc.canonicalize_v3(label)==label
  gc.check_caps_v3(label,window_len=L)
  assert gc.reverse_complement_v3(gc.reverse_complement_v3(label,L),L)==label
  if label=='<empty>': assert tokens==['<s>','<empty>','</s>']; counts['empty']+=1
 except Exception as e:
  out['first_bad_label']={'species':sp,'geneModel':gm,'error':repr(e)}; check('all_actual_tokenizer_canonical_caps_involution',False)
 counts['rows']+=1; counts['tokens']+=len(ids); maxima[sp]=max(maxima[sp],len(ids))
out['actual_tokenizer']=dict(counts); out['actual_token_maxima']=dict(maxima); check('all_actual_tokenizer_canonical_caps_involution',True)
# Exercise the unchanged dataset __getitem__ decoder path for every actual empty row.
# Cache the already-read database row; the encoder stub avoids model download and tests no encoder behavior.
sys.path.insert(0,str(S/'src'))
import torch
from transgenic.datasets.datasets import isoformDataHyena
class Cached:
 def sql(self,*a,**k): return self
 def fetchall(self): return [self.row]
cache=Cached()
class Encoder:
 pad_token_id=4
 def __call__(self,*a,**kw): return {'input_ids':torch.zeros((1,2),dtype=torch.long)}
ds=isoformDataHyena.__new__(isoformDataHyena); ds._index_map=[1]; ds._get_connection=lambda:cache; ds.dt=tok; ds.mode='train'; ds.maxlength=8192; ds.strict=True; ds.encoder_tokenizer=Encoder(); ds._con=None
expected=[tok.vocab[t] for t in ['<s>','<empty>','</s>']]
n=0
cur=con.execute("SELECT geneModel,start,fin,strand,chromosome,sequence,gff,static_fpb,static_tpb,five_prime_buf,three_prime_buf,rn FROM geneList WHERE gff='<empty>'")
while True:
 rows=cur.fetchmany(100)
 if not rows: break
 for row in rows:
  cache.row=row; ds._index_map=[row[-1]]
  check_value=ds[0][2].tolist()==[expected]
  if not check_value: check('all_empty_dataset_targets_three_tokens',False)
  n+=1
out['actual_empty_dataset_rows_checked']=n; check('all_empty_dataset_targets_three_tokens',n==counts['empty'])
old.close(); con.close(); out['status']='PASS'; save(); print('PASS',dict(counts),flush=True)
