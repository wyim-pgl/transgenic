"""Read-only corpus audit; in-memory builder replay, no database writes.
Counterfactual toggles only no-CDS retention at the existing cap-exclusion point.
"""
import collections as C, csv, hashlib, json, pathlib, random, sys, types
import duckdb
R=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1')
S=R/'source'; A=R/'audit'
def load(path,name,text=None):
 m=types.ModuleType(name); m.__file__=str(path); sys.modules[name]=m
 exec(compile(text or path.read_text(),str(path),'exec'),m.__dict__); return m
b=load(S/'src/transgenic/datasets/build_b5.py','audit_builder'); gc=b.gc
COLS=['geneModel','start','fin','strand','chromosome','sequence','gff','split','strict_holdout','is_rc','gsf_token_count','contig_boundary','n_transcripts','train_weight','qc_flags']
def signature(d):
 vals=[d[k] for k in COLS]; vals[5]=hashlib.md5(vals[5].encode()).hexdigest()
 return hashlib.sha256(json.dumps(vals,separators=(',',':')).encode()).hexdigest()
def read_rows(con):
 cols=[f'md5(sequence) AS sequence' if c=='sequence' else '"'+c+'"' for c in COLS]
 out={}
 for row in con.execute('SELECT '+','.join(cols)+' FROM geneList').fetchall():
  out[row[0]]=hashlib.sha256(json.dumps(row,separators=(',',':')).encode()).hexdigest()
 return out
class Capture:
 def __init__(self): self.rows={}; self.members=set(); self.keys=[]; self.blocks=[]; self.result=[]
 def sql(self,*a,**kw): return self
 def execute(self,q,p=None):
  if q.startswith('INSERT INTO geneList'):
   d=dict(zip(b.LEGACY_COLUMNS+b.NEW_COLUMNS,p)); self.rows[d['geneModel']]=signature(d)
  elif q.startswith('INSERT INTO gene_key_map'): self.keys.append(tuple(p))
  elif q.startswith('SELECT chromosome'): self.result=[(r[4],r[5],r[6]) for r in self.keys]
  return self
 def executemany(self,q,rows):
  if q.startswith('INSERT INTO window_genes'): self.members.update(tuple(r) for r in rows)
  elif q.startswith('INSERT INTO tile_blocks'): self.blocks.extend(tuple(r) for r in rows)
  return self
 def fetchall(self): return self.result

def replay(sp,con,counterfactual=False):
 cur=con.execute('SELECT * FROM build_manifest'); manifest=dict(zip([x[0] for x in cur.description],cur.fetchone()))
 genome=b.load_fasta(manifest['fasta'])
 assert b.sha256(manifest['fasta'])==manifest['fasta_sha256']
 assert b.sha256(manifest['gff'])==manifest['gff_sha256']
 splits,splitsha=b.read_split_table(str(S/'data/splits/b5_orthogroup_split_v1.tsv'))
 qcs=b.read_qc_flags([f'/home/pgl/scratch1/wyim/transgenic_data/qc/{sp}.geenuff_flags.tsv', f'/home/pgl/scratch1/wyim/transgenic_data/swissprot_v3/{sp}.swissprot_flags.tsv'])
 stats=C.Counter(); fates={}; no_cds={}; eligible={}; population={}
 # Hooks observe the snapshot's local values; they do not change build choices.
 def observe(stage,v):
  if stage=='population':
   meta=v['gene_meta']; population.update(denominator=v['n_genes'],numerator=v['n_masked'],decoy_rate=v['decoy_rate'])
   eligible.update({g.gene_id:g for genes in v['by_chrom'].values() for g in genes})
   no_cds.update({k:m for k,m in meta.items() if not m['labelable']})
   return
  key=f"{sp}:{v['chrom']}:{v['ws']}-{v['we']}"
  if stage=='candidate': stats['candidate_windows']+=1
  elif stage=='sampled_out': stats['empty_sampled_out']+=1; fates[key]='empty_sampling'
  elif stage=='cap': stats['window_cap_rejected']+=1; fates[key]='cap:'+str(v['e'])
  elif stage=='mask_drop': stats['mask_fraction_rejected']+=1; fates[key]='mask_fraction'
  elif stage=='retained':
   fates[key]='retained'; stats['forward_retained']+=1
   seq=v['seq']; ws=v['ws']; masked_ids=v['masked_ids']; labelled_ids={g.gene_id for g in v['labelled']}
   assert not (masked_ids & labelled_ids),(sp,key,'masked_label')
   assert not (set(no_cds) & labelled_ids),(sp,key,'no_cds_label')
   for name in ('leak','hard','decoy','masked_ids'): stats[name]+=len(v[name])
   stats['component_added']+=len(masked_ids)-len(v['seed_ids'])
   stats['dup_collapsed']+=v['dup_collapsed']
   for g in v['masked']:
    assert set(seq[g.start0-ws:g.end0-ws]) <= {'N'},(key,g.gene_id,'mask_coverage')
   for g in v['inside']:
    if g.gene_id in no_cds:
     stats['no_cds_complete_occurrences']+=1
     if g.gene_id in v['leak']: stats['no_cds_leak']+=1
     if g.gene_id in v['hard']: stats['no_cds_hard_only']+=1
     if g.gene_id in v['seed_ids']: stats['no_cds_seed_union']+=1
     if g.gene_id in masked_ids: stats['no_cds_N_covered_occurrences']+=1
   stats['N_bases']+=seq.count('N'); stats['bases']+=len(seq)
 text=(S/'src/transgenic/datasets/build_b5.py').read_text()
 if counterfactual:
  assert text.count('if not no_cds:\n                    continue')==1
  text=text.replace('if not no_cds:\n                    continue','if True:\n                    continue')
 hooks=[('    block_rng =', '    observe("population", locals())\n    block_rng ='),
 ('                if not inside and', '                observe("candidate", locals())\n                if not inside and'),
 ('if not inside and rng.random() > gc.EMPTY_KEEP_PROB:\n                    continue','if not inside and rng.random() > gc.EMPTY_KEEP_PROB:\n                    observe("sampled_out", locals())\n                    continue'),
 ('rejected.append({"gene_id": f"{chrom}:{ws}-{we}", "reason": f"window {e}"})','observe("cap", locals())\n                        rejected.append({"gene_id": f"{chrom}:{ws}-{we}", "reason": f"window {e}"})'),
 ('                    rejected.append({"gene_id": f"{chrom}:{ws}-{we}", "reason": f"masked fraction', '                    observe("mask_drop", locals())\n                    rejected.append({"gene_id": f"{chrom}:{ws}-{we}", "reason": f"masked fraction'),
 ('                weight = 1.0','                observe("retained", locals())\n                weight = 1.0')]
 for old,new in hooks:
  assert text.count(old)==1,(old,text.count(old)); text=text.replace(old,new)
 m=load(S/'src/transgenic/datasets/build_b5.py','replay_builder',text); m.observe=observe
 cap=Capture(); rejected=[]
 result=m._build_species_tiles(cap,sp,manifest['fasta'],manifest['gff'],splits,splitsha,'isoform-only',123,False,'train','',manifest['fasta_sha256'],manifest['gff_sha256'],False,qcs,genome,random.Random(123),rejected,0.3)
 stats['rows']=result['rows']; stats['rc_rows']=result['rc_rows']
 stats['no_cds_loci']=len(no_cds)
 population['no_cds_by_split']=dict(C.Counter(v['split'] for v in no_cds.values()))
 population['no_cds_hard']=sum(m['weight']==0 for m in no_cds.values())
 population['no_cds_union_numerator']=sum(m['weight']==0 or gc.SPLIT_RANK.get(m['split'] or 'train',0)>0 for m in no_cds.values())
 return cap,dict(stats),fates,population,rejected

out={'method':'Snapshot replay into in-memory row fingerprints, membership and key maps; only counterfactual edit removes no-CDS retention. Current replay must match every current source row, membership, key map, block and rejection before causal attribution.','species':{},'status':'RUNNING'}
(A/'reconciliation.json').write_text(json.dumps(out,indent=2))
for path in sorted((R/'full').glob('*.db')):
 sp=path.stem; print('replay',sp,flush=True)
 new=duckdb.connect(str(path),read_only=True); old=duckdb.connect(f'/home/pgl/scratch1/wyim/transgenic_data/b5/full/{sp}.db',read_only=True)
 nr=read_rows(new); oldrows=read_rows(old)
 cap,stats,fates,pop,rej=replay(sp,new)
 diff=[k for k in set(nr)|set(cap.rows) if nr.get(k)!=cap.rows.get(k)]
 item={'old_rows':len(oldrows),'new_rows':len(nr),'delta':len(nr)-len(oldrows),'current_replay_row_mismatches':len(diff),'first_mismatches':diff[:10], 'current_stats':stats,'current_population':pop}
 out['species'][sp]=item
 checks={'rows':not diff,'membership':cap.members==set(new.execute('SELECT * FROM window_genes').fetchall()),'key_map':set(cap.keys)==set(new.execute('SELECT * FROM gene_key_map').fetchall()),'blocks':set(cap.blocks)==set(new.execute('SELECT * FROM tile_blocks').fetchall()),'rejections':C.Counter((r['gene_id'],r['reason']) for r in rej)==C.Counter(new.execute('SELECT gene_id,reason FROM rejected_records').fetchall()),'old_key_map_unchanged':set(cap.keys)==set(old.execute('SELECT * FROM gene_key_map').fetchall())}
 item['checks']=checks
 (A/'reconciliation.json').write_text(json.dumps(out,indent=2))
 if not all(checks.values()): out['status']='FAILED'; (A/'reconciliation.json').write_text(json.dumps(out,indent=2)); raise SystemExit(f'{sp}: replay check failed {checks}')
 baseline=cap
 if pop['no_cds_by_split']:
  print('counterfactual',sp,flush=True)
  baseline,bstats,bfates,bpop,brej=replay(sp,new,True)
  item.update(counterfactual_stats=bstats,counterfactual_population=bpop)
  with open(A/f'{sp}.population-row-changes.tsv','w') as fh:
   w=csv.writer(fh,delimiter='\t'); w.writerow(['geneModel','change','counterfactual_fate','current_fate'])
   for k in sorted(set(baseline.rows)|set(nr)):
    if baseline.rows.get(k)!=nr.get(k):
     coord=k.removesuffix('-rc'); w.writerow([k,'added' if k not in baseline.rows else 'removed' if k not in nr else 'changed',bfates.get(coord,'grid_absent'),fates.get(coord,'grid_absent')])
 else: bfates=fates
 ordering=old.execute("SELECT gene_id,reason FROM rejected_records WHERE reason LIKE '%canonical order%'").fetchall()
 ordering_keys={f'{sp}:{coord}' for coord,_ in ordering}
 recovered=[k for k in baseline.rows if k not in oldrows]
 removed=[k for k in oldrows if k not in baseline.rows]
 other_added=[k for k in recovered if k.removesuffix('-rc') not in ordering_keys]
 changed=[k for k in set(oldrows)&set(baseline.rows) if oldrows[k]!=baseline.rows[k]]
 item.update(ordering_forward=sum(not k.endswith('-rc') for k in recovered if k.removesuffix('-rc') in ordering_keys),ordering_rc=sum(k.endswith('-rc') for k in recovered if k.removesuffix('-rc') in ordering_keys),unexplained_added=len(other_added),unexplained_removed=len(removed),unexplained_shared_semantic_changes=len(changed),mask_population_added=sum(k not in baseline.rows for k in nr),mask_population_removed=sum(k not in nr for k in baseline.rows),mask_population_shared_changed=sum(nr[k]!=baseline.rows[k] for k in set(nr)&set(baseline.rows)))
 item['ordering_fates']=[{'coordinate':coord,'old_reason':reason,'ordering_only_fate':bfates.get(f'{sp}:{coord}','grid_absent'),'current_fate':fates.get(f'{sp}:{coord}','grid_absent'),'current_forward':f'{sp}:{coord}' in nr,'current_rc':f'{sp}:{coord}-rc' in nr} for coord,reason in ordering]
 item['unexplained_examples']={'added':other_added[:10],'removed':removed[:10],'changed':changed[:10]}
 (A/'reconciliation.json').write_text(json.dumps(out,indent=2))
 if other_added or removed or changed:
  out['status']='FAILED_UNEXPLAINED'; (A/'reconciliation.json').write_text(json.dumps(out,indent=2)); raise SystemExit(f'{sp}: unexplained non-population differences')
 new.close(); old.close(); print(sp,item['delta'],'explained',flush=True)
out['status']='PASS'; (A/'reconciliation.json').write_text(json.dumps(out,indent=2))
