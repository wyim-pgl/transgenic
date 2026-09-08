"""Independent interval-graph closure and actual stored sequence audit for Vvinifera."""
import bisect,collections as C,json,pathlib,random,sys,types
import duckdb
R=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1'); S=R/'source'; A=R/'audit'
p=S/'src/transgenic/datasets/build_b5.py'; b=types.ModuleType('no_cds_builder'); b.__file__=str(p); sys.modules[b.__name__]=b; exec(compile(p.read_text(),str(p),'exec'),b.__dict__); gc=b.gc
sp='Vvinifera'; con=duckdb.connect(str(R/'full'/f'{sp}.db'),read_only=True,config={'memory_limit':'512MB','threads':2})
fasta,gff=con.execute('SELECT fasta,gff FROM build_manifest').fetchone(); genome=b.load_fasta(fasta)
splits,_=b.read_split_table(str(S/'data/splits/b5_orthogroup_split_v1.tsv'))
qc=b.read_qc_flags([f'/home/pgl/scratch1/wyim/transgenic_data/qc/{sp}.geenuff_flags.tsv',f'/home/pgl/scratch1/wyim/transgenic_data/swissprot_v3/{sp}.swissprot_flags.tsv'])
by_chrom=C.defaultdict(list); meta={}; original_noncds={}; stats=C.Counter(); fails=[]
with open(gff) as fh:
 for g in gc.parse_gff3(fh,species_code=gc.species_code(sp)):
  noncds=not any(f.type=='CDS' for tx in g.transcripts.values() for f in tx)
  split=splits[(sp,g.gene_id)]['split']
  if noncds: original_noncds[g.gene_id]=split
  flags=b.flags_for_gene(qc,sp,g); weight=1
  if flags:
   weight,keep,_=b.loss_mask_decision(flags,g.transcripts.keys())
   if weight>0 and len(keep)<len(g.transcripts): g=gc.Gene(g.gene_id,g.chrom,g.strand,g.start0,g.end0,{t:g.transcripts[t] for t in keep},g.gene_id_original,g.name_original)
  if g.chrom not in genome: continue
  try: gc.check_caps(gc.gene_to_gsf(g,g.start0))
  except gc.CapError:
   if not noncds: continue
  meta[g.gene_id]=(split,weight,noncds); by_chrom[g.chrom].append(g)
noncds_ids=set(original_noncds)
assert len(noncds_ids)==6205 and C.Counter(original_noncds.values())=={'train':4600,'valid':634,'test':971}
assert noncds_ids <= set(meta)
all_members=con.execute('SELECT window_id,gene_id,is_rc FROM window_genes').fetchall(); members=C.defaultdict(set)
assert not any(g in noncds_ids for _,g,_ in all_members)
for w,g,rc in all_members: members[(w,rc)].add(g)
rc_keys={w for w, in con.execute('SELECT gene_id FROM geneList WHERE is_rc').fetchall()}
rank={'train':0,'valid':1,'test':2}
den=len(meta); num=sum(weight==0 or rank[split]>0 for split,weight,_ in meta.values()); rate=min(.05,num/den/3)
for chrom,genes in by_chrom.items(): by_chrom[chrom]=sorted(genes,key=lambda g:(g.start0,g.end0))
starts={ch:[g.start0 for g in genes] for ch,genes in by_chrom.items()}
cur=con.execute('SELECT gene_id,chromosome,start,fin,split,sequence FROM geneList WHERE NOT is_rc')
while True:
 rows=cur.fetchmany(50)
 if not rows: break
 for wid,ch,ws,we,split,seq in rows:
  gs=by_chrom[ch]; st=starts[ch]; inside=[g for g in gs[bisect.bisect_left(st,ws):bisect.bisect_left(st,we)] if g.end0<=we]
  leak={g.gene_id for g in inside if rank[meta[g.gene_id][0]]>rank[split]}
  hard={g.gene_id for g in inside if meta[g.gene_id][1]==0}
  decoy=set(); drng=random.Random(f'123:{sp}:{ch}:{ws}-{we}:decoy')
  if split=='train':
   for g in inside:
    if not meta[g.gene_id][2] and g.gene_id not in leak|hard and drng.random()<rate: decoy.add(g.gene_id)
  seeded=leak|hard|decoy
  # Fixed-point graph expansion is independent of the builder's sorted sweep components.
  masked={g.gene_id for g in inside if g.gene_id in seeded}
  while True:
   old=set(masked)
   for g in inside:
    if any(h.gene_id in masked and g.start0<h.end0 and h.start0<g.end0 for h in inside): masked.add(g.gene_id)
   if old==masked: break
  assert not masked & members[(wid,False)],(wid,'masked component has label')
  assert not noncds_ids & decoy
  for g in inside:
   if g.gene_id in masked:
    assert set(seq[g.start0-ws:g.end0-ws])<={'N'},(wid,g.gene_id,'missing N coverage')
   if g.gene_id in noncds_ids:
    stats['complete_occurrences_forward']+=1; stats['complete_occurrences_RC']+=wid in rc_keys
    for name,ids in [('leak',leak),('hard',hard),('hard_only',hard-leak),('leak_hard_union',leak|hard),('component_masked',masked)]:
     if g.gene_id in ids: stats[name+'_forward']+=1; stats[name+'_RC']+=wid in rc_keys
  stats['forward_windows_checked']+=1
out={'status':'PASS','source_no_cds':len(noncds_ids),'source_by_split':dict(C.Counter(original_noncds.values())),'eligible_no_cds':len(noncds_ids & set(meta)),'coding_memberships':0,'decoy_memberships':0,'population_denominator':den,'population_numerator':num,'decoy_rate':rate,'checks':'Independent fixed-point interval closure for every retained Vvinifera forward window; actual stored N coverage across each masked complete locus; RC coverage follows exhaustive stored RC sequence equality in ledger. Partial edge loci are outside the existing complete-locus guarantee.','violations':[], 'occurrences':dict(stats)}
(A/'no-cds-independent.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
