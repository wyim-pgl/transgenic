import collections,json,pathlib,sys,types
import duckdb
R=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1'); S=R/'source'
p=S/'src/transgenic/datasets/build_b5.py'; b=types.ModuleType('supplement_b5'); b.__file__=str(p); sys.modules[b.__name__]=b; exec(compile(p.read_text(),str(p),'exec'),b.__dict__)
out={'checks':{},'rejections':{},'source_references':{},'nominal':{}}
for path in sorted((R/'full').glob('*.db')):
 sp=path.stem; c=duckdb.connect(str(path),read_only=True); old=duckdb.connect(f'/home/pgl/scratch1/wyim/transgenic_data/b5/full/{sp}.db',read_only=True)
 q='SELECT fasta,fasta_sha256,gff,gff_sha256,split_file_sha256,rc_mode,ordering_version,window_policy,duckdb_version FROM build_manifest'
 same=c.execute(q).fetchall()==old.execute(q).fetchall(); out['checks'][sp+'_references']=same
 out['source_references'][sp]=c.execute(q).fetchall()
 bad=c.execute("SELECT count(*) FROM geneList g JOIN build_manifest m USING(species_id) WHERE g.static_fpb<>0 OR g.static_tpb<>0 OR g.five_prime_buf<>0 OR g.three_prime_buf<>0 OR g.orthogroup_id IS NOT NULL OR g.gene_id<>CASE WHEN g.is_rc THEN left(g.geneModel,length(g.geneModel)-3) ELSE g.geneModel END OR g.gene_id_original<>g.gene_id OR g.ordering_version<>m.ordering_version OR g.window_policy<>m.window_policy OR g.source_fasta_sha256<>m.fasta_sha256 OR g.source_gff_sha256<>m.gff_sha256 OR g.split_file_sha256<>m.split_file_sha256 OR g.build_version<>m.build_version").fetchone()[0]
 out['checks'][sp+'_remaining_row_fields']=bad==0
 out['rejections'][sp]=dict(collections.Counter(b.rejection_class(reason) for reason, in c.execute('SELECT reason FROM rejected_records').fetchall()))
 out['nominal'][sp]=c.execute('SELECT split,count(*),sum(CAST(strict_holdout AS INT)) FROM gene_split GROUP BY 1 ORDER BY 1').fetchall()
 assert same and bad==0,(sp,same,bad)
 c.close(); old.close()
out['status']='PASS'; (R/'audit/supplement.json').write_text(json.dumps(out,indent=2)); print('PASS')
