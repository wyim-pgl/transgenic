import json,pathlib,sys,types
import duckdb
R=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1'); p=R/'source/src/transgenic/datasets/build_b5.py'
b=types.ModuleType('margin_b5'); b.__file__=str(p); sys.modules[b.__name__]=b; exec(compile(p.read_text(),str(p),'exec'),b.__dict__)
out={'status':'RUNNING','species':{}}
for path in sorted((R/'full').glob('*.db')):
 c=duckdb.connect(str(path),read_only=True,config={'memory_limit':'512MB','threads':2}); fasta,recorded=c.execute('SELECT fasta,tier_margin_unguaranteed FROM build_manifest').fetchone()
 genome=b.load_fasta(fasta); calculated=b.tier_margin_summary(c,path.stem,genome)
 assert calculated==json.loads(recorded),path.stem
 out['species'][path.stem]=calculated; c.close(); del genome
out['status']='PASS'; (R/'audit/margins-recomputed.json').write_text(json.dumps(out,indent=2)); print('PASS')
