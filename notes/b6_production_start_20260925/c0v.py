import json, hashlib, sys
from pathlib import Path
P=Path('/data/gpfs/assoc/pgl/data/Transgenic/evidence/b6_prepare_20260925/c0')
for spec in sys.argv[1:]:
    d_,sp,src=spec.split(':')
    q=P/d_/'out'/f'{sp}.{src}.qc.json'; db=P/d_/'out'/f'{sp}.{src}.c0.db'
    d=json.load(open(q)); x=d['qc']; h=hashlib.sha256()
    with open(db,'rb') as f:
        for b in iter(lambda:f.read(16<<20),b''): h.update(b)
    gates=sum(v for k,v in x.items() if k.startswith('gate_') and k not in ('gate_heldout_overlapping','gate_heldout_excluded'))
    print('C0',d_,d['status'],'sha_ok',h.hexdigest()==d['output_sha256'],d['output_sha256'][:12],'leak',x.get('leakage_nonzero'),'gates',gates,
          'exh',x.get('exhaustiveness'),'audit_n',x.get('audit_n'),'reads_in',x.get('reads_in'),'accepted',x.get('accepted_reads'),'hsp_rows',x.get('hsp_rows'))
