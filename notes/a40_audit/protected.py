import hashlib,json,pathlib
R=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/a40_20260905_v1')
freeze=R/'source/data/freeze/b5_full_v1.freeze.json'; f=json.load(open(freeze))
def md5(p):
 h=hashlib.md5()
 with open(p,'rb') as fh:
  for b in iter(lambda:fh.read(4194304),b''): h.update(b)
 return h.hexdigest()
out={'old_database':{},'old_sources':{},'status':'RUNNING'}
p=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/merged/b5_full_v1.db')
s=p.stat(); out['old_database']={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'md5':md5(p),'expected_md5':f['file_md5']}
assert out['old_database']['md5']==f['file_md5']
assert s.st_size==23491260416 and s.st_mtime_ns==1788422329816219148
for sp,h in f['source_md5'].items():
 p=pathlib.Path('/home/pgl/scratch1/wyim/transgenic_data/b5/full')/(sp+'.db')
 actual=md5(p); out['old_sources'][sp]={'md5':actual,'expected_md5':h}; assert actual==h,sp
out['status']='PASS'; (R/'audit/protected-artifacts-v2.json').write_text(json.dumps(out,indent=2))
print('PASS: old merged DB and all source DB hashes unchanged')
