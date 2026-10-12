"""Local metadata-only candidate freeze. Does not import source or submit work."""
import ast, hashlib, json
from pathlib import Path
H=Path(__file__).resolve().parent
N=H.parents[1]
S=Path('/tmp/mvp-seed123-integration-20260917')
D='/data/gpfs/assoc/pgl/data/Transgenic/evidence/seed123_panel_compat_auth_resume_20260917'
def ident(p):
 b=p.read_bytes(); return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),size=len(b))
def save(name,obj):
 with (H/name).open('x') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n')
M=N/'power_recovery_20260917/panel_arrays/copied/panel/manifest.json'
assert ident(M)['sha256']=='7528c9bff5a9a9b177b1ad333ad76525a1d02d3c4cde4e0550865ff12e0acc0e'
m=json.loads(M.read_text())
old=N/'power_recovery_20260917/panel_result/copied/outputs/job-6165628/request.json'
assert (ident(old)['sha256'],ident(old)['size'])==(m['inputs'][-1]['sha256'],m['inputs'][-1]['size'])
r=json.loads(old.read_text()); r.update(profile='development-single',seed=123,aug_seed=20261037,panel_inputs=str(Path(m['inputs'][-1]['path']).parent/'panel'))
save('candidate_request.json',r)
pin=ident(H/'candidate_request.json'); pin['path']=D+'/request.json'
save('request_pin.json',dict(status='CANDIDATE_REQUIRES_INDEPENDENT_CONTROLLER_ACCEPTANCE',expected_verification_request=pin,local_bytes=ident(H/'candidate_request.json')))
rels=['revision/scripts/mvp/'+f for f in ['mvp_panel_compatibility.py','mvp_verify_panel_inputs.py','mvp_prepare_panel_inputs.py','mvp_produce_counts.py','mvp_decode_panel.py','mvp_decode_runtime.py']]+['src/transgenic/evidence/'+f for f in ['mvp_tracks.py','mvp_windows.py','mvp_decode.py']]+['src/transgenic/training/'+f for f in ['mvp_cli.py','mvp_cli_runtime.py','mvp_runtime.py','mvp_profiles.py']]
closure=[]
for rel in rels:
 x=ident(S/rel); x.update(relative_path=rel,proposed_consumer_path=D+'/source/'+rel); closure.append(x)
wrapper=ident(S/'revision/scripts/mvp/mvp_verify_pinned_panel.py')
assert wrapper['sha256']=='307749d202a0facdce86880cd1dad07c3da4e291099e5d2c99203c88f7b4dab2'
assert closure[0]['sha256']=='ffc67337496cae5c2b3cdcf3bfb708bac3d7d74f09ec96f4b618fc5c3e0a0aa2'
save('consumer_closure_candidate.json',dict(status='LOCAL_BYTE_SNAPSHOT_NOT_DEPLOYED',source_root=str(S),proposed_consumer_root=D+'/source',consumer_closure=closure,launcher_separate=wrapper,warning='Final consumer must use identical canonical source paths and bytes. Packaging requires complete import tree and frozen Git authority, beyond this report closure.'))
save('historical_inputs.json',dict(manifest_pin=ident(M),canonical_manifest=r['panel_inputs']+'/manifest.json',inputs=m['inputs'],historical_packages=m['packages'],authority=m['authority']))
save('candidate_config.json',dict(status='BLOCKED_NOT_FOR_SUBMISSION',namespace=D,request=pin,receipt=D+'/output/compatibility.json',consumer_root=D+'/source',runtime=dict(sif='/work/nvme/bilv/wcyim/containers/transgenic-ngc-26.08.sif',sif_sha256='1a5dee27b8575efde88774a28559727156843eb057db8723c6d73f779ef7dff6',sif_size=11120398336,image_architecture='aarch64',python=None,numpy=None,duckdb='1.5.5',torch='2.14.0a0+4fdf77b940.nv26.08',versions_status='DuckDB/Torch from prior build record; Python/NumPy await intended consumer metadata'),budget_request=dict(cpus=4,memory_gib=16,wall_minutes=60,max_cpu_hours=4,gpus=0,requeue=False,approved=False),environment=dict(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1'),wrapper_argv=['python','-B',D+'/source/revision/scripts/mvp/mvp_verify_pinned_panel.py','--input',pin['path'],'--expected-sha256',pin['sha256'],'--expected-size',str(pin['size']),'--out',D+'/output/compatibility.json']))
print(json.dumps(pin,indent=2))
