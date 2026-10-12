import hashlib, importlib.util, json, sys, collections
from pathlib import Path
ROOT = Path('/mnt/f/scratch/develop/Transgenic/transgenic_local/notes/seed123_recovery_20260917_1400')
SRC = Path('/tmp/est-real-source-binding-20260917')
OUT = ROOT/'producer_length_fix_review_evidence'
def digest(b): return hashlib.sha256(b).hexdigest()
spec = importlib.util.spec_from_file_location('retained_review', SRC/'tasknotes/real_sample/replay_retained.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
b = m.binder
paths = [ROOT/'est_result/raw/worker/sample/report.json', ROOT/'est_result/raw/worker/sample/diagnostics.jsonl', ROOT/'est_failure_diagnosis/producer_path_proof.json']
parent, diag, proof = [p.read_bytes() for p in paths]
before = {str(p):digest(p.read_bytes()) for p in paths}
registry = json.loads(proof)
assert digest(proof) == b.PRODUCER_PATH_PROOF_SHA256
resolved = [b.resolve_producer_path(proof, r['search_preimage']) for r in registry['producers']]
checks = 0
for r in registry['producers']:
    pre = r['search_preimage']
    for bad in (None, b'', proof+b' ', proof[:-1], bytearray(proof), b'x'*(b.PRODUCER_PATH_PROOF_MAX_BYTES+1)):
        try: b.resolve_producer_path(bad, pre)
        except ValueError: checks += 1
        else: raise AssertionError('invalid proof accepted')
    for field in pre:
        for op in ('change','remove'):
            altered = dict(pre)
            if op == 'change': altered[field] += 'x'
            else: del altered[field]
            try: b.resolve_producer_path(proof, altered)
            except ValueError: checks += 1
            else: raise AssertionError('wrong preimage accepted')
    altered = dict(pre, producer_path_proof_sha256=digest(proof))
    try: b.resolve_producer_path(proof, altered)
    except ValueError: checks += 1
    else: raise AssertionError('extra preimage accepted')
result = m.replay(parent, diag, proof)
new_bytes = b.canonical(result)+b'\n'
(OUT/'replay.json').write_bytes(new_bytes)
old = {r['id']:r for r in map(json.loads, diag.splitlines()) if r['type']=='hsp'}
newly_accepted = [h for h in result['hsps'] if h['length_accepted'] is True and old[h['parent_hsp_id']].get('length_accepted') is not True]
new_geometry_fail = [g for h in newly_accepted for g in h['geometry'] if not g['geometry_success']]
for s in result['searches']:
    bnd=s['new']; assert b.bound_id('search', bnd['preimage']) == bnd
    assert all(bnd['preimage'][k]==v for k,v in s['resolved'].items())
searches = {s['new']['id']:s for s in result['searches']}
for h in result['hsps']:
    assert h['preimage']['search_id']==h['search_id']
    assert b.bound_id('hsp',h['preimage'])['id']==h['id']
    assert all(h[k]==v for k,v in searches[h['search_id']]['resolved'].items())
    if 'replay' in h: assert h['replay']['length_contract']==h['length_contract']
    for g in h['geometry']:
        if 'result' in g:
            assert g['result']['hsp_id']==h['id']
            assert g['result']['producer_contract_sha256']==m.SOURCE_IDENTITIES['revision/scripts/evidence/est_hsp_contract.py']
for key in ('training_eligible','full_c0','molecule_counts_complete','raster','route_a_complete','original_inputs_reopened','historical_result_reclassified'):
    assert result[key] is False
assert before == {str(p):digest(p.read_bytes()) for p in paths}
files = ['revision/scripts/evidence/est_hsp_contract.py','revision/scripts/evidence/est_real_sample.py','tests/test_est_hsp_contract.py','tests/test_est_real_sample.py','tests/test_est_phase_conflicts.py','tests/test_est_approved_contract_integration.py','tasknotes/real_sample/replay_retained.py']
hashes = {f:digest((SRC/f).read_bytes()) for f in files}
summary = dict(python=sys.version, source_hashes=hashes, protected_inputs_unchanged=before, proof_size=len(proof), resolved=resolved, adversarial_rejections=checks, replay_sha256=digest(new_bytes), counts=result['counts'], newly_accepted=len(newly_accepted), newly_accepted_geometry_failures=len(new_geometry_fail), newly_accepted_geometry_failure_reasons=dict(collections.Counter(reason for g in new_geometry_fail for reason in g.get('result',{}).get('reasons',[]) if reason.endswith('_unresolved'))))
(OUT/'verification.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
print(json.dumps(summary,indent=2,sort_keys=True))
