"""Readonly adversarial review; assertions describe observed pre-fix behavior."""
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path('/tmp/est-approved-contract-20260917')
p = ROOT / 'revision/scripts/tests/test_est_molecule_closure.py'
spec = importlib.util.spec_from_file_location('a11_review_fixture', p)
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)
m = t.api()

def brief(r):
    return {k:r[k] for k in ('diagnostic_ready','blockers','components','edges','exhaustiveness')}

out = {}
rows = [t.member(m, 'old', accession='X.1', original_accession='X.1', clone='clone-x'),
        t.member(m, 'new', accession='X.2', original_accession='X.2', clone='clone-x', library='lib2')]
libs = {'lib':'lib', 'lib2':'lib2'}
ps = [t.placement(m, 'old'), t.placement(m, 'new')]
r = t.run(m, rows, ps, library_aliases=libs)
assert r['diagnostic_ready'] and len(r['components']) == 2
assert all(c['ready'] and len(c['training_support_members']) == 1 for c in r['components'])
out['contradictory_libraries_two_ready_versions'] = brief(r)
late = m.Placement('old:excluded-secondary', 'old', 'chr9', '-', 100, 200, ((100,200),), 'secondary', 30)
e = m.Exclusion('held', 'heldout', 'global', None, 'chr9', 150, 160)
r = t.run(m, rows, ps+[late], library_aliases=libs, exclusions=[e])
assert r['diagnostic_ready']
by_uid = {c['members'][0]:c for c in r['components']}
assert by_uid['old']['disposition'] == 'excluded'
assert by_uid['new']['ready'] and by_uid['new']['training_support_members'] == ('new',)
assert m.recheck_retained(r, exclusions=[e], context='tile1')
out['old_exclusion_fails_to_propagate_and_rejoin_passes'] = brief(r)
# Different download datasets do not remove accession identity.
r = t.run(m, [rows[0], dataclasses.replace(rows[1],dataset='other-download')], ps+[late],
          library_aliases=libs, exclusions=[e])
assert any(c['ready'] and c['members'] == ('new',) for c in r['components'])
out['cross_dataset_same_bug'] = brief(r)
# A real library alias is not a conflict and correctly restores closure.
r = t.run(m, rows, ps+[late], library_aliases={'lib':'lib','lib2':'lib'}, exclusions=[e])
assert len(r['components']) == 1 and r['components'][0]['disposition'] == 'excluded'
assert r['components'][0]['support_members'] == {'primary':('new',)}
out['alias_control_propagates'] = brief(r)
# An explicit relation catches the conflicting-library case, but is not required
# by the auto-version API; requiring callers to duplicate it would hide the defect.
try:
    t.run(m, rows, ps, library_aliases=libs, relations=[m.Relation('old','new','version','R')])
except ValueError as exc:
    out['explicit_version_control_rejects'] = str(exc)
else:
    raise AssertionError('explicit contradiction was not rejected')
# Distinct accessions from genuine different libraries must stay separate.
r = t.run(m, [rows[0], dataclasses.replace(rows[1],accession='Y.2',original_accession='Y.2')],
          ps, library_aliases=libs)
assert len(r['components']) == 2 and all(c['ready'] for c in r['components'])
out['genuinely_distinct_libraries_control'] = brief(r)
# No implicit known/fallback signature mixing.
a = t.member(m, clone='known')
b = t.member(m,'b',clone='',clone_status='clone_unavailable')
r = t.run(m,[a,b])
assert len(r['components']) == 2
out['known_and_fallback_not_signature_merged'] = brief(r)
# Caller-asserted explicit parent can link known/unavailable identities. It is
# not evidence attestation; absent provenance or uninspected status never readies.
r = t.run(m,[a,b],relations=[m.Relation('a','b','parent','caller-parent-proof')])
assert len(r['components']) == 1 and r['diagnostic_ready']
out['explicit_parent_known_unavailable_is_trusted'] = brief(r)
for changed in [dataclasses.replace(b,clone_status='uninspected'),dataclasses.replace(b,library='unknown')]:
    if changed.library == 'unknown':
        r = t.run(m,[a,changed])
    else:
        r = t.run(m,[a,changed],relations=[m.Relation('a','b','parent','caller-parent-proof')])
    assert not r['diagnostic_ready'] and not any(c['ready'] for c in r['components'])
    out['whole_input_block_'+changed.clone_status+'_'+changed.library] = brief(r)
for label,other,provenance in [('empty_proof',b,''),('contradictory_known',t.member(m,'b'),'R')]:
    try:
        t.run(m,[a,other],relations=[m.Relation('a','b','parent',provenance)])
    except ValueError as exc:
        out[label] = str(exc)
    else:
        raise AssertionError(label+' not rejected')
# Exact MAPQ boundary and all record-order permutations.
assert t.run(m,placements=[t.placement(m,mapq=5)])['components'][0]['ready']
assert t.run(m,placements=[t.placement(m,mapq=4)])['components'][0]['disposition'] == 'excluded'
from itertools import permutations
members = [t.member(m,x,clone='physical-x') for x in ('a','b','c')]
placements = [t.placement(m,x.uid) for x in members]
relations = [m.Relation('a','b','mate','R1'),m.Relation('b','c','parent','R2')]
expected = t.run(m,members,placements,relations=relations)
for ms in permutations(members):
    for pp in permutations(placements):
        for rr in permutations(relations):
            assert t.run(m,list(ms),list(pp),relations=list(rr)) == expected
out['determinism'] = '72 permutations identical'
out['mapq_boundary'] = '5 retained; 4 excluded'
out['source_sha256'] = {str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in
                      [ROOT/'revision/scripts/evidence/est_molecule_closure.py',p,
                       ROOT/'tasknotes/a11_molecule_closure_20260917.md']}
print(json.dumps(out,indent=2,sort_keys=True))
