"""Independent readonly oracle checks. Run from reviewed worktree via PYTHONPATH."""
from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
from itertools import combinations
import random
from revision.scripts.evidence.est_phase_conflicts import Observation, resolve_conflicts


def digest(s):
    return sha256(s.encode()).hexdigest()


def b(**kw):
    d = dict(species='sp', reference_sha256=digest('ref'), contig='chr', pos0=0,
             strand='+', c0=0, route='B', protein_sha256=digest('p'),
             hsp_sha256=digest('h'), member_id='m', placement_id='primary', molecule_id='mol')
    d.update(kw)
    return Observation(**d)


def a(**kw):
    return b(route='A', hsp_sha256=None, member_id=None, placement_id=None,
             molecule_id=None, **kw)


def run(rows):
    return resolve_conflicts(rows, complete_hsps=True, primary_unambiguous=True)


def base(o):
    return o.species, o.reference_sha256, o.contig, o.pos0


def hsp(o):
    return o.species, o.reference_sha256, o.member_id, o.placement_id, o.hsp_sha256


def oracle(rows):
    rows = set(rows)
    reasons = {o: set() for o in rows}
    bad = set()
    # Pair enumeration, deliberately unlike implementation's phase-set groups.
    for x, y in combinations(rows, 2):
        if x.route == y.route == 'B' and base(x) == base(y) and (x.member_id, x.placement_id) == (y.member_id, y.placement_id) and (x.strand, x.c0) != (y.strand, y.c0):
            bad.update((hsp(x), hsp(y)))
    for x in rows:
        if x.route == 'B':
            if hsp(x) in bad:
                reasons[x].add('HSP_conflict')
            if any(y.route == 'B' and (x.species, x.reference_sha256, x.member_id) == (y.species, y.reference_sha256, y.member_id) and x.placement_id != y.placement_id for y in rows):
                reasons[x].add('placement_unresolved')
    eligible = {o for o in rows if not reasons[o]}
    for x, y in combinations(eligible, 2):
        if x.route != y.route and base(x) == base(y) and (x.strand, x.protein_sha256) == (y.strand, y.protein_sha256) and x.c0 != y.c0:
            reasons[x].add('AB_conflict')
            reasons[y].add('AB_conflict')
    remaining = {o for o in rows if o.route == 'B' and not reasons[o]}
    for x, y in combinations(remaining, 2):
        if base(x) == base(y) and x.molecule_id == y.molecule_id and (x.strand, x.c0) != (y.strand, y.c0):
            reasons[x].add('molecule_base_unresolved')
            reasons[y].add('molecule_base_unresolved')
    support = defaultdict(set)
    hsps, molecules = defaultdict(set), defaultdict(set)
    for x in rows:
        if not reasons[x]:
            if x.route == 'A':
                reasons[x].add('crosscheck_only')
            else:
                agrees = any(y.route == 'A' and base(x) == base(y) and (x.strand, x.protein_sha256, x.c0) == (y.strand, y.protein_sha256, y.c0) for y in eligible)
                reasons[x].update(('retained', 'A_agrees' if agrees else 'A_missing'))
                support[base(x) + (x.strand, x.c0)].add(x.molecule_id)
        if x.route == 'B':
            hsps[hsp(x)].update(reasons[x])
            molecules[base(x) + (x.molecule_id,)].update(reasons[x])
    freeze = lambda m: {k: frozenset(v) for k, v in m.items()}
    return {k: len(v) for k, v in support.items()}, freeze(reasons), freeze(hsps), freeze(molecules)


def check(rows):
    r = run(rows)
    assert (r.counts, r.observations, r.hsps, r.molecules) == oracle(rows)
    assert r.training_eligible is False
    return r


# A/B removal precedes molecule ambiguity; unrelated protein on SAME molecule survives.
x = b()
y = b(member_id='sibling', hsp_sha256=digest('h2'), protein_sha256=digest('p2'), c0=1)
r = check([x, y, a(c0=2)])
assert 'retained' in r.observations[y] and 'AB_conflict' in r.observations[x]
# Source labels cannot increase combined molecule counts.
r = check([x, replace(x, source_id='Swiss'), replace(x, source_id='OrthoDB')])
assert list(r.counts.values()) == [1] and len(r.observations) == 3
# Mixed A labels mask direct conflicting participants, never greedy survivors.
rows = [x, a(), a(c0=1), b(member_id='independent', molecule_id='other', hsp_sha256=digest('h3'), protein_sha256=digest('p3'))]
r = check(rows)
assert list(r.counts.values()) == [1]
assert r.observations[a()] == frozenset({'crosscheck_only'})
# Base-local A/B removal does not drop nonconflicting flank of same full HSP.
r = check([x, replace(x, pos0=100), a(c0=1)])
assert {'AB_conflict', 'retained', 'A_missing'} == set(r.hsps[hsp(x)])
# Opposite B strands on distinct molecules both survive (not global strand veto).
r = check([x, b(member_id='other', molecule_id='other', hsp_sha256=digest('opp'), strand='-')])
assert len(r.counts) == 2
# Same HSP/member digest can be independently namespaced by species/reference.
r = check([x, replace(x, species='other', c0=1), replace(x, reference_sha256=digest('otherref'), c0=2)])
assert len(r.counts) == 3

rng = random.Random(20260917)
for trial in range(1000):
    rows = []
    for h in range(8):
        member = h // 2
        domain = h % 2 if trial % 3 == 0 else 0
        protein = digest('p' + str(h % 3))
        for _ in range(rng.randrange(1, 5)):
            rows.append(b(species='sp'+str(domain), reference_sha256=digest('ref'+str(trial % 2)),
                          contig='chr'+str(rng.randrange(2)), pos0=rng.randrange(3),
                          member_id='m'+str(member), molecule_id='mol'+str(member // 2),
                          hsp_sha256=digest('h'+str(h)), protein_sha256=protein,
                          placement_id='alt' if trial % 7 == 0 and rng.randrange(3) == 0 else 'primary',
                          strand=rng.choice('+-'), c0=rng.randrange(3), source_id=rng.choice(['Swiss', 'OrthoDB'])))
    for _ in range(8):
        rows.append(a(species='sp'+str(rng.randrange(2)), reference_sha256=digest('ref'+str(trial % 2)),
                      contig='chr'+str(rng.randrange(2)), pos0=rng.randrange(3),
                      protein_sha256=digest('p'+str(rng.randrange(3))), strand=rng.choice('+-'), c0=rng.randrange(3)))
    result = check(rows)
    rng.shuffle(rows)
    assert check(rows + rows[:3]) == result
print('PASS: 6 targeted scenarios; 1000 randomized pair-enumeration oracle cases and shuffled/duplicate replay')
