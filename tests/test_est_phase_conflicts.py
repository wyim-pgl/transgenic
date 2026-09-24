"""Synthetic, already projected observations; no acceptance/geometry shortcut."""
from dataclasses import replace
from itertools import permutations
import importlib
import pytest


def api():
    # A missing implementation is an assertion failure during the initial RED run.
    import importlib.util
    assert importlib.util.find_spec('revision.scripts.evidence.est_phase_conflicts'), 'conflict library not implemented'
    return importlib.import_module('revision.scripts.evidence.est_phase_conflicts')


def obs(**kw):
    defaults = dict(species='sp', reference_sha256='a'*64, contig='chr1', pos0=10,
                    strand='+', c0=0, route='B', protein_sha256='b'*64,
                    hsp_sha256='c'*64, member_id='member', placement_id='primary', molecule_id='mol')
    defaults.update(kw)
    if defaults['route'] == 'A':
        for key in ('hsp_sha256', 'member_id', 'placement_id', 'molecule_id'):
            defaults[key] = None
    return api().Observation(**defaults)


def run(rows, **kw):
    return api().resolve_conflicts(rows, complete_hsps=True, primary_unambiguous=True, **kw)


def cell(o):
    return (o.species, o.reference_sha256, o.contig, o.pos0, o.strand, o.c0)


def test_missing_a_real_zero_duplicates_and_no_cap():
    b = obs()
    r = run([b, b])
    assert r.counts == {cell(b): 1}
    assert r.observations[b] == frozenset({'retained', 'A_missing'})
    assert r.training_eligible is False
    assert run([replace(b, member_id=f'm{i}', molecule_id=f'n{i}', hsp_sha256=f'{i:064x}') for i in range(70)]).counts == {cell(b): 70}


def test_chain_conflicts_collected_before_removal_and_whole_hsp():
    a = obs()
    b = obs(hsp_sha256='d'*64, c0=1)
    b_far = replace(b, pos0=20)
    c = obs(hsp_sha256='e'*64, pos0=20, c0=2)
    a_far = replace(a, pos0=30)
    unrelated = obs(hsp_sha256='f'*64, pos0=40)
    rows = [a, b, b_far, c, a_far, unrelated]
    r = run(rows)
    assert r.counts == {cell(unrelated): 1}
    assert all(r.observations[o] == frozenset({'HSP_conflict'}) for o in rows[:-1])
    assert len(r.hsps) == 4
    assert sum('HSP_conflict' in s for s in r.hsps.values()) == 3
    assert run(list(reversed(rows))) == r


@pytest.mark.parametrize('change', [{'c0': 2}, {'strand': '-'}])
def test_self_conflict_rejects_whole_hsp(change):
    b = obs()
    assert run([b, replace(b, **change), replace(b, pos0=999)]).counts == {}


def test_pair_local_other_protein_and_molecule_survive():
    a = obs(route='A', c0=1)
    bad = obs()
    good = obs(protein_sha256='e'*64, hsp_sha256='d'*64, member_id='other', molecule_id='other')
    a_good = obs(route='A', protein_sha256='e'*64)
    far = replace(bad, pos0=11)
    r = run([a, bad, good, a_good, far])
    assert r.counts == {cell(good): 1, cell(far): 1}
    assert r.observations[bad] == r.observations[a] == frozenset({'AB_conflict'})
    assert 'retained' in r.observations[far]
    assert r.observations[good] == frozenset({'retained', 'A_agrees'})


def test_ab_multilabel_simultaneous_not_first_match():
    a0, a1, b = obs(route='A'), obs(route='A', c0=1), obs()
    expected = run([a0, a1, b])
    assert expected.counts == {}
    assert 'AB_conflict' in expected.observations[b]
    for order in permutations([a0, a1, b]):
        assert run(order) == expected


def test_opposite_strand_a_not_comparable():
    b = obs()
    r = run([obs(route='A', strand='-', c0=2), b])
    assert r.counts == {cell(b): 1}
    assert 'A_missing' in r.observations[b]


@pytest.mark.parametrize('change', [{'c0': 1}, {'strand': '-'}])
def test_cross_member_molecule_unresolved_preserves_other_molecule(change):
    b = obs()
    sibling = obs(member_id='sibling', hsp_sha256='d'*64, **change)
    independent = obs(member_id='independent', molecule_id='independent', hsp_sha256='e'*64)
    r = run([b, sibling, independent])
    assert r.counts == {cell(b): 1}
    assert 'molecule_base_unresolved' in r.observations[b]
    assert 'molecule_base_unresolved' in r.observations[sibling]
    assert 'molecule_base_unresolved' in set().union(*r.molecules.values())


def test_dedup_proteins_members_and_hsps():
    rows = [obs(), obs(hsp_sha256='d'*64, protein_sha256='e'*64),
            obs(member_id='sibling', hsp_sha256='f'*64)]
    expected = run(rows)
    assert expected.counts == {cell(rows[0]): 1}
    for order in permutations(rows):
        assert run(order + order) == expected


@pytest.mark.parametrize('change', [{'species': 'other'}, {'reference_sha256': 'f'*64}, {'contig': 'chr2'}])
def test_domain_identity_prevents_accidental_joins(change):
    b = obs()
    other = replace(b, c0=1, **change)
    r = run([b, other, obs(route='A', c0=2, **change)])
    assert r.counts == {cell(b): 1}
    assert 'HSP_conflict' not in r.observations[b]


def test_different_placements_explicitly_unresolved_not_merged():
    b = obs()
    alternative = replace(b, placement_id='alternative', pos0=999)
    independent = obs(member_id='other', molecule_id='other', hsp_sha256='d'*64)
    r = run([b, alternative, independent])
    assert r.counts == {cell(independent): 1}
    assert r.observations[b] == r.observations[alternative] == frozenset({'placement_unresolved'})


@pytest.mark.parametrize('field,value', [('c0', True), ('c0', 3), ('c0', None), ('pos0', True),
    ('pos0', -1), ('pos0', 1.5), ('strand', '?'), ('route', 'C'), ('species', ''),
    ('species', 123), ('contig', ' x'), ('reference_sha256', 'a'), ('protein_sha256', 'G'*64),
    ('hsp_sha256', None), ('member_id', ''), ('placement_id', []), ('molecule_id', False)])
def test_malformed_observations_rejected(field, value):
    with pytest.raises(ValueError):
        run([obs(**{field: value})])


def test_malformed_records_and_identity_reuse_rejected():
    with pytest.raises(ValueError):
        run([{}])
    b = obs()
    for change in ({'protein_sha256': 'e'*64}, {'molecule_id': 'other'}, {'member_id': 'other'}):
        with pytest.raises(ValueError):
            run([b, replace(b, **change)])
    with pytest.raises(ValueError):
        run([b, replace(b, hsp_sha256='d'*64, molecule_id='other')])
    with pytest.raises(ValueError):
        run([replace(obs(route='A'), molecule_id='not-an-est')])


@pytest.mark.parametrize('complete,primary', [(False, True), (True, False), (1, True), (True, 1)])
def test_scope_attestations_required(complete, primary):
    with pytest.raises(ValueError):
        api().resolve_conflicts([obs()], complete_hsps=complete, primary_unambiguous=primary)


def test_empty_scope_no_counts():
    r = run([])
    assert r.counts == r.observations == r.hsps == r.molecules == {}
