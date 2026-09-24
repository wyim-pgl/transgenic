"""Route-A retained-trace fixture: real recorded evidence, fail-closed gaps.

The fixture carries only fields actually recorded in the retained receipts
(notes/power_recovery_20260917/phase_input_inventory.md and
notes/auth_resume_20260917/est/INVENTORY.md). It is NOT a certified route-A
observation: the reference SHA256, the normalized protein SHA256 and the
replayed codon ordinal are missing, and the acceptance/ambiguity membership of
this alignment ID was never inspected. These tests pin that the loader refuses
every substitution that would paper over those gaps.
"""
import importlib.util
from functools import lru_cache
from pathlib import Path
import sys

import pytest

EVIDENCE = Path(__file__).parents[1] / 'evidence'
PATH = EVIDENCE / 'est_routea_fixture.py'


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(None)
def api():
    assert PATH.exists(), 'route-A retained-trace fixture loader not implemented'
    return _load(PATH, 'est_routea_fixture')


def conflicts():
    # Same module instance the loader uses; a second load would make the
    # Observation dataclass a different type.
    return api().conflict_api()


def test_fixture_carries_the_exact_retained_raw_trace():
    f = api().load()
    assert f['alignment_id'] == 'MP000002'
    assert f['species'] == 'Sitalica'
    assert f['contig'] == 'scaffold_7'
    assert f['strand'] == '-'
    assert f['protein'] == '1000413_0:000002'
    assert f['paf']['protein_length'] == 172
    assert f['paf']['query_interval0'] == [0, 160]
    # Raw GFF is 1-based inclusive; coding-order CDS blocks with their phases.
    assert f['cds'] == [
        {'start1': 35312723, 'end1': 35312847, 'phase': 0, 'target': [1, 41],
         'interval0': [35312722, 35312847]},
        {'start1': 35312121, 'end1': 35312451, 'phase': 1, 'target': [42, 160],
         'interval0': [35312120, 35312451]},
    ]
    assert f['producer']['tool'] == 'miniprot' and f['producer']['version'] == '0.18-r281'
    assert f['producer']['job'] == '6153195_42'


def test_recorded_digests_are_labelled_by_what_they_actually_identify():
    f = api().load()
    assert f['raw']['md5'] == '0e17de830f3458c81b25b3366320ec77'
    assert f['reference']['md5'] == 'd8f77d532d8c87f314f6175146fa6728'
    assert f['reference']['sha256'] is None
    # A whole-FASTA hash is not this protein's normalized sequence hash.
    assert f['protein_source']['sha256'] == \
        'fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f'
    assert f['protein_sequence_sha256'] is None


def test_fixture_is_not_certified_and_names_every_blocking_gap():
    m = api()
    f = m.load()
    assert m.is_certified(f) is False
    assert set(f['blocking_gaps']) == {
        'reference_sha256_unrecorded',
        'normalized_protein_sha256_unrecorded',
        'codon_ordinal_requires_reference_protein_replay',
        'acceptance_verdict_uninspected',
        'mapping_ambiguity_membership_uninspected',
    }


def test_observation_build_refuses_until_the_gaps_are_closed():
    m = api()
    with pytest.raises(m.RouteAFixtureError) as excinfo:
        m.to_observation(m.load())
    message = str(excinfo.value)
    for gap in ('reference_sha256_unrecorded', 'codon_ordinal_requires_reference_protein_replay'):
        assert gap in message


@pytest.mark.parametrize('substitution', [
    {'reference_sha256': 'd8f77d532d8c87f314f6175146fa6728'},
    {'reference_sha256': 'e' * 64,
     'protein_sha256': 'fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f'},
    {'reference_sha256': 'e' * 64, 'protein_sha256': 'f' * 64, 'c0_from_gff_phase': True},
])
def test_refuses_md5_source_file_hash_and_raw_gff_phase_substitutions(substitution):
    m = api()
    with pytest.raises(m.RouteAFixtureError):
        m.to_observation(m.load(), **substitution)


def test_closed_fixture_builds_a_valid_route_a_crosscheck_observation():
    m = api()
    observation = m.to_observation(
        m.load(), reference_sha256='e' * 64, protein_sha256='f' * 64,
        c0=1, replay_provenance='replay:bounded-test-only')
    assert observation.route == 'A'
    assert (observation.contig, observation.strand, observation.c0) == ('scaffold_7', '-', 1)
    assert observation.pos0 == 35312722
    assert observation.hsp_sha256 is None and observation.member_id is None
    conflicts()._validate(observation)
