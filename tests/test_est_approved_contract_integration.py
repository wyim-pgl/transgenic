"""Small synthetic API connection, NOT real BAM/C0/route-A certification."""
import hashlib

import pytest

from revision.scripts.evidence.est_hsp_contract import adapt_raw13, classify_rich17
from revision.scripts.evidence.est_hsp_expanded import rc
from revision.scripts.evidence.est_molecule_closure import Member, Placement, Audit, close_molecules
from revision.scripts.evidence.est_phase_project import LABEL_CONVENTION, project_codons
from revision.scripts.evidence.est_phase_conflicts import Observation, resolve_conflicts


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize('frame', [1, -1])
@pytest.mark.parametrize('flag', [0, 16])
def test_validated_hsp_through_splice_geometry_and_a11_identity(frame, flag):
    dna = 'ATG' * 40
    if frame < 0:
        dna = rc(dna)
    raw = ['member', '120', 'protein', '40',
           '1' if frame > 0 else '120', '120' if frame > 0 else '1',
           '1', '40', '40', '1e-20', '100', str(frame), '40']
    accepted = classify_rich17(adapt_raw13(raw, dna, 'M' * 40), dna, 'M' * 40)
    assert accepted['accepted_by_hsp_rule']
    identity = dict(member='member', source_sha256=sha('synthetic-source'),
                    sequence_sha256=sha(dna), source_start0=0, source_end0=120)
    hsp = dict(id='hsp', identity=identity, producer_contract_sha256=sha(accepted['length_contract']),
               label_convention=LABEL_CONVENTION, qstart=int(raw[4]), qend=int(raw[5]), frame=frame)
    placement = dict(identity=identity, sequence=dna, flag=flag, pos1=101,
                     contig='chr', contig_length=1000, cigar='5M7N115M',
                     ts='+' if frame > 0 else '-')
    projected = project_codons(hsp, accepted['codon_events'], placement)
    assert not any(x.endswith('_unresolved') for x in projected['reasons'])
    strand = '+' if (frame > 0) != bool(flag & 16) else '-'
    member = Member('member', 'sp', 'assembly', 'dataset', 'primary', 'ACC.1',
                    'library', 'clone', 'known', 'ACC.1', 120, (0, 120), sha(dna),
                    'synthetic-provenance', 'bioproject', 'unknown', 'train_track', True, 'report')
    closed = close_molecules(species='sp', assembly='assembly', members=[member],
        placements=[Placement('placement', 'member', 'chr', strand, 100, 227,
                              ((100, 105), (112, 227)), 'primary', 60)],
        relations=[], audits=[Audit('member', 1, 0, True, 'synthetic-audit')],
        exclusions=[], library_aliases={'library': 'library'}, inventory_complete=True,
        identity_complete=True, exclusions_complete=True, context='tile',
        policy='A11_A21_A46_normalized_v1')
    assert closed['diagnostic_ready'] and not closed['training_eligible']
    assert closed['components'][0]['training_support_members'] == ('member',)
    molecule = ':'.join(closed['components'][0]['id'])
    observations = []
    for segment in projected['segments']:
        for pos in range(segment['start0'], segment['end0']):
            phase = (segment['c0'] + segment['step'] * (pos-segment['start0'])) % 3
            observations.append(Observation('sp', sha('synthetic-reference'), 'chr', pos,
                segment['strand'], phase, 'B', sha('M'*40), sha('\t'.join(raw)),
                'member', 'placement', molecule, 'synthetic'))
    result = resolve_conflicts(observations, complete_hsps=True, primary_unambiguous=True)
    expected = {('sp', sha('synthetic-reference'), 'chr', 100+u+(7 if u >= 5 else 0),
                 strand, (u if strand == '+' else 119-u) % 3): 1 for u in range(120)}
    assert result.counts == expected
    assert not result.training_eligible
    assert all('A_missing' in reasons for reasons in result.observations.values())
