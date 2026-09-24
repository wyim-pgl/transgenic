"""Retained route-A trace fixture for Sitalica MP000002, fail-closed.

Every value below was transcribed from receipts already in this repository:
`notes/power_recovery_20260917/phase_input_inventory.md` (raw GFF/PAF fields,
candidate table) and `notes/auth_resume_20260917/est/INVENTORY.md` (raw,
reference, protein, chunk, producer and acceptance receipts). No raw alignment,
reference or protein byte was read to build it, and no field was invented.

This is a REAL trace fixture in the only sense currently available: exact
recorded producer output fields. It is NOT a certified route-A observation.
`est_phase_conflicts.Observation` needs a reference-sequence SHA256, a
NORMALIZED protein-sequence SHA256 and a coding-direction codon ordinal, and
none of the three is recorded anywhere locally:

  * the reference receipt carries an MD5 only;
  * the protein receipt hashes the whole filtered FASTA, not this protein;
  * raw GFF column 8 is miniprot's CDS phase, not the codon ordinal `c0`, so
    `c0` exists only after replaying the alignment against the exact reference
    and protein sequences.

The acceptance verdict and the mapping-ambiguity membership of this alignment
ID were also never inspected, and an `accepted` verdict alone does not show
A43-retained support. `to_observation` therefore refuses every one of those
substitutions and names the gaps instead of guessing. Closing them requires the
recorded artifacts, not a new miniprot run.
"""
import importlib.util
import re
import sys
from pathlib import Path

__all__ = ['RouteAFixtureError', 'load', 'is_certified', 'to_observation', 'conflict_api',
           'BLOCKING_GAPS']

BLOCKING_GAPS = (
    'reference_sha256_unrecorded',
    'normalized_protein_sha256_unrecorded',
    'codon_ordinal_requires_reference_protein_replay',
    'acceptance_verdict_uninspected',
    'mapping_ambiguity_membership_uninspected',
)

_SHA256 = re.compile('[0-9a-f]{64}')


class RouteAFixtureError(ValueError):
    pass


def conflict_api():
    """Load the conflict API beside this file without assuming a package import."""
    name = 'est_phase_conflicts'
    module = sys.modules.get(name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).resolve().parent / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


def _fixture():
    return {
        'alignment_id': 'MP000002',
        'species': 'Sitalica',
        'contig': 'scaffold_7',
        'strand': '-',
        'protein': '1000413_0:000002',
        'evidence': (
            'notes/power_recovery_20260917/phase_input_inventory.md:74',
            'notes/auth_resume_20260917/est/INVENTORY.md',
        ),
        'raw': {
            'path': 'evidence/protein_alignments/Sitalica/Sitalica.protein.gff.gz',
            'bytes': 9647215761,
            'md5': '0e17de830f3458c81b25b3366320ec77',
            'sha256': None,
            'gff_lines': [5, 8],
        },
        'paf': {'protein_length': 172, 'query_interval0': [0, 160]},
        # Raw GFF coordinates are 1-based inclusive, listed in coding order.
        'cds': [
            {'start1': 35312723, 'end1': 35312847, 'phase': 0, 'target': [1, 41],
             'interval0': [35312722, 35312847]},
            {'start1': 35312121, 'end1': 35312451, 'phase': 1, 'target': [42, 160],
             'interval0': [35312120, 35312451]},
        ],
        'reference': {
            'path': 'genomes/Sitalica_312_v2.fa',
            'md5': 'd8f77d532d8c87f314f6175146fa6728',
            'sha256': None,
        },
        'protein_source': {
            'path': 'evidence/protein/odb12_Viridiplantae.filtered.fa.gz',
            'md5': '453cb32b02e0799950d7d5f4de5f62ac',
            'sha256': 'fa0795b60ea8460f0f8956ca91ec64914d96e6689dd8441d0be98ce448e1c25f',
        },
        'protein_sequence_sha256': None,
        'chunk': {'path': 'evidence/protein/chunks_v1/chunk0.faa.gz',
                  'md5': 'e8c6980e0ca2611a7e6c721a49253299', 'proteins': 2019181},
        'producer': {
            'tool': 'miniprot',
            'version': '0.18-r281',
            'job': '6153195_42',
            'executable': '/data/gpfs/assoc/pgl/bin/conda/conda_envs/busco5/bin/miniprot',
            'package_record_sha256':
                'f8822f41eceb53a6ca611bd6606faf035aeb913030d1dede7fd7889ab37d4f52',
            'argv': ['-t', '32', '--gff', '--outs=0.97', '-N', '20',
                     'Sitalica.mpi', 'chunk0.faa.gz'],
        },
        'acceptance': {
            'table': 'evidence/protein_alignments/Sitalica/accept/Sitalica.alignments.tsv',
            'ambiguity_list':
                'evidence/protein_alignments/Sitalica/accept/Sitalica.mapping_ambiguous.txt',
            'support_version': 2,
            'support_excludes_mapping_ambiguous': True,
            'verdict_for_this_id': None,
            'ambiguity_membership_for_this_protein': None,
        },
        'blocking_gaps': list(BLOCKING_GAPS),
        'certified': False,
    }


def load():
    """Return a fresh copy of the retained fixture."""
    return _fixture()


def is_certified(fixture):
    return bool(fixture.get('certified')) and not fixture.get('blocking_gaps')


def _require(ok, message):
    if not ok:
        raise RouteAFixtureError(message)


def to_observation(fixture, *, reference_sha256=None, protein_sha256=None, c0=None,
                   replay_provenance=None, c0_from_gff_phase=False, block=0):
    """Build a route-A crosscheck Observation, or refuse and name the gaps.

    The three closing inputs must be supplied by a caller that actually recovered
    them: a reference-sequence SHA256, this protein's normalized sequence SHA256
    and a replayed coding-direction codon ordinal with its replay provenance.
    """
    _require(not c0_from_gff_phase,
             'codon_ordinal_requires_reference_protein_replay: raw GFF phase is not c0')
    missing = []
    if reference_sha256 is None:
        missing.append('reference_sha256_unrecorded')
    if protein_sha256 is None:
        missing.append('normalized_protein_sha256_unrecorded')
    if c0 is None:
        missing.append('codon_ordinal_requires_reference_protein_replay')
    _require(not missing, 'route-A fixture is not closed: ' + ', '.join(missing))

    _require(reference_sha256 != fixture['reference']['md5'] and
             _SHA256.fullmatch(reference_sha256) is not None,
             'reference_sha256 must be a SHA256 of the reference sequence, not the recorded MD5')
    _require(protein_sha256 != fixture['protein_source']['sha256'] and
             _SHA256.fullmatch(protein_sha256) is not None,
             'protein_sha256 must be the normalized protein sequence hash, '
             'not the whole protein-FASTA hash')
    _require(isinstance(replay_provenance, str) and bool(replay_provenance),
             'replay provenance required for the codon ordinal')
    _require(isinstance(c0, int) and not isinstance(c0, bool) and c0 in (0, 1, 2),
             'c0 must be integer 0, 1 or 2')

    Observation = conflict_api().Observation
    cds = fixture['cds'][block]
    return Observation(
        species=fixture['species'], reference_sha256=reference_sha256,
        contig=fixture['contig'], pos0=cds['interval0'][0], strand=fixture['strand'],
        c0=c0, route='A', protein_sha256=protein_sha256,
        source_id='miniprot:%s:%s:%s' % (fixture['producer']['version'],
                                         fixture['producer']['job'],
                                         fixture['alignment_id']))
