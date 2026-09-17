"""Preparatory, pure conflict resolution for small, already validated scopes.

NOT a projector, route-A parser, raster writer, acceptance gate, A46 approval,
or full-A11 certification. c0 is coding-direction codon ordinal, including
real zero; absence is no observation. Inputs must already reflect exact strand,
intron and frameshift geometry. Nothing here certifies that geometry.

Caller must supply ALL projected bases of ALL accepted HSPs for each included
member/placement (including overlapping competitors), and all relevant members
of included molecules. A truncated genomic window cannot certify whole-HSP
consistency: a conflict outside it rejects the HSP inside it too. Run resolution
on complete scopes BEFORE slicing. Complete route-A crosschecks are also the
caller's responsibility; A_missing means absent from this input, not evidence
that no route-A alignment exists. The explicit attestations are preconditions,
not independently checked completeness proofs.

IDs must be canonical and namespaced upstream. reference_sha256 identifies the
reference sequence/dictionary; protein_sha256 identifies the normalized protein
sequence, not an accession. hsp_sha256 identifies the FULL source HSP identity
(source/search provenance, screened member identity, subject sequence identity,
raw endpoints/frame and full traceback), never a window or constant-frame
segment. This module validates digest syntax and identity consistency, not hash
preimages. Member/placement/molecule IDs must come from validated provenance,
primary/unambiguous placement selection and molecule closure, not inferred here.
"""
from collections import defaultdict
from dataclasses import dataclass, field
import re
from typing import Dict, FrozenSet, Iterable, Optional, Tuple


@dataclass(frozen=True)
class Observation:
    species: str
    reference_sha256: str
    contig: str
    pos0: int
    strand: str
    c0: int
    route: str
    protein_sha256: str
    hsp_sha256: Optional[str] = None
    member_id: Optional[str] = None
    placement_id: Optional[str] = None
    molecule_id: Optional[str] = None
    source_id: str = 'unspecified'


@dataclass(frozen=True)
class Resolution:
    """Sparse counts and reason sets; exact duplicate records collapse.

    counts key: (species, reference, contig, pos0, strand, c0).
    hsps key: (species, reference, member, placement, full-HSP digest).
    molecules key: (species, reference, contig, pos0, molecule).
    HSP/molecule summaries UNION observation reasons (partial retention is
    therefore explicit); only observation-level 'retained' contributes counts.
    A_agrees denotes a phase crosscheck only, never same-codon correspondence.
    """
    counts: Dict[Tuple, int]
    observations: Dict[Observation, FrozenSet[str]]
    hsps: Dict[Tuple, FrozenSet[str]]
    molecules: Dict[Tuple, FrozenSet[str]]
    training_eligible: bool = field(default=False, init=False)


def _validate(o):
    if type(o) is not Observation:
        raise ValueError('expected Observation record')
    ids = ('species', 'contig', 'source_id')
    hashes = ('reference_sha256', 'protein_sha256')
    if type(o.route) is not str or o.route not in ('A', 'B'):
        raise ValueError('route must be A or B')
    if o.route == 'B':
        ids += ('member_id', 'placement_id', 'molecule_id')
        hashes += ('hsp_sha256',)
    elif any(getattr(o, k) is not None for k in
             ('hsp_sha256', 'member_id', 'placement_id', 'molecule_id')):
        raise ValueError('route A is a crosscheck, not an EST/HSP molecule')
    for k in ids:
        value = getattr(o, k)
        if type(value) is not str or not value or any(c.isspace() or not c.isprintable() for c in value):
            raise ValueError(k + ' must be a nonempty canonical ID')
    for k in hashes:
        value = getattr(o, k)
        if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
            raise ValueError(k + ' must be a lowercase SHA256 digest')
    if type(o.pos0) is not int or o.pos0 < 0:
        raise ValueError('pos0 must be a nonnegative integer')
    if type(o.c0) is not int or o.c0 not in (0, 1, 2):
        raise ValueError('c0 must be integer 0, 1 or 2, never absence/bool')
    if type(o.strand) is not str or o.strand not in ('+', '-'):
        raise ValueError('strand must be + or -')


def _base(o):
    return (o.species, o.reference_sha256, o.contig, o.pos0)


def _hsp(o):
    return (o.species, o.reference_sha256, o.member_id, o.placement_id, o.hsp_sha256)


def resolve_conflicts(observations: Iterable[Observation], *, complete_hsps: bool,
                      primary_unambiguous: bool) -> Resolution:
    """Resolve simultaneous HSP conflicts, then A/B pairs, then molecules.

    No winner selection, MAPQ threshold, source summation or coverage cap.
    Multiple placements for a member are explicitly unresolved, not merged or
    silently selected. Malformed records or inconsistent identity reuse raise
    ValueError. Unknown completeness/placement eligibility fails closed.
    Memory scales with supplied observations; use bounded COMPLETE scopes, not
    chromosome rasters. Missing cells mean zero retained molecules, not c0=0.
    """
    if complete_hsps is not True or primary_unambiguous is not True:
        raise ValueError('complete-HSP and primary/unambiguous scope attestations required')
    rows = set()
    for o in observations:
        _validate(o)
        rows.add(o)
    reasons = {o: set() for o in rows}
    identities, members = {}, {}
    placements = defaultdict(set)
    overlaps = defaultdict(list)
    for o in rows:
        if o.route != 'B':
            continue
        domain = (o.species, o.reference_sha256)
        identity = domain + (o.hsp_sha256,)
        metadata = (o.member_id, o.molecule_id, o.protein_sha256)
        if identities.setdefault(identity, metadata) != metadata:
            raise ValueError('full-HSP identity reused with inconsistent metadata')
        member = domain + (o.member_id,)
        if members.setdefault(member, o.molecule_id) != o.molecule_id:
            raise ValueError('member belongs to inconsistent molecules')
        placements[member].add(o.placement_id)
        overlaps[_base(o) + (o.member_id, o.placement_id)].append(o)
    # Collect all direct participants before removing any, including self and
    # chains. Every label at a multilabel base conflicts with some other label.
    rejected = set()
    for group in overlaps.values():
        if len({(o.strand, o.c0) for o in group}) > 1:
            rejected.update(_hsp(o) for o in group)
    for o in rows:
        if o.route == 'B':
            if _hsp(o) in rejected:
                reasons[o].add('HSP_conflict')
            if len(placements[(o.species, o.reference_sha256, o.member_id)]) > 1:
                reasons[o].add('placement_unresolved')

    comparable = defaultdict(list)
    for o in rows:
        if not reasons[o]:
            comparable[_base(o) + (o.strand, o.protein_sha256)].append(o)
    crosscheck = {}
    for group in comparable.values():
        phases = {route: {o.c0 for o in group if o.route == route} for route in ('A', 'B')}
        for o in group:
            opposite = phases['A' if o.route == 'B' else 'B']
            if opposite - {o.c0}:
                reasons[o].add('AB_conflict')
            elif o.route == 'B':
                crosscheck[o] = 'A_agrees' if opposite else 'A_missing'

    molecule_bases = defaultdict(list)
    for o in rows:
        if o.route == 'B' and not reasons[o]:
            molecule_bases[_base(o) + (o.molecule_id,)].append(o)
    for group in molecule_bases.values():
        if len({(o.strand, o.c0) for o in group}) > 1:
            for o in group:
                reasons[o].add('molecule_base_unresolved')

    support = defaultdict(set)
    hsps, molecules = defaultdict(set), defaultdict(set)
    for o in rows:
        if not reasons[o]:
            reasons[o].add('retained' if o.route == 'B' else 'crosscheck_only')
            if o.route == 'B':
                reasons[o].add(crosscheck[o])
                support[_base(o) + (o.strand, o.c0)].add(o.molecule_id)
        if o.route == 'B':
            hsps[_hsp(o)].update(reasons[o])
            molecules[_base(o) + (o.molecule_id,)].update(reasons[o])
    return Resolution({key: len(value) for key, value in support.items()},
                      {o: frozenset(value) for o, value in reasons.items()},
                      {key: frozenset(value) for key, value in hsps.items()},
                      {key: frozenset(value) for key, value in molecules.items()})
